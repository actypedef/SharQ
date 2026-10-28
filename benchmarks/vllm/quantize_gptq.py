"""GPTQ weight quantization for SharQ, on the deployment grid (E2M1 codes + UE4M3 per-group scales).

Why a dedicated script: SharQ stores 4-bit E2M1 weight codes with one UE4M3 scale per **32**
elements, which is a hard constraint of CUTLASS's NVFP4 *sparse* MMA (SfVectorSize == 32 for the
sparse schedules, see cutlass/gemm/collective/builders/sm1xx_common.inl). A finer grid is therefore
not deployable, and re-projecting a finer-grid solution onto g32 throws the gain away
(measured: -0.03 PPL vs -0.19 for solving on g32 directly). So the layer-wise error compensation
has to run on the g32 grid itself.

Pipeline:  HF model -> per-layer Hessian from calibration activations -> GPTQ on the g32 grid
           -> save_sharq_checkpoint() (same packing/merging path as the RTN checkpoints).

Usage (in the `sharq` env, i.e. the one with sharq_ops built):

    python benchmarks/vllm/quantize_gptq.py \
        --model /path/to/Llama-3.1-8B \
        --calib-ids /path/to/wikitext2_train_ids.json --calib-windows 64 \
        --method gptq_opt --damp 0.1 \
        --save-sharq /path/to/quantized/sharq_gptq/Llama-3.1-8B

Reference (Llama-3.1-8B, WikiText-2 test, 141 x 2048): RTN 6.7481 -> GPTQ 6.6134.
Cost: ~10 min on one GPU, peak extra memory ~2.5 GB (Hessian + its inverse).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from quantize import quantize_activation_sharq_sim  # noqa: E402

LINEARS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
KEYS = tuple(LINEARS)

E2M1 = torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def e2m1_round(x):
    vals = E2M1.to(x.device, x.dtype)
    return vals[(x.unsqueeze(-1) - vals).abs().argmin(dim=-1)]


def ue4m3_round(x):
    """Round positive float to UE4M3 (4 exp / 3 mantissa, bias 7, subnormals, max 448)."""
    x = x.clamp(min=2.0 ** -9, max=448.0)
    sub = x < 2.0 ** -6
    q_sub = torch.round(x / 2.0 ** -9) * 2.0 ** -9
    e = torch.floor(torch.log2(x.clamp(min=2.0 ** -6)))
    m = x / torch.pow(2.0, e) - 1.0
    q_norm = (1.0 + torch.round(m * 8) / 8) * torch.pow(2.0, e)
    return torch.where(sub, q_sub, q_norm)


def q_act(x):
    xs, xr, scale = quantize_activation_sharq_sim(x)
    return ((xs.float() + xr.float()) * scale).to(x.dtype)


def group_scales(wblock, opt, mults=(0.6, 0.7, 0.8, 0.9, 1.0)):
    """wblock: (N, g), already normalised by the per-tensor global scale."""
    amax = wblock.abs().amax(dim=1, keepdim=True)
    base = (amax / 6.0).clamp(min=2.0 ** -9)
    if not opt:
        return ue4m3_round(base), None
    best_s = best_q = best_err = None
    for m in mults:
        s = ue4m3_round(base * m)
        q = e2m1_round(wblock / s) * s
        err = (q - wblock).pow(2).sum(dim=1, keepdim=True)
        if best_err is None:
            best_s, best_q, best_err = s, q, err
        else:
            take = err < best_err
            best_s = torch.where(take, s, best_s)
            best_q = torch.where(take, q, best_q)
            best_err = torch.where(take, err, best_err)
    return best_s, best_q


@torch.no_grad()
def quantize_rtn(W, gs, group=32, opt=False):
    K = W.shape[1]
    Wn = W / gs
    out = torch.zeros_like(Wn)
    for i in range(0, K, group):
        blk = Wn[:, i:i + group]
        s, best = group_scales(blk, opt)
        out[:, i:i + group] = best if best is not None else e2m1_round(blk / s) * s
    return out * gs


@torch.no_grad()
def quantize_gptq(W, H, gs, group=32, opt=False, damp=1e-2):
    """GPTQ on the deployment grid. W: (N,K) fp32, H: (K,K) fp32 = sum of X^T X.

    Uses the exact OBQ update (d = H^-1[k,k], propagation with H^-1[k, >k]); the Cholesky-of-inverse
    form used by some GPTQ implementations over-corrects here and makes the weight error worse.
    """
    N, K = W.shape
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    if damp > 0:
        d = torch.diag(H)
        H += damp * d.mean() * torch.eye(K, device=H.device, dtype=H.dtype)
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    del H

    Wn = W / gs
    Wn[:, dead] = 0.0
    Q = torch.zeros_like(Wn)
    for i in range(0, K, group):
        g = min(group, K - i)
        Wb = Wn[:, i:i + g].clone()
        s, _ = group_scales(Wb, opt)          # one scale per group, as deployed
        Err = torch.zeros_like(Wb)
        for j in range(g):
            w = Wb[:, j:j + 1]
            q = e2m1_round(w / s) * s
            Q[:, i + j:i + j + 1] = q
            d = Hinv[i + j, i + j]
            if d.abs() < 1e-8:
                Wb[:, j + 1:] = 0
                continue
            err = (w - q) / d
            if j + 1 < g:
                Wb[:, j + 1:] -= err @ Hinv[i + j, i + j + 1:i + g].unsqueeze(0)
            Err[:, j:j + 1] = err
        if i + g < K:
            Wn[:, i + g:] -= Err @ Hinv[i:i + g, i + g:]
    return Q * gs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--ids", required=True, help="token ids for the PPL evaluation (test split)")
    p.add_argument("--calib-ids", required=True,
                   help="token ids for the Hessian; must be a different split than --ids")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--calib-windows", type=int, default=64)
    p.add_argument("--eval-windows", type=int, default=20)
    p.add_argument("--method", default="gptq_opt", choices=["rtn", "rtn_opt", "gptq", "gptq_opt"])
    p.add_argument("--group", type=int, default=32, help="32 is forced by CUTLASS's sparse NVFP4 MMA")
    p.add_argument("--damp", type=float, default=1e-1)
    p.add_argument("--max-layers", type=int, default=0, help="debug: only quantize the first N layers")
    p.add_argument("--no-act", action="store_true", help="skip the +SharQ-activation PPL")
    p.add_argument("--save-sharq", default=None, help="write a deployable SharQ checkpoint here")
    p.add_argument("--save-deq", default=None, help="write the quantized bf16 HF model here")
    p.add_argument("--out", default=None, help="write the measured PPL numbers here (json)")
    args = p.parse_args()

    dev = "cuda"
    ids = json.load(open(args.ids))
    calib_src = json.load(open(args.calib_ids))
    calib = [torch.tensor([calib_src[i * args.seqlen:(i + 1) * args.seqlen]], device=dev)
             for i in range(args.calib_windows)]
    evalw = [torch.tensor([ids[i * args.seqlen:(i + 1) * args.seqlen]], device=dev)
             for i in range(args.eval_windows)]

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev).eval()
    opt = args.method.endswith("_opt")
    use_gptq = args.method.startswith("gptq")

    n_layers = len(model.model.layers)
    layers = range(n_layers if not args.max_layers else args.max_layers)
    t_start = time.time()
    for li in layers:
        blk = model.model.layers[li]
        mods = {}
        for nm in LINEARS:
            mod = getattr(blk.self_attn, nm, None)
            mods[nm] = mod if mod is not None else getattr(blk.mlp, nm)
        # Hessian from the activations of the *already quantized* prefix of the model
        Hs = {nm: None for nm in LINEARS}
        handles = []
        for nm, mod in mods.items():
            def hook(m, inp, nm=nm):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                h = x.t() @ x
                Hs[nm] = h if Hs[nm] is None else Hs[nm] + h
            handles.append(mod.register_forward_pre_hook(hook))
        with torch.no_grad():
            for w in calib:
                model(w)
        for h in handles:
            h.remove()

        info = []
        for nm, mod in mods.items():
            t0 = time.time()
            W = mod.weight.data.float()
            gs = (W.abs().max() / (448.0 * 6.0)).clamp(min=1e-12)
            if use_gptq:
                Q = quantize_gptq(W, Hs[nm], gs, group=args.group, opt=opt, damp=args.damp)
            else:
                Q = quantize_rtn(W, gs, group=args.group, opt=opt)
            info.append(f"{nm}={time.time() - t0:.0f}s")
            mod.weight.data = Q.to(torch.bfloat16)
            del W, Q, Hs[nm]
            torch.cuda.empty_cache()
        print(f"layer {li:2d} done ({' '.join(info)}) elapsed {time.time() - t_start:.0f}s", flush=True)

    def ppl():
        nll, cnt = 0.0, 0
        with torch.no_grad():
            for w in evalw:
                out = model(w, labels=w)
                nll += out.loss.float().item() * (w.shape[1] - 1)
                cnt += w.shape[1] - 1
        return math.exp(nll / cnt)

    res = {"method": args.method, "group": args.group, "damp": args.damp,
           "calib_windows": args.calib_windows, "eval_windows": args.eval_windows}
    res["w_only"] = ppl()
    print(f"{args.method} weights only -> ppl {res['w_only']:.4f}", flush=True)
    if not args.no_act:
        handles = []
        for name, mod in model.named_modules():
            if mod.__class__.__name__ == "Linear" and any(k in name for k in KEYS):
                handles.append(mod.register_forward_pre_hook(lambda m, inp: (q_act(inp[0]),)))
        res["with_act"] = ppl()
        print(f"{args.method} + SharQ act -> ppl {res['with_act']:.4f}", flush=True)

    if args.save_deq:
        model.save_pretrained(args.save_deq)
        AutoTokenizer.from_pretrained(args.model).save_pretrained(args.save_deq)
        print(f"saved bf16 weights to {args.save_deq}", flush=True)

    if args.save_sharq:
        from sharq_quant import save_sharq_checkpoint

        model.to("cpu")
        torch.cuda.empty_cache()
        save_sharq_checkpoint(model, model.config, AutoTokenizer.from_pretrained(args.model),
                              args.save_sharq, src_dir=args.model)
        print(f"saved SharQ checkpoint to {args.save_sharq}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
