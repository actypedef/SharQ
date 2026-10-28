# SharQ × vLLM：配置、运行与测试指南

本目录提供把 SharQ（稀疏 FP4 主路径 + 稠密 FP4 残差路径）接入 vLLM 做端到端部署与测速的完整工具链：CUDA Graph 安全的算子模块、vLLM 插件、量化脚本、baseline 生成脚本、以及延迟/吞吐/分阶段/profiling 全套测试脚本。

- 本文是**操作与复现指南**（中文）；英文简版见 [`README.md`](README.md)。
- 性能数字、问题定位过程、优化收益见 [`PERF_RETROSPECTIVE.zh.md`](PERF_RETROSPECTIVE.zh.md)。
- 实测环境与结果摘要见仓库根目录 [`README.md`](../../README.md)。

---

## 1. 组成

| 部分 | 位置 | 说明 |
|---|---|---|
| vLLM 算子模块 | `kernels/src/vllm_ops.cu`（产物 `sharq_ops_vllm`） | absmax、激活量化（两 GEMM 版 / stacked 版）、pair-add、CUTLASS GEMM 变体（含 Stream-K）、按 M 自动选择执行方式。所有 kernel 跟随当前 CUDA stream，无主机同步，可被 CUDA Graph 捕获 |
| vLLM 插件 | `plugin/sharq_vllm_plugin/` | 通过 `vllm.general_plugins` 入口注册 `sharq` 量化方法；自定义算子 `vllm::sharq_linear`。**不需要修改 vLLM 源码** |
| SharQ 权重生成（免校准） | `sharq_quant.py`（仓库自带） | 用原仓库 `sharq_ops` 量化，预合并 qkv / gate_up |
| SharQ 权重生成（GPTQ，可选） | `quantize_gptq.py` | 在部署用的 g32 网格上做 Hessian 误差补偿 + UE4M3 尺度搜索；需要校准数据，PPL 比 RTN 低 0.13 |
| Baseline 权重生成 | `make_baselines.py` | llm-compressor RTN：NVFP4（需小规模校准）、INT4 g128（W4A8 用）、FP8 dynamic |
| 测试脚本 | 本目录，见第 6 节 | 正确性、延迟、吞吐、满负载、分阶段、profiling |

> 原仓库的 `sharq.py` / `sharq_graph_visible_oom.py`（现归档在 `legacy/`）是"直接改 vLLM 源码"的旧方案，其 kernel 不跟随 stream、每次调用分配显存、首次调用会做带同步的 autotune，**无法与 CUDA Graph 共存**（实测在 graph 内计不到时间）。现已由 `plugin/` + `sharq_ops_vllm` 取代，仅作历史参考保留。

---

## 2. 环境要求

实测组合（下文命令均以此为准）：

| 项目 | 版本 |
|---|---|
| GPU | NVIDIA RTX PRO 5000 72GB Blackwell（`sm_120a`，110 SM，L2 96 MB）。B200 请用 `-DSHARQ_CUDA_ARCH=sm100a` |
| 驱动 / CUDA | 580.105.08 / CUDA 13.0（`/usr/local/cuda`） |
| Python | 3.12 |
| PyTorch | 2.11.0+cu130 |
| vLLM | 0.24.0（pip 发布版，未改源码） |
| FlashInfer | 0.6.12 |
| 平台 | Ubuntu 22.04 |

注意事项：

- **编译用的 nvcc 必须和 torch 的 CUDA 版本一致**。仓库的 `kernels/CMakeLists.txt` 会硬编码 `/usr/local/cuda/include`，若本机 CUDA 与 torch 的 cu 版本不同，请把 toolkit 装到 `/usr/local/cuda` 或改这一行。
- **每个 torch 版本单独一个构建目录**（扩展有 ABI 依赖）。本机现有：
  - `kernels/build_cmake_sm120a` → `sharq` 环境（torch 2.9.1），用于量化权重；
  - `kernels/build_vllm_sm120a` → torch 2.11 的初版；
  - `kernels/build_vllm2_sm120a` → **当前版本**，插件默认优先加载。
- 插件与算子都只支持 **TP=1**，且线性层 `K % 256 == 0`。

### 环境准备

```bash
conda create -n sharq-vllm python=3.12 -y
conda activate sharq-vllm
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install vllm==0.24.0
```

`requirements.txt` 里的 `transformers==4.44.0` 是给原仓库评测路径用的；vLLM 0.24 需要更新版本的 transformers，**不要**在同一个环境里同时装两者。

---

## 3. 一次性配置

### 3.1 编译两个算子模块

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate sharq-vllm
cd /path/to/SharQ

cmake -S kernels -B kernels/build_vllm2_sm120a \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DSHARQ_CUDA_ARCH=sm120a \
  -DPython3_EXECUTABLE=$(which python)

cmake --build kernels/build_vllm2_sm120a --target sharq_ops       -j 16   # 量化权重用（原仓库算子）
cmake --build kernels/build_vllm2_sm120a --target sharq_ops_vllm  -j 16   # vLLM 推理用
```

产物为 `kernels/build_vllm2_sm120a/{sharq_ops.so, sharq_ops_vllm.so}`。编译约 2 分钟（16 线程）。

B200：

```bash
cmake -S kernels -B kernels/build_vllm2_sm100a -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DSHARQ_CUDA_ARCH=sm100a \
  -DPython3_EXECUTABLE=$(which python)
```

### 3.2 安装 vLLM 插件

```bash
pip install -e benchmarks/vllm/plugin --no-deps --no-build-isolation
```

验证入口已注册：

```bash
python -c "from importlib.metadata import entry_points as e; print([x for x in e(group='vllm.general_plugins') if 'sharq' in x.name])"
# [EntryPoint(name='sharq', value='sharq_vllm_plugin:register', group='vllm.general_plugins')]
```

插件启动时把算子目录加入 `sys.path`，优先级为：`SHARQ_VLLM_KERNEL_BUILD` → `SHARQ_KERNEL_BUILD` → `kernels/build_vllm2_sm120a` → `kernels/build_vllm_sm120a` → `kernels/build_vllm_sm100a`。

### 3.3 生成 SharQ 权重

```bash
SHARQ_KERNEL_BUILD=/path/to/SharQ/kernels/build_vllm2_sm120a \
CUDA_VISIBLE_DEVICES=0 \
python benchmarks/vllm/sharq_quant.py \
  --model /path/to/Llama-3.1-8B \
  --save_path /path/to/quantized/sharq/Llama-3.1-8B
```

- 脚本在 CPU 上加载模型、逐层在 GPU 上量化，输出单文件 `model.safetensors`（Llama-3.1-8B 约 6.1 GB），qkv / gate_up 已预合并。
- `config.json` 中会写入 `quantization_config.quant_method = "sharq"`，vLLM 自动识别，**不需要**额外 `--quantization` 参数。
- 脚本会自动把原始模型的 `generation_config.json` 复制到输出目录；`tie_word_embeddings=true` 的模型（如 Qwen2.5-1.5B）会自动去掉重复的 `lm_head.weight`，避免 safetensors 报"共享内存"错误。

### 3.4 可选：GPTQ 权重（更准，需要校准）

3.3 的默认路径是 **RTN**（直接取整）。`quantize_gptq.py` 在**部署用的 g32 网格上**做 GPTQ（逐层 Hessian
误差补偿 + UE4M3 尺度搜索），Llama-3.1-8B 上 WikiText-2 PPL 从 **6.7481 降到 6.6134**，而 checkpoint 格式、
kernel 和运行速度完全不变。

为什么不能在 g16 上解完再投影回 g32：SharQ 权重的尺度粒度被 CUTLASS 的稀疏 NVFP4 MMA 固定成 32
（`nv_float4_t` 在 sparse schedule 下 `SfVectorSize == 32`，见
`cutlass/gemm/collective/builders/sm1xx_common.inl`）。实测"g16 解 + 投影回 g32"只能拿到 −0.03 PPL，
直接在 g32 上解能拿到 −0.13。

```bash
# 1) 准备 token id：train 用于校准，test 用于评估（不能混用，见下面的注意事项）
python - <<'EOF'
import json
from datasets import load_dataset
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/path/to/Llama-3.1-8B')
for split, out in (('train', '/tmp/wt2_train.json'), ('test', '/tmp/wt2_test.json')):
    text = '\n\n'.join(load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split=split)['text'])
    json.dump(tok(text)['input_ids'], open(out, 'w'))
EOF

# 2) GPTQ（单卡约 10 分钟，峰值额外显存约 2.5 GB），直接产出可部署 checkpoint
SHARQ_KERNEL_BUILD=/path/to/SharQ/kernels/build_cmake_sm120a CUDA_VISIBLE_DEVICES=0 \
python benchmarks/vllm/quantize_gptq.py \
  --model /path/to/Llama-3.1-8B \
  --calib-ids /tmp/wt2_train.json --calib-windows 64 \
  --ids /tmp/wt2_test.json --eval-windows 141 \
  --method gptq_opt --damp 0.1 \
  --save-sharq /path/to/quantized/sharq_gptq/Llama-3.1-8B
```

- `--method`：`rtn` / `rtn_opt`（RTN + 尺度搜索）/ `gptq` / `gptq_opt`（默认）。
- **校准必须用 train split**。用 test split 时校准集与评估窗口重叠，收益会被高估约 0.06 PPL。
- 该脚本在 `sharq` 环境（有原仓库 `sharq_ops`）里跑，不需要 `sharq-vllm`；产出走的是和 3.3 完全相同的
  打包/合并路径（`sharq_quant.save_sharq_checkpoint`）。
- `--ids` 的词表必须与模型一致（拿 Llama-3 的 id 喂给 32k 词表的模型会触发 embedding 越界 assert）。

| 配置 | WikiText-2 PPL（141×2048） | 需要校准 | 推理开销 |
|---|---|---|---|
| BF16 | 6.2396 | — | — |
| **SharQ + GPTQ（g32）** | **6.6134** | 是（64×2048 token） | 与 RTN 版相同 |
| SharQ RTN（3.3） | 6.7481 | 否 | — |
| NVFP4 RTN（3.5） | 6.9387 | 是 | — |

### 3.5 生成 baseline 权重（可选，用于对比）

baseline 用 llm-compressor 生成，建议**单独一个环境**（会装 `llmcompressor==0.14.0.1` + `compressed-tensors==0.19.0`）：

```bash
conda create -n llmc python=3.12 -y && conda activate llmc
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install llmcompressor

cd /path/to/SharQ/benchmarks/vllm
# 每个 scheme 单独一个进程，避免显存碎片导致 OOM
CUDA_VISIBLE_DEVICES=0 python make_baselines.py --model /path/to/Llama-3.1-8B \
  --out-root /path/to/quantized --schemes NVFP4
CUDA_VISIBLE_DEVICES=0 python make_baselines.py --model /path/to/Llama-3.1-8B \
  --out-root /path/to/quantized --schemes W4A16
CUDA_VISIBLE_DEVICES=0 python make_baselines.py --model /path/to/Llama-3.1-8B \
  --out-root /path/to/quantized --schemes FP8_DYNAMIC
```

| 目录 | 方案 | 说明 |
|---|---|---|
| `nvfp4/` | NVFP4 W4A4 | 需要校准（脚本内置 64×512 token 的 wikitext2），因为 vLLM 的 NVFP4 格式要求静态激活全局尺度 |
| `w4a16/` | INT4 group=128 | 运行时配 `VLLM_MARLIN_INPUT_DTYPE=fp8` 即变成 W4A8 |
| `fp8_dynamic/` | FP8 W8A8 | 权重按通道、激活按 token 动态，无需校准 |

vLLM 会按 checkpoint 自动选择 kernel：NVFP4 → `FlashInferCutlassNvFp4LinearKernel`，FP8 → `CutlassFP8ScaledMMLinearKernel`，INT4 + `VLLM_MARLIN_INPUT_DTYPE=fp8` → `MarlinLinearKernel`。BF16 直接用原始模型。

### 3.6 冒烟验证

```bash
cd benchmarks/vllm

# 1) 算子正确性（逐位一致性 + CUDA Graph + 全部执行方式）
CUDA_VISIBLE_DEVICES=0 python test_sharq_vllm_ops.py
#   期望最后一行：ALL OK

# 2) vLLM 生成
CUDA_VISIBLE_DEVICES=0 python smoke_generate.py \
  --model /path/to/quantized/sharq/Llama-3.1-8B

# 3) WikiText-2 困惑度（需要预分词好的 token id 文件，见脚本注释）
CUDA_VISIBLE_DEVICES=0 python ppl_vllm.py --label SHARQ \
  --model /path/to/quantized/sharq/Llama-3.1-8B --tokenizer /path/to/Llama-3.1-8B
```

Llama-3.1-8B 参考值：SharQ ppl 6.748，NVFP4 6.939（含校准），FP8 6.289，BF16 6.240。

---

## 4. 运行配置要点

### 4.1 显存与 KV cache

- vLLM 启动时会用 `gpu_memory_utilization` 做一次"空闲显存是否够"的检查，**即使设置了 `kv_cache_memory_bytes` 也会检查**。共享 GPU 上如果别的进程占了大半显存，必须把这个比例调低（例如 `0.2`），否则直接报 `Free memory ... is less than desired`。
- 需要精确控制并发时用 `kv_cache_memory_bytes`（它会覆盖按比例算出的 KV 大小）。经验值：Llama-3.1-8B 每 token KV 约 128 KB，2.25 GiB ≈ 18K tokens，40 GiB ≈ 328K tokens。
- **吞吐对比必须固定 KV 大小**，否则量化方法因为权重更小会自动获得更大 KV、更高并发，比较不公平。

### 4.2 常用参数

| 参数 | 建议 | 说明 |
|---|---|---|
| `--max-model-len` | `2 × 最大输出长度 + 余量` | 不影响显存（按需分配） |
| `--max-num-seqs` | 想测满负载时设 512 | 同时也是 CUDA Graph 捕获上限（`min(2×, 512)`）的依据 |
| `--max-num-batched-tokens` | 8192 | prefill chunk 上限，决定 M 的峰值 |
| `--gpu-memory-utilization` | 独占卡 0.9；共享卡按空闲比例 | 见 4.1 |
| `--kv-cache-memory-bytes` | 固定实验时显式指定 | 见 4.1 |
| `--enforce-eager` | 只在调试时用 | 关掉 CUDA Graph，SharQ 会明显变慢 |

### 4.3 SharQ 专用环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `SHARQ_VLLM_KERNEL_BUILD` | 空 | 指定 `sharq_ops_vllm.so` 所在目录 |
| `SHARQ_VLLM_GEMM` | `flashinfer` | `cutlass` 则所有 M 都走 CUTLASS kernel（消融用） |
| `SHARQ_STACK_MAX_M` | `192` | M ≤ 此值时用 stacked 路径（单 GEMM，权重只读一遍） |
| `SHARQ_STACK_MAX_M_LONG_K` | `256` | 长 K（≥8192，如 down_proj）时的 stacked 阈值 |
| `SHARQ_FORCE_MODE` | 空 | `0`=强制两 GEMM，`1`=强制 stacked（仅对 CUTLASS 分支生效） |
| `SHARQ_FORCE_DENSE` / `SHARQ_FORCE_SPARSE` | 空 | 强制某项 GEMM 变体（0..3 / 0..1），调参用 |

阈值 192 / 256 是在本机用 `bench_full_stacked.py` 扫出来的交叉点（M≈200～260）。换 GPU 后建议重扫，见第 8 节。

### 4.4 CUDA Graph 与 torch.compile

- `sharq_ops_vllm` 的所有 kernel 都在当前 stream 上启动、无主机同步，因此 `vllm::sharq_linear` 这个不透明算子可以正常被 CUDA Graph 捕获。若自行改写 kernel，**务必保持这一点**，否则 graph 回放会静默漏掉 kernel。
- vLLM 启动时会对 FlashInfer 做 autotune（日志 `Autotuning process starts ... ends`）。插件会在 tuning 模式下额外按 stacked 形状（2M 行）调一次，让解码路径也拿到调优后的 tactic。
- 不要同时在环境里保留多个版本的 `sharq_ops_vllm.so`；用 `SHARQ_VLLM_KERNEL_BUILD` 明确指定。

---

## 5. 测试脚本一览

| 脚本 | 用途 | 典型输出 |
|---|---|---|
| `test_sharq_vllm_ops.py` | 算子正确性：与原仓库 kernel 逐位一致、压力分布、全部执行方式、CUDA Graph 回放 | `ALL OK` |
| `smoke_generate.py` | vLLM 加载 + 贪心生成；`--baselines` 一次跑完 BF16/FP8/NVFP4/W4A8 | 4 条 prompt 的续写 |
| `ppl_vllm.py` | WikiText-2 困惑度（vLLM 内 prompt logprobs） | `ppl = x.xxxx` |
| `bench_e2e.py` | 端到端延迟 + 吞吐（一个引擎跑完所有长度） | `results/<tag>/<METHOD>.json` |
| `run_all.sh` | 批量跑多个方法（每个方法一张卡） | 同上 |
| `bench_phases.py` | 单独测 prefill（TTFT）与 decode（每步耗时） | `results/<tag>/phases_<METHOD>.json` |
| `bench_decode_fair.py` | 解码步微基准，对照组与 vLLM 实际跑法一致（静态尺度 + autotune） | 终端表格 |
| `bench_full_stacked.py` | 跨 M 对比执行策略（auto / 全 stacked / 全两 GEMM） | 终端表格 |
| `profile_phases.py` + `run_phases.sh` | 对 prefill / decode 窗口做 nsys 采集 | `prof/<tag>/` |
| `summarize.py` | 三类报告：`e2e`（延迟/吞吐/满负载/PPL）、`phases`（分阶段性能 + kernel 归因）、`kern`（单个 nsys CSV 分类） | `summary.md` / 终端 |
| `quantize_gptq.py` | GPTQ 权重量化（g32 网格 + 尺度搜索），直接产出可部署 checkpoint（见 3.4） | 量化 checkpoint + PPL |
| `legacy/`（`sharq.py`、`sharq_graph_visible_oom.py`、`run_exp.py`） | 旧方案（直接改 vLLM 源码）与上游实验脚本，不参与任何流程，仅作历史参考 | — |
| `analysis_ppl_decomp.py` | 误差归因：只量化激活 / 只量化权重 / 两者，用于判断精度瓶颈在哪一侧 | 终端表格 |
| `make_baselines.py` | 生成 baseline 权重（见 3.5） | 量化 checkpoint |
| `lmeval.py` | 下游零样本评测（论文 7 个任务）：`run` 跑评测、`summarize` 出表，见 6.6 | `results/lmeval/` |

---

## 6. 复现实验

下面命令都假设已 `conda activate sharq-vllm` 且 `cd benchmarks/vllm`。用 `M`/`Q` 表示模型与量化权重根目录：

```bash
M=/path/to/models/Llama-3.1-8B
Q=/path/to/quantized
R=/path/to/results
```

### 6.1 延迟（batch=4，输入=输出长度）

```bash
CUDA_VISIBLE_DEVICES=0 python bench_e2e.py --label SHARQ \
  --model $Q/sharq/Llama-3.1-8B \
  --kv-cache-gib 40 --gpu-mem-util 0.9 \
  --lengths 256,512,1024,2048 --latency-batch 4 --latency-iters 3 \
  --targets latency --out $R/latency_SHARQ.json
```

### 6.2 满负载吞吐

每组 100 万 token（256×2048 / 512×1024 / 1024×512 / 2048×256），`--log-stats` 会记录每步平均并发与抢占次数：

```bash
CUDA_VISIBLE_DEVICES=0 python bench_e2e.py --label SHARQ \
  --model $Q/sharq/Llama-3.1-8B \
  --kv-cache-gib 40 --gpu-mem-util 0.9 --max-num-seqs 512 \
  --targets throughput --tp-warmup 64 --log-stats \
  --tp-prompts 256:2048,512:1024,1024:512,2048:256 \
  --out $R/tput_SHARQ.json
```

多个方法并行（每个方法一张卡，各自独立 tag）：

```bash
EXTRA="--kv-cache-gib 40 --gpu-mem-util 0.9 --max-num-seqs 512 \
  --targets throughput --tp-warmup 64 --log-stats --tp-prompts 256:2048,512:1024,1024:512,2048:256"
CUDA_VISIBLE_DEVICES=0 EXTRA_ARGS="$EXTRA" bash run_all.sh fullload SHARQ  > /tmp/sh.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 EXTRA_ARGS="$EXTRA" bash run_all.sh fullload NVFP4  > /tmp/nv.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 EXTRA_ARGS="$EXTRA" VLLM_MARLIN_INPUT_DTYPE=fp8 \
  bash run_all.sh fullload W4A8 > /tmp/w4.log 2>&1 &
wait
```

### 6.3 分阶段（prefill / decode）

```bash
CUDA_VISIBLE_DEVICES=0 python bench_phases.py --label SHARQ \
  --model $Q/sharq/Llama-3.1-8B --out $R/phases_SHARQ.json
```

- prefill：单请求 1K/2K/4K/8K token，`max_tokens=1`，耗时即 TTFT。
- decode：16 token 输入，`(t(max_tokens=1+128) − t(max_tokens=1)) / 128`，prefill 被减掉。

### 6.4 Profiling 与 kernel 归因

```bash
CUDA_VISIBLE_DEVICES=0 bash run_phases.sh ph1 SHARQ     # 测速 + prefill/decode4/decode64 三组 nsys
python3 summarize.py phases $R/ph1/summary.md            # 性能表 + 各类别占比/绝对耗时
```

单独抓某一组：

```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0
nsys profile -f true --capture-range=cudaProfilerApi --capture-range-end=stop \
  --cuda-graph-trace=node -t cuda -o prof/SHARQ_decode64 \
  python profile_phases.py --model $Q/sharq/Llama-3.1-8B --mode decode --batch 64 --steps 128
nsys stats -q -r cuda_gpu_kern_sum -f csv prof/SHARQ_decode64.nsys-rep > prof/SHARQ_decode64.kern.csv
python3 summarize.py kern prof/SHARQ_decode64.kern.csv
```

`--cuda-graph-trace=node` 必须加，否则解码阶段的 kernel（在 CUDA Graph 里）不会被展开。`VLLM_ENABLE_V1_MULTIPROCESSING=0` 让引擎与 `cudaProfilerStart` 处于同一进程。

### 6.5 执行策略对比

```bash
# A：当前策略（按 M 自动切换）
CUDA_VISIBLE_DEVICES=0 python bench_phases.py --label A --model $Q/sharq/Llama-3.1-8B --out $R/ph_A.json
# B：全部走 stacked
CUDA_VISIBLE_DEVICES=1 SHARQ_STACK_MAX_M=999999 SHARQ_STACK_MAX_M_LONG_K=999999 \
  python bench_phases.py --label B --model $Q/sharq/Llama-3.1-8B --out $R/ph_B.json
# C：全部走两 GEMM
CUDA_VISIBLE_DEVICES=2 SHARQ_VLLM_GEMM=cutlass SHARQ_FORCE_MODE=0 \
  python bench_phases.py --label C --model $Q/sharq/Llama-3.1-8B --out $R/ph_C.json
```

跨 M 的逐层对比用 `bench_full_stacked.py`（含 stacked 内部 prepare / GEMM / pair-add 的拆解）。

### 6.6 下游零样本任务（可选）

论文用的 7 个任务：`piqa, arc_challenge, boolq, hellaswag, winogrande, lambada_openai, arc_easy`，0-shot。

```bash
# 一个方法一张卡，脚本内部起 lm_eval（vLLM 后端），跑完 7 个任务
CUDA_VISIBLE_DEVICES=0 python lmeval.py run SHARQ_GPTQ $Q/sharq_gptq/Llama-3.1-8B
python3 lmeval.py summarize        # 汇总成 markdown 表
```

注意：`lm_eval==0.4.8` 的 vLLM 后端是按旧版 vLLM API 写的，在 vLLM 0.24 上需要打 3 处补丁，具体内容写在
`lmeval.py` 的 docstring 里；`datasets` 需要 3.6.x（5.x 移除了脚本式数据集、裸名解析也变了）。
如果不想改 lm_eval，可以改用 vLLM 的 OpenAI 兼容服务 + `lm_eval --model local-completions`，代价是
字符串重分词会让选项类任务的分数漂移 0.2～0.7 个百分点。

---

## 7. 结果解读注意事项

1. **吞吐数字必须连同 KV 大小一起看**。KV 太小时"吞吐测试"实际是小 batch 解码测试，长序列几组会退化到 4～9 条并发。用 `--log-stats` 记录的真实并发（`avg_seqs_per_step`）来确认是否满负载。
2. **抢占**：KV 装不下所有请求增长到最终长度时，vLLM 会逐出部分请求并在之后重算 prefill。各方法抢占次数一致时可横向比较，但绝对值含重算开销。
3. **prefill 与 decode 的开销结构完全不同**：SharQ 在 prefill 受两 GEMM 算力限制，在小 batch decode 受 prepare kernel 限制，不要用总吞吐代替分阶段结论。
4. **单层微基准是 L2-hot 的**（96 MB L2 能装下一层权重），小 M 的结论会偏乐观；跨方法对比请用 `bench_decode_fair.py`（对照组与 vLLM 跑法一致）。
5. profiler 统计的是 GPU 内核时间，batch 较大时与端到端计时会差 5～10%（引擎侧 CPU 间隙）。
6. **GPTQ 权重只影响精度，不影响速度**：checkpoint 格式、kernel、显存占用与 RTN 版完全相同，所以 6.1～6.5 的测速结果对两版都成立。评估 GPTQ 收益时只需看 PPL / 下游任务，且务必确认校准集与评估集来自不同 split（见 3.4）。

---

## 8. 换 GPU 后的调参流程

1. `python3 bench_full_stacked.py --ms 1,4,16,64,128,192,256,384,512,1024,2048,4096,8192`
   看 `STACKED_FI / TWOG_CU` 在哪个 M 穿过 1.0，得到新的 `SHARQ_STACK_MAX_M`（长 K 用 `SHARQ_STACK_MAX_M_LONG_K`）。
2. 若新卡上 CUTLASS 的 tile/Stream-K 表现变化，用 `SHARQ_FORCE_DENSE` / `SHARQ_FORCE_SPARSE` 逐项扫一遍。
3. 固定阈值后跑 6.3 与 6.4，确认 prefill 与 decode 的占比符合预期。

---

## 9. 故障排查

| 现象 | 原因 / 处理 |
|---|---|
| `Unknown quantization method: sharq` | 插件未装或未生效。检查 3.2 的 entry point；注意在插件工程目录内运行 Python 时目录名会遮蔽包名，已改为 `plugin/` |
| `Free memory ... less than desired GPU memory utilization` | 见 4.1，降低 `--gpu-memory-utilization` |
| `ModuleNotFoundError: sharq_ops_vllm` | 构建目录不在搜索路径。设 `SHARQ_VLLM_KERNEL_BUILD` 指向含 `sharq_ops_vllm.so` 的目录 |
| `undefined symbol` / 导入崩溃 | `.so` 与当前 torch 版本不匹配（ABI）。用当前环境重新编译到独立构建目录 |
| 量化脚本 OOM | `make_baselines.py` 每个 scheme 单独进程；`sharq_quant.py` 默认在 CPU 加载模型 |
| nsys 里看不到解码 kernel | 缺 `--cuda-graph-trace=node`，或没设 `VLLM_ENABLE_V1_MULTIPROCESSING=0` |
| 结果明显偏慢 | 确认没开 `--enforce-eager`；确认 `Autotuning process` 日志出现过；确认加载的是预期构建目录 |
| 形状报错 | 线性层 `K % 256 == 0`（Llama/Qwen2 满足）；TP 只支持 1 |

---

## 10. 数据与文档

| 路径 | 内容 |
|---|---|
| `PERF_RETROSPECTIVE.zh.md` | 性能问题定位、三轮优化（v1→v3）的修改与收益、分阶段归因、执行策略对比 |
| `plugin/` | vLLM 插件源码（`pyproject.toml` + `sharq_vllm_plugin/`） |
| `kernels/src/vllm_ops.cu` | vLLM 算子实现 |
| `/path/to/results/r1/` | 端到端（全部方法） |
| `/path/to/results/fullload/` | 满负载吞吐 |
| `/path/to/results/ph1/` | 分阶段测速与汇总 |
| `/path/to/results/stk{A,B,C}/` | 执行策略对比 |
| `/path/to/results/lmeval/` | 下游 7 个任务的原始结果与日志 |
| `/path/to/prof/ph1/` | 分阶段 nsys 采集与 kernel CSV |
| `/path/to/quantized/sharq_gptq/` | GPTQ 版 SharQ checkpoint（与 RTN 版同格式、同大小） |
