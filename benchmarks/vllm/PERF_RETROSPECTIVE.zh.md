# SharQ 在 vLLM 上的性能优化复盘

本文记录把 SharQ（稀疏 FP4 主路径 + 稠密 FP4 残差路径）接入 vLLM 后的性能问题定位、修改内容和收益，覆盖从第一版集成（v1）到当前版本（v3）的全过程。文中数字都来自本机实测，原始数据位置见文末。

## 0. 结论速览

| 版本 | 主要改动 | bs=4 延迟（256/256，s） | 相对 v1 | 相对 NVFP4 | 低并发吞吐（256×256，tok/s） | 相对 NVFP4 |
|---|---|---|---|---|---|---|
| v1 | 首次接入 vLLM：算子跟随当前 stream，可被 CUDA Graph 捕获 | 3.785 | 1.00x | 2.42x | 4402 | 0.49x |
| v2 | prepare 重写；解码时稀疏、残差两部分拼成一个 GEMM（stacked 路径）；Stream-K；自动调度 | 2.089 | 1.81x | 1.34x | 6008 | 0.67x |
| v3-CUTLASS | prepare 改用硬件 FP4/FP8 转换；absmax 修正 | 1.947 | 1.94x | 1.25x | 6816 | 0.76x |
| **v3** | stacked 路径改用 FlashInfer GEMM，复用 vLLM 启动时的 autotune | **1.742** | **2.17x** | **1.115x** | **6945** | **0.77x** |

参考：BF16 3.747 s / 3977 tok/s；NVFP4 1.562 s / 9018 tok/s；W4A8 1.520 s / 7366 tok/s。表中"低并发吞吐"在 KV cache 只有 2.25 GiB 的条件下测得，实际并发受 KV 容量限制，含义见 1.1 节；满负载吞吐见 7.2 节。

- **速度**：端到端延迟降到 v1 的 46%（快 2.17 倍），低并发吞吐提升 1.58～1.88 倍。但 SharQ 仍比 NVFP4 慢：延迟慢 9.5～11.5%，低并发吞吐低 10～23%。
- **满负载**（KV cache 40 GiB，实际并发 85～502 条序列）：SharQ v3 的吞吐是 NVFP4 的 0.71～0.89 倍、BF16 的 1.27～1.52 倍，与 FP8 W8A8 基本持平（差别在 ±1% 内），比 W4A8 高 3～14%。
- **精度**：WikiText-2 PPL 全程不变，v1 为 6.7475，v2 和 v3 为 6.7481。作为对照，NVFP4 为 6.9387，W4A8 为 7.0313，BF16 为 6.2396。每次 kernel 重写后，中间量化数据都与原仓库 kernel 逐位一致。
- **分阶段**（7.3 节）：prefill 阶段 SharQ 与 FP8 同速（19246 vs 20150 tok/s），是 BF16 的 1.63～1.79 倍、NVFP4 的 0.57～0.63 倍；decode 阶段 batch ≤ 16 时只比 NVFP4 慢 9～14%，batch=256 时慢 46%。
- **剩余差距的归因**（7.4 节，profiler 实测）：
  - **prefill**：差值的 80% 来自两 GEMM 结构（GEMM 255 ms vs NVFP4 130 ms，每 8192 token）；另约 17% 来自缺融合——SharQ 的 prepare + RMSNorm + SiLU 要独立跑 57 ms，而 NVFP4 把这些融合进 31 ms 的量化 kernel。
  - **decode 小 batch（≤16）**：GEMM 已与 NVFP4 持平（bs=4 时 +0.03 ms），差距全部来自 prepare kernel（0.56 ms/步，约 4.4 µs × 128 个线性层），而 NVFP4 的激活量化只要 0.25 ms。
  - **decode 大 batch（≥64）**：差值的 80% 来自两 GEMM 结构（bs=256 时 GEMM 11.75 vs 7.33 ms）。
- **执行策略**（7.5 节）：stacked 与两 GEMM 的交叉点在 M≈200～260。全 stacked 会使 prefill 慢 6～12%、满负载吞吐低 2.4～6.7%（解码不受影响）；全两 GEMM 则让解码慢 1.7 倍。当前按 M 切换的策略在两个极端都最优。

---

## 1. 测试环境与方法

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA RTX PRO 5000 72GB Blackwell（sm_120，110 SM，L2 96 MB，实测拷贝带宽约 1.1 TB/s） |
| 软件 | CUDA 13.0；vLLM 0.24.0 + torch 2.11.0+cu130 + FlashInfer 0.6.12（conda 环境 `sharq-vllm`）；原仓库环境 `sharq` 为 torch 2.9.1+cu130 |
| 模型 | Llama-3.1-8B，单卡，TP=1 |
| 量化权重 | SharQ 用仓库自带的 `sharq_quant.py`（6.1 GB）；其余 baseline 用 llm-compressor 生成，详见 1.2 节 |
| vLLM 选用的 kernel | 详见 1.2 节（都来自启动日志） |
| 端到端设置 | CUDA Graph + torch.compile 开启，随机 token 输入，关闭前缀缓存，`ignore_eos`；延迟测试 batch=4、输入=输出长度；两套吞吐测试的参数见 1.1 节 |
| 精度 | WikiText-2 测试集，seqlen 2048，141 个窗口，在 vLLM 内用 prompt logprobs 计算 |

不同轮次在不同时间、不同 GPU 上运行。最后一轮在全部空闲的 GPU 上重跑了全部 baseline，与第一轮相比偏差都在 1% 以内（BF16 3.742→3.747 s，NVFP4 1.566→1.562 s），所以跨轮次的数字可以直接比较。

### 1.1 吞吐测试的设置

吞吐测试写成 `长度 × 请求数`。例如 `256×256` 表示：每条请求输入 256 个 token、输出 256 个 token，一共 256 条请求，用 vLLM 的 `LLM.generate()` 一次性全部提交，计时从提交到全部完成，只跑一次。

- 长度同时是输入长度和输出长度。输入是随机 token id（100～100000），输出用 `ignore_eos=True` 固定生成满指定长度，采样温度 1.0。
- 这是**离线批处理**：调度器自行排队和做 continuous batching，没有模拟请求陆续到达。
- 指标"总 tok/s" = (输入 + 输出 token 数) / 耗时。结果 JSON 里也记录了只算输出 token 的值。
- `max_num_batched_tokens=8192`，长 prompt 的 prefill 会被切块（chunked prefill）。

一共做了两套吞吐测试：

| | 低并发吞吐（第 0、4、6 节和 7.1 节的吞吐数字） | 满负载吞吐（7.2 节） |
|---|---|---|
| KV cache | 2.25 GiB（18,432 tokens） | 40 GiB（327,680 tokens） |
| `max_num_seqs` | 256 | 512 |
| 配置 | 256×256、512×128、1024×64、2048×32 | 256×2048、512×1024、1024×512、2048×256 |
| 每组总 token 数 | 131,072 | 1,048,576 |
| 实际并发 | 受 KV 容量限制。按容量估算：256 长度约 36～72 条，2048 长度约 4～9 条（未实测） | 用 vLLM 指标实测：每步平均 502.2 / 333.9 / 169.2 / 85.5 条序列 |
| 测试前的预热 | 同一引擎先跑过相同长度的延迟测试 | 64 条 128/128 请求 |

- **两套配置都让请求数随长度减半**，所以同一套里各组的总 token 量相同。
- **低并发测试为什么用 2.25 GiB**：最初测试时 GPU 上还有别的服务占着约 52 GB，BF16 模型加载后只剩这么多空间，之后所有方法都统一用这个值。这让长序列几组基本退化成小 batch 解码测试（2048×32 只有 4～9 条并发，和 batch=4 的延迟测试差不多），所以它**不代表各方法的最大吞吐**。
- **满负载测试**是在 8 张卡全部空闲后补做的。"实际并发"的算法是：区间内生成的 token 数除以引擎步数（vLLM 指标 `vllm:generation_tokens` / `vllm:iteration_tokens_total` 的计数），即每一步平均有多少条序列在解码。

### 1.2 Baseline 的实现来源

所有 baseline 都是**现成的开源实现**，我没有改动任何代码：

- **推理**：pip 安装的 vLLM 0.24.0 发布版 wheel（非源码编译），FlashInfer 0.6.12 同样是 pip 发布版。
- **权重量化**：llm-compressor 0.14.0.1（compressed-tensors 0.19.0），脚本是 `benchmarks/vllm/make_baselines.py`。它只是调用 `QuantizationModifier(targets="Linear", scheme=..., ignore=["lm_head"])` 这个标准 preset，产出 compressed-tensors 格式，lm_head 保持 BF16。
- **kernel 选择**：都由 vLLM 自动完成（`linear_backend=auto`），表中的 kernel 名都来自 `results/r1/*.log` 里的启动日志。
- **公共部分**：attention 都走 vLLM 自带的 FlashAttention（日志 `Using FLASH_ATTN attention backend`），KV cache 为 BF16（`kv_cache_dtype=auto`）。所以各方法只在线性层上有差别。

| Baseline | 权重来源与量化配置（取自 checkpoint 的 `config.json`） | vLLM 中的量化方案类 | 线性层 kernel（日志确认） | 底层算子 |
|---|---|---|---|---|
| **BF16** | 原始 Llama-3.1-8B（ModelScope 下载），`--dtype bfloat16` | `UnquantizedLinearMethod` | `default_unquantized_gemm` → `torch.nn.functional.linear` | PyTorch 2.11 自带的 cuBLAS GEMM |
| **FP8 W8A8** | llm-compressor `FP8_DYNAMIC`：权重 FP8 E4M3 按输出通道静态量化（RTN，minmax）；激活 FP8 按 token 动态量化；无需校准数据 | `CompressedTensorsW8A8Fp8`（`compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py`） | `CutlassFP8ScaledMMLinearKernel`（`kernels/linear/scaled_mm/cutlass.py`） | 激活量化用 `QuantFP8` → `_C.dynamic_per_token_scaled_fp8_quant`；GEMM 用 `_C.cutlass_scaled_mm`（vLLM 自带的 CUTLASS kernel） |
| **W4A8** | llm-compressor `W4A16`：INT4 对称、group=128（RTN，minmax），无需校准数据。运行时设 `VLLM_MARLIN_INPUT_DTYPE=fp8`，让 Marlin 把激活按 token 在线量化成 FP8，从而成为 W4A8。这是 vLLM 在 SM89 / SM12x 上官方支持的 W4A8 方式；vLLM 另一条 `CutlassW4A8LinearKernel` 路径只支持 SM90，本机不可用 | `CompressedTensorsWNA16`（`compressed_tensors/schemes/compressed_tensors_wNa16.py`） | `MarlinLinearKernel`（`kernels/linear/mixed_precision/marlin.py`） | 加载时用 `marlin_int4_fp8_preprocess` 重排权重，权重 scale ×512；运行时激活经 `QuantFP8`（按 token）量化后调用 `_C.gptq_marlin_gemm`（vLLM 自带的 Marlin kernel） |
| **NVFP4 W4A4** | llm-compressor `NVFP4`：权重和激活都是 FP4 E2M1，group=16，组尺度为 FP8 E4M3，另有一个逐张量全局尺度。激活全局尺度是静态的，需要校准：用 64 条 × 512 token 的 wikitext2 训练集（`static_minmax`）。权重为 RTN | `CompressedTensorsW4A4Fp4`（`compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py`） | `FlashInferCutlassNvFp4LinearKernel`（`kernels/linear/nvfp4/flashinfer.py`） | 激活量化用 `_C.scaled_fp4_quant`（swizzled 尺度布局）。在 torch.compile 下，vLLM 会把它融合进 RMSNorm 和 SiLU-mul（profile 中看到 `fused_add_rms_norm_scaled_fp4_quant`、`silu_mul_cvt_fp16_to_fp4`）。GEMM 用 FlashInfer `mm_fp4(backend="cutlass")`，启动时由 vLLM 做 FlashInfer autotune |
| **SharQ（对照）** | 仓库 `benchmarks/vllm/sharq_quant.py`，调用原仓库 `sharq_ops.quantize_w32_shared`：共享 NVFP4 权重数据 + 32 组和 16 组两套尺度 + 逐张量尺度；qkv / gate_up 在量化时预先合并；无需校准数据 | 本项目插件的 `SHARQConfig` / `SHARQLinearMethod`（`benchmarks/vllm/plugin/`） | 自定义算子 `vllm::sharq_linear` | 见第 2、4、6 节 |

需要说明的几点：

- **NVFP4 是唯一用了校准数据的方法。** 原因是 vLLM 的 NVFP4 checkpoint 格式要求提供静态的激活全局尺度。FP8、W4A8 和 SharQ 都不需要校准数据。因此 NVFP4 在精度和速度上都比"完全免校准"多占了一点便宜：PPL 更低，而且激活量化不用在线算 absmax。
- **两个 4bit baseline 都是 RTN**，没有使用 GPTQ、AWQ 之类的权重优化算法。它们的 PPL（NVFP4 6.9387，W4A8 7.0313）代表的是"标准 preset、不做额外优化"的水平，不是各自格式能达到的最好精度。
- **版本差异**：生成权重用的 compressed-tensors 是 0.19.0，vLLM 环境里是 0.17.0。各 checkpoint 都能正常加载，贪心生成和 PPL 也都正常。

---

## 2. 起点：原仓库的集成方式与 v1

### 2.1 原仓库的问题

原仓库的 `benchmarks/vllm/sharq.py` 是对 vLLM 源码的直接修改，底层调用 `sharq_ops`。逐个检查后，发现它不适合在 vLLM 里做服务部署：

| 问题 | 位置 / 证据 | 影响 |
|---|---|---|
| kernel 都在默认 stream 上启动（`<<<grid, block>>>`、`gemm_op.run()`），不跟随 PyTorch 当前 stream | `fused_sparse_prepare.cu`、`sparse_nvfp4.cu`、`bindings.cpp` | 与 vLLM 的 CUDA Graph 捕获不兼容。实测在 graph 内计时只有 0.2 µs，说明 kernel 根本没被捕获 |
| GEMM workspace 每次调用都用 `cutlass::device_memory::allocation` 分配和释放 | `sparse_nvfp4.cu:186`、`bindings.cpp:217` | 热路径上频繁调用 cudaMalloc / cudaFree |
| 每个新形状第一次调用时做 autotune，并调用 `cudaDeviceSynchronize` | `nvfp4.cu:309`、`sparse_nvfp4.cu:315` | 主机同步，无法放进 CUDA Graph |
| 逐张量激活尺度在 Python 里计算（`x.abs().max()`、`x / scale`、`.to(bf16)`） | `sharq.py:215-216` | 多出几个 elementwise kernel，还要把激活多读写一遍 |

### 2.2 v1 的修改

- **新模块 `sharq_ops_vllm`**（`kernels/src/vllm_ops.cu`）：
  - 所有 kernel 和 CUTLASS GEMM 都在 `getCurrentCUDAStream()` 上启动；workspace 改用 torch 分配；热路径上没有任何主机同步。
  - 激活 absmax 在设备端归约，尺度直接传给 prepare kernel，并通过 GEMM epilogue 的 `alpha_ptr` 折算进输出，不需要回传主机。
  - 残差 GEMM 用 `beta=1` 直接累加到稀疏 GEMM 的输出上（epilogue 融合）。
- **改为插件形式**（`benchmarks/vllm/plugin/`）：通过 `vllm.general_plugins` 入口注册 `sharq` 量化方法，不需要修改 vLLM 源码；用 `direct_register_custom_op` 注册自定义算子，对 torch.compile 来说是不透明的。
- **验证**：prepare 产出的各个缓冲区都与原 kernel 逐位一致；输出与原路径的相对误差为 3.2e-3（bf16 输出舍入的量级）；CUDA Graph 回放结果与直接执行一致；贪心生成正常。

**v1 端到端结果：跑通了，但速度与 BF16 相当**（3.785 s vs BF16 3.747 s），延迟是 NVFP4 的 2.42 倍。

---

## 3. 定位性能问题

### 3.1 逐层微基准（v1，L2-hot，4 个线性层耗时之和）

| M | BF16 | FP8 | NVFP4* | SharQ v1 | SharQ / NVFP4 |
|---|---|---|---|---|---|
| 4 | 353.1 µs | 156.9 | 195.8 | 357.8 | 1.83x |
| 256 | 588.8 | 355.8 | 231.4 | 422.4 | 1.83x |
| 2048 | 4154.6 | 2179.8 | 1338.2 | 2658.1 | 1.99x |

\* 这里的 NVFP4 对照组不规范（每次调用现场算 absmax、GEMM 未 autotune），见第 5 节。

### 3.2 kernel 耗时拆解（torch.profiler）

M=4 时（qkv 层，N=6144，K=4096），各 kernel 耗时：

| kernel | 耗时 |
|---|---|
| prepare | 21.7 µs |
| 稠密 GEMM | 19.6 µs |
| 稀疏 GEMM | 13.3 µs |
| absmax | 1.3 µs |

**问题 1：prepare kernel 的耗时与 M 基本无关，固定约 21 µs。**
- 原因：kernel 的结构是"一个线程处理一个 32 元素组"。M=4、K=4096 时总共只有 512 个线程（4 个 block），要铺到 110 个 SM 上，并行度严重不足。每个线程还要串行处理 32 个元素，并用软件方式做 FP4 转换。原仓库 kernel 还有 176 B 的栈溢出（REG 91 / STACK 176），v1 没有溢出（REG 92 / STACK 0），但结构相同。
- 影响：每个解码步要调用 32 层 × 4 = 128 次线性层，仅 prepare 一项就约 2.7 ms。
- 大 M 时同样慢：M=8192、K=4096 时耗时 415.6 µs，而访存量只有约 98 MB，耗时约为带宽下限的 4～5 倍。

**问题 2：解码时权重被读了两遍。**
稀疏 GEMM 和稠密残差 GEMM 各读一遍 W，而解码阶段的瓶颈正是显存带宽。我用一个 GEMM 方案对比脚本（`explore_decode_gemm.py`，整理时已删除，同类对比现由 `bench_full_stacked.py` 提供）在 M=4 下验证了另一种做法：把两部分拼成 2M 行，跑一个稠密 GEMM，W 只读一遍（耗时单位 µs）：

| 层 | 稀疏 GEMM | 稠密 GEMM | 两个合计 | 拼成 2M 行（SharQ CUTLASS） | 拼成 2M 行（FlashInfer） |
|---|---|---|---|---|---|
| qkv | 14.5 | 22.6 | 37.0 | 20.6 | 16.6 |
| o | 14.4 | 18.5 | 32.9 | 18.5 | 16.2 |
| gate_up | 49.2 | 83.3 | 132.2 | 66.1 | 61.9 |
| down | 32.9 | 51.3 | 84.1 | 51.3 | 48.5 |

结论：拼成一个 GEMM，GEMM 耗时几乎减半。同时验证了 FlashInfer 的 `mm_fp4` 可以直接使用 SharQ 的权重数据和尺度布局，输出与 `sharq_ops.matmul` 完全相同（`torch.equal` 为 True）。这意味着可以复用 vLLM 自己的 NVFP4 GEMM。

**问题 3：大 M 时两个 GEMM 的开销是结构性的。**
M=8192 时，稀疏 GEMM 的耗时只有稠密 GEMM 的 0.65～0.69 倍（qkv 0.68，gate_up 0.65，down 0.69），离理论上的 0.5 倍还有距离。所以"稀疏 + 残差"两个 GEMM 加起来，约是单个稠密 GEMM 的 1.65～1.7 倍。这是方法本身的代价，工程优化去不掉。

---

## 4. v2：重写 prepare，新增 stacked 解码路径，自动调度

### 4.1 修改内容（`kernels/src/vllm_ops.cu`）

- **prepare v2**
  - 改为一个线程处理一个 8 元素块，4 个 lane 合作完成一个 32 元素组；组内的 32 组 / 16 组最大值用 warp shuffle 归约。
  - 每个线程做一次 16 B 加载，写出也改成向量化。
  - 用模板生成两个版本：stacked 版和两 GEMM 版。
- **stacked 模式（M 较小时使用）**
  - 稀疏部分展开成稠密 NVFP4 形式（被剪掉的值填 0，32 组尺度复制到对应的两个 16 组），与残差部分沿行方向拼成 2M 行。
  - 跑一个稠密 NVFP4 GEMM，再用 pair-add 把上下两半相加。
  - 反量化后的操作数取值与原方法完全相同，只是累加顺序不同。
- **GEMM 变体**：稠密 GEMM 有 128×128×128 和 128×128×256 两种 tile，各自又分 persistent 和 Stream-K 两种调度；稀疏 GEMM 分 persistent 和 Stream-K。
- **自动调度规则**（依据调优脚本 `tune_sharq_plans.py` 的扫描结果；该脚本整理时已删除，同类扫描现由 `bench_full_stacked.py` 提供）：
  - M ≤ 192 时走 stacked（K ≥ 8192 的 down_proj 放宽到 M ≤ 256）。
  - 长 K 且 SM 占不满时用 Stream-K。
  - 可用环境变量覆盖：`SHARQ_STACK_MAX_M`、`SHARQ_STACK_MAX_M_LONG_K`、`SHARQ_FORCE_*`。

### 4.2 收益

- **prepare**：M=4 时从 21.7 µs 降到 6.1 µs；M=8192、K=4096 时从 415.6 µs 降到 316.8 µs。
- **微基准**（M=4，L2-hot，4 层合计）：从 357.8 µs 降到约 174 µs。
- **端到端**：延迟从 3.785 s 降到 2.089 s（-45%）；低并发 256×256 吞吐从 4402 升到 6008 tok/s（+36%）。
- **正确性**：7 组形状 × 6 种执行方式 + 自动调度，都与参考实现一致（相对误差 3.2～3.3e-3）；CUDA Graph 在 M=4 和 M=256 下回放结果与直接执行一致。

---

## 5. 纠偏：微基准与端到端结果对不上

### 5.1 发现矛盾

为了让权重不留在 L2 里，我写了 L2-cold 的多层解码步基准（`bench_decode_step.py`，整理时已删除，其结论已被 `bench_decode_fair.py` 与 `bench_full_stacked.py` 取代）：16 层，每层权重各不相同，总量远超 96 MB 的 L2，用 CUDA Graph 计时。它的结果显示，M ≤ 16 时 SharQ v2 只需 NVFP4 的 0.87～0.89 倍耗时。

**但端到端里 v2 的延迟仍是 NVFP4 的 1.34 倍。** 两个结论互相矛盾，必须查清楚。

### 5.2 用 nsys 分析真实引擎

方法：一个运行在 vLLM 引擎进程内的 profiling 脚本（`profile_decode.py`，整理时已由 `profile_phases.py` 取代）跑 batch=4、输入 256 / 输出 128；用 `nsys --cuda-graph-trace=node` 把 graph 内的 kernel 逐个记录下来，再按 kernel + grid 聚合（同功能现由 `summarize.py kern` 按功能类别提供）。

- **GPU kernel 总时间**：SharQ v2 为 988.7 ms，NVFP4 为 720.9 ms，比值 1.37x，与端到端结果吻合。
- **GEMM 逐个对比**（层与 kernel 的对应关系由 grid 推断）：

| 层 | SharQ v2（CUTLASS，stacked） | NVFP4（FlashInfer，已 autotune） |
|---|---|---|
| gate_up | 70.5 µs | 59.0 µs |
| down | 40.2 µs | 34.6 µs |
| qkv | 18.6 µs | 14.1 µs |
| o | 17.3 µs | 10.3 µs |

- **SharQ 独有的 kernel**：prepare 6.0 µs，absmax 1.7 µs，pair-add 0.8 µs，每层每步都要执行。
- **NVFP4 的激活量化**：独立的 `cvt_fp16_to_fp4` 只需 1.0 µs，其余部分已被 vLLM 融合进 `fused_add_rms_norm_scaled_fp4_quant` 和 `silu_mul_cvt_fp16_to_fp4`。

### 5.3 原因：微基准里的 NVFP4 对照组跟 vLLM 实际跑法不一样

- vLLM 的 NVFP4 使用**校准好的静态激活尺度**，不需要每次算 absmax；我的微基准却在每次调用时现场算 absmax。
- vLLM 启动时会**对 FlashInfer GEMM 做 autotune**；我的微基准没有做。

修正后写了 `bench_decode_fair.py`：NVFP4 用静态尺度，所有 FlashInfer GEMM 都先 autotune。重测结果是：M=1 时 SharQ v2 实际为 NVFP4 的 1.29 倍（stacked + FlashInfer）和 1.51 倍（CUTLASS）。

> 我此前对外报告的"小 M 时 SharQ 与 NVFP4 持平甚至更快"是错误结论，原因就在这里。

---

## 6. v3：prepare 改用硬件转换，stacked 路径改用 FlashInfer GEMM

### 6.1 prepare 改用硬件转换（对照 SASS 分析）

- **发现**：用 `nvdisasm` 反汇编 v2 的 stacked prepare kernel，共 3208 条指令，其中有 14 条 CALL、141 对 BSSY/BSYNC 分支同步，`LOP3`、`BRA` 等整数和分支指令占了大部分。原因是 CUTLASS 的 `float_e2m1_t` 标量转换在这个编译单元里走的是软件模拟。
- **修改内容**：
  - FP4 转换改用 `cvt.rn.satfinite.e2m1x2.f32`，8 个值只需 4 条指令就能打包完成。
  - e2m1 反量化改为无分支的位运算；e4m3 尺度的转换改用 `cvt.rn.satfinite.e4m3x2.f32` 和 `cvt.rn.f16x2.e4m3x2`。
  - `x / scale` 改为 `x * rcp(scale)`。当结果落在 bf16 舍入边界附近（低 16 位与 0x8000 相差在 ±32 以内）时，退回精确除法 `__fdiv_rn`，保证与原实现逐位一致。
- **absmax**：单 block 路径的适用阈值从 2^21 个元素降到 2^16。原因是 down_proj 在 M=64 时有 917K 个元素，仍走单 block，耗时 23.4 µs。

| prepare 耗时（µs） | 原仓库 | v1 | v2 | v3 |
|---|---|---|---|---|
| K=4096，M=4 | 21.2 | 21.7 | 6.1 | **2.0** |
| K=4096，M=2048 | 125.0 | 102.8 | 82.6 | **20.3** |
| K=4096，M=8192 | — | 415.6 | 316.8 | **77.5** |
| K=14336，M=2048 | — | 364.6 | 276.7 | **67.3** |
| K=14336，M=8192 | — | 1403.7 | 1094.3 | **297.8** |

- 指令数从 3208 降到 912；与 v1 相比快 4.7～10.9 倍。
- M=8192、K=4096 时，有效带宽约 1.2 TB/s（有部分输入命中 L2），已接近访存瓶颈。
- **正确性**：用 5 类输入分布（重尾、离群通道、极小值、大量零值、大量相同值）× 3 个随机种子 × 3 种形状做压力测试，prepare 的输出与原 kernel 全部逐位一致。

### 6.2 stacked 路径改用 FlashInfer GEMM（插件侧）

- 新增 `sharq_prepare_stacked_fast`：直接输出 FlashInfer 需要的格式，包括 2M 行的数据、fp8 视图的 128×4 swizzle 尺度和设备端的 alpha。另外新增独立的 `sharq_pair_add` 算子。
- 权重尺度 `SFW_dense` 直接以视图方式交给 FlashInfer 作为 B 矩阵的尺度，不需要拷贝。
- **复用 vLLM 启动时的 autotune**：autotune 预热时，vLLM 用 M = `max_num_batched_tokens` 做一次 dummy run。插件检测到处于 tuning 模式后，会额外按 stacked 形状调用一次 FlashInfer GEMM，让它的各个 M 档位都被调优。日志确认启动时已执行 `Autotuning process starts/ends`。
- **路由规则**：M ≤ 192（down_proj 为 256）时走 stacked + FlashInfer；更大的 M 走论文的稀疏 + 稠密两 GEMM 路径（CUTLASS）。设置 `SHARQ_VLLM_GEMM=cutlass` 可强制全部走 CUTLASS，用于消融对比。

### 6.3 收益

**解码步微基准**（`bench_decode_fair.py`，L2-cold，16 层，在空闲 GPU 上测）：

| M | NVFP4（与 vLLM 一致） | SharQ CUTLASS（改前 → 改后） | SharQ FlashInfer（改前 → 改后） | FlashInfer 版 / NVFP4 |
|---|---|---|---|---|
| 1 | 2.035 ms | 3.065 → 2.826 | 2.614 → 2.358 | 1.29x → **1.16x** |
| 4 | 2.084 | 3.098 → 2.865 | 2.668 → 2.417 | 1.28x → **1.16x** |
| 64 | 2.240 | 3.973 → 3.099 | 3.798 → 2.828 | 1.70x → **1.26x** |
| 256 | 3.026 | 8.011 → 6.074 | 7.577 → 5.562 | 2.55x → **1.84x** |

"改前"的数据来自同一脚本上一轮的运行，当时的 NVFP4 耗时为 2.033 / 2.087 / 2.232 / 2.971 ms，与本轮差别在 0.3% 以内。

**端到端**：

| 长度 | v2 | v3-CUTLASS | v3 | v3 / NVFP4 |
|---|---|---|---|---|
| 延迟 256 | 2.089 s | 1.947 | **1.742** | 1.115x |
| 延迟 2048 | 19.186 s | 17.993 | **16.387** | 1.095x |
| 低并发吞吐 256×256 | 6008 tok/s | 6816 | **6945** | 0.770x |
| 低并发吞吐 2048×32 | 1041 tok/s | 1110 | **1201** | 0.898x |

拆开看各项改动的贡献（256 延迟）：

| 改动 | 延迟变化 | 降幅 |
|---|---|---|
| prepare 硬件转换 + absmax 修正 | 2.089 → 1.947 s | -6.8% |
| 改用 FlashInfer GEMM（含 autotune） | 1.947 → 1.742 s | -10.5% |

PPL 不变：v3 为 6.7481，与 v2 一致。

---

## 7. 最终对比

### 7.1 延迟、低并发吞吐与精度

| 指标 | BF16 | FP8 | W4A8 | NVFP4 | SharQ v1 | SharQ v3 |
|---|---|---|---|---|---|---|
| 延迟 256（s） | 3.747 | 2.168 | **1.520** | 1.562 | 3.785 | 1.742 |
| 延迟 2048（s） | 32.537 | 19.891 | **14.595** | 14.970 | 32.795 | 16.387 |
| 低并发吞吐 256×256（tok/s） | 3977 | 6566 | 7366 | **9018** | 4402 | 6945 |
| 低并发吞吐 2048×32（tok/s） | 634 | 1021 | 1335 | **1337** | 639 | 1201 |
| WikiText-2 PPL | 6.2396 | 6.2893 | 7.0313 | 6.9387 | 6.7475 | 6.7481 |
| 权重大小 | 15 GB | 8.5 GB | 5.4 GB | 5.7 GB | 6.1 GB | 6.1 GB |

- 在 4bit 档位里，SharQ 精度最好，且不需要校准数据：PPL 比 NVFP4 低 0.19，比 W4A8 低 0.28。
- 小 batch 场景下，SharQ v3 的速度约为 BF16 的 2 倍，但仍落后于 NVFP4 和 W4A8。

### 7.2 满负载吞吐

KV cache 40 GiB（327,680 tokens），`max_num_seqs=512`，`max_num_batched_tokens=8192`，每组 1,048,576 个 token。设置见 1.1 节。单位为总 tok/s，越高越好。

| 长度 × 请求数 | 实际并发（序列/步） | 抢占次数 | BF16 | FP8 W8A8 | W4A8 | NVFP4 | SharQ v3-CUTLASS | SharQ v3 | SharQ v3 / NVFP4 | SharQ v3 / BF16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 256 × 2048 | 502.2 | 0 | 9503 | 14564 | 12745 | **20328** | 14651 | 14486 | 0.713x | 1.52x |
| 512 × 1024 | 333.9 | 391～394 | 5924 | 8142 | 7426 | **10238** | 8171 | 8127 | 0.794x | 1.37x |
| 1024 × 512 | 169.2 | 311 | 4409 | 5791 | 5410 | **6912** | 5752 | 5753 | 0.832x | 1.30x |
| 2048 × 256 | 85.5 | 169～171 | 2958 | 3724 | 3639 | **4229** | 3739 | 3747 | 0.886x | 1.27x |

读这张表需要注意：

- **除 256 长度外都发生了抢占**：KV 容量装不下所有并发请求增长到最终长度，vLLM 会把一部分请求逐出，之后重新计算它们的 prefill。这是 KV 不够时 vLLM 的正常行为。
  - 各方法的 KV 都是 BF16、容量相同，所以实际并发（差别不到 0.1%，表中列的是 NVFP4 的值）和抢占次数（差别不超过 3 次）几乎一致，横向比较是公平的。
  - 但吞吐的绝对值里包含了重算 prefill 的开销。
- **实际并发与 SharQ 走的路径**：实际并发就是解码时每个线性层的 M。
  - 256 和 512 两组的 M 超过 192，SharQ 走论文的稀疏 + 稠密两 GEMM 路径。
  - 1024 和 2048 两组的 M 在 192 以内，走 stacked 路径。
  - 另外，prefill 的 chunk 最大有 8192 行，任何一组的 prefill 都走两 GEMM 路径。
- **SharQ v3 与 v3-CUTLASS 差别在 1.2% 以内**：满负载时 GEMM 耗时差异已经不是主要因素。

结论：

- **高并发下 SharQ 与 NVFP4 的差距比低并发时更大**：256 长度时为 0.71x（低并发同长度为 0.77x）。原因是大 M 时两个 GEMM 的开销是结构性的（第 3 节问题 3）。
- **序列越长，各方法越接近**：2048 长度时 NVFP4 只比 BF16 快 1.43 倍，SharQ 为 NVFP4 的 0.89x。此时 attention 和抢占重算占比上升，这些开销各方法都一样。
- **满负载时 SharQ ≈ FP8 W8A8 > W4A8**：SharQ 与 FP8 差别在 ±1% 内；比 W4A8 高 3～14%，因为 Marlin 是为小 batch 优化的 kernel，大 batch 时优势消失。小 batch 的结论则相反（7.1 节，W4A8 最快）。

### 7.3 prefill 与 decode 分开测

前面所有吞吐数字都是 prefill + decode 混在一起的。为了看清两个阶段各自的表现，单独测了两段（脚本 `bench_phases.py`，数据在 `results/ph1/`）：

- **prefill**：1 条请求、8192 token 输入、`max_tokens=1`，耗时基本就是 TTFT。
- **decode**：16 token 输入，先测 `max_tokens=1` 的耗时 t1，再测 `max_tokens=1+128` 的耗时 t2，(t2−t1)/128 就是纯解码的每步耗时，prefill 被减掉了。

**prefill（单请求，tok/s，越高越好）**

| 输入长度 | BF16 | FP8 W8A8 | W4A8 | NVFP4 | SharQ v3-CUTLASS | SharQ v3 | SharQ / NVFP4 | SharQ / BF16 |
|---|---|---|---|---|---|---|---|---|
| 1024 | 13655 | 24283 | 19087 | **40150** | 24421 | 23951 | 0.597x | 1.75x |
| 2048 | 13294 | 24416 | 19124 | **41684** | 24298 | 23843 | 0.572x | 1.79x |
| 4096 | 12863 | 22846 | 18158 | **37675** | 22519 | 22033 | 0.585x | 1.71x |
| 8192 | 11772 | 20150 | 16315 | **30478** | 19698 | 19246 | 0.631x | 1.63x |

**decode（每步耗时 ms，越低越好，括号内为 tok/s）**

| batch | BF16 | FP8 W8A8 | W4A8 | NVFP4 | SharQ v3-CUTLASS | SharQ v3 | SharQ / NVFP4 |
|---|---|---|---|---|---|---|---|
| 1 | 13.28 (75) | 8.21 (122) | **5.57** (179) | 5.85 (171) | 7.28 (137) | 6.40 (156) | 1.094x |
| 4 | 14.26 (280) | 8.20 (488) | **5.59** (715) | 5.87 (681) | 7.31 (548) | 6.51 (615) | 1.108x |
| 16 | 14.45 (1107) | 8.44 (1896) | **5.85** (2737) | 6.06 (2642) | 7.62 (2099) | 6.88 (2327) | 1.135x |
| 64 | 15.31 (4181) | 9.19 (6964) | 6.99 (9154) | **6.88** (9308) | 8.59 (7450) | 8.38 (7642) | 1.218x |
| 256 | 27.88 (9183) | 18.54 (13806) | 18.72 (13678) | **11.99** (21358) | 17.83 (14354) | 17.55 (14590) | 1.464x |

两个阶段的结论不一样：

- **prefill 阶段 SharQ 与 FP8 基本同速**（19246 vs 20150 tok/s），是 BF16 的 1.63～1.79 倍，但只有 NVFP4 的 0.57～0.63 倍。prefill 是算力受限，SharQ 要跑两个 GEMM（稀疏 + 稠密残差），而 NVFP4 只需要一个。
- **decode 阶段 batch 越大差距越大**：batch ≤ 16 时只比 NVFP4 慢 9～14%，batch=256 时慢 46%。原因是 batch 超过 192 后 SharQ 从 stacked 路径切到两 GEMM 路径，而小 batch 时瓶颈是权重带宽、stacked 路径只读一遍权重。
- W4A8 在小 batch decode（≤16）里最快，prefill 也居中，是 Marlin 的典型特性。

### 7.4 profiler 归因：时间花在哪

对每个方法分别抓了两段 nsys（`profile_phases.py`，`--cuda-graph-trace=node`），再用 `summarize.py phases`（内置分类器）按功能把 kernel 归类。下表是**每次迭代 / 每步的 GPU 内核时间（ms）**，不是占比，这样能直接看出差值来自哪里。

**prefill（1 条请求 × 8192 token，3 次迭代取平均）**

| 类别 | BF16 | FP8 W8A8 | W4A8 | NVFP4 | SharQ v3-CUTLASS | SharQ v3 | SharQ − NVFP4 |
|---|---|---|---|---|---|---|---|
| **合计** | **701.68** | **408.37** | **506.75** | **270.41** | **415.88** | **425.81** | **+155.40** |
| GEMM | 560.63 | 267.99 | 393.68 | 129.85 | 248.67 | 255.09 | +125.24 |
| attention | 105.22 | 104.63 | 106.39 | 103.38 | 103.92 | 106.76 | +3.38 |
| 激活量化 | — | — | — | 31.02 | — | — | −31.02 |
| SharQ prepare | — | — | — | — | 25.36 | 25.84 | +25.84 |
| RMSNorm | 11.85 | 12.50 | 0.18 | 0.18 | 12.48 | 12.57 | +12.39 |
| SiLU/激活 | 18.00 | 15.32 | — | — | 18.80 | 18.81 | +18.81 |
| 其他（rope/KV/sample 等） | 5.97 | 7.93 | 6.50 | 6.00 | 6.64 | 6.73 | +0.73 |

- attention 一列几乎完全相同（103～106 ms），可以当作公共底噪，说明测量和分类是可靠的。
- **SharQ 的 GEMM 是 NVFP4 的 1.96 倍**（255.09 vs 129.85 ms），差 125 ms，占总差值 155 ms 的 80%。这就是两 GEMM 结构（稀疏 + 稠密残差）的直接代价。
- 但 **SharQ 的 prefill GEMM 与 FP8 W8A8 几乎相同**（255 vs 268 ms）：4bit 稀疏 + 4bit 残差 ≈ 8bit 稠密的开销。
- **融合缺失的代价约 26 ms**：SharQ 需要独立的 prepare（25.84）+ RMSNorm（12.57）+ SiLU（18.81）= 57.2 ms，而 NVFP4 把这些全部融合进 31.02 ms 的量化 kernel 里。
- 按占比看（8192 token）：SharQ 是 GEMM 59.9% / attention 25.1% / prepare 6.1% / SiLU 4.4% / RMSNorm 3.0%；NVFP4 是 GEMM 48.0% / attention 38.2% / 激活量化 11.5%。

**decode（128 步，每步 GPU 时间 ms）**

| 类别 | BF16 | FP8 W8A8 | NVFP4 | SharQ v3 | SharQ − NVFP4 |
|---|---|---|---|---|---|
| bs=4 合计 | 14.03 | 7.96 | 5.40 | 5.87 | +0.47 |
| └ GEMM | 13.38 | 7.09 | 4.63 | 4.65 | **+0.03** |
| └ attention | 0.29 | 0.29 | 0.29 | 0.30 | +0.01 |
| └ 激活量化 | — | — | 0.25 | — | −0.25 |
| └ SharQ prepare | — | — | — | 0.56 | +0.56 |
| bs=64 合计 | 15.44 | 9.14 | 6.53 | 7.62 | +1.10 |
| └ GEMM | 13.97 | 7.55 | 4.98 | 5.44 | +0.47 |
| └ SharQ prepare | — | — | — | 0.69 | +0.69 |
| bs=256 合计 | 29.28 | 19.92 | 12.15 | 17.61 | +5.46 |
| └ GEMM | 24.43 | 14.53 | 7.33 | 11.75 | **+4.42** |
| └ SharQ prepare | — | — | — | 1.08 | +0.46 |

这张表把 decode 的问题彻底分开了：

- **bs=4：GEMM 与 NVFP4 完全持平（+0.03 ms）**，说明 stacked 路径确实解决了"权重读两遍"的问题。差距 0.47 ms 里，最大一块是 SharQ 的 prepare（0.56 ms，约 4.4 µs × 128 个线性层），而 NVFP4 的激活量化只要 0.25 ms。
- **bs=64：差距变成 GEMM 0.47 + prepare 0.69**，GEMM 开始变慢（1.09 倍），因为 M=64 时还在 stacked 路径但已经接近算力受限。
- **bs=256：差距 5.46 ms 里 4.42 ms 来自 GEMM**（1.60 倍）。此时 M > 192 走两 GEMM 路径，与第 3 节微基准测到的"稀疏 0.65～0.69 倍稠密 + 一个稠密 ≈ 1.68 倍"一致。prepare 只占 0.46 ms。
- 按占比看（bs=64）：SharQ 是 GEMM 71.4% / attention 12.1% / prepare 9.1%；NVFP4 是 GEMM 76.3% / attention 13.7% / 量化 4.6%。

profiler 统计的每步 GPU 时间与 7.3 节外部计时的对比：bs=4 为 5.87 vs 5.87、bs=256 为 17.61 vs 17.55，两者一致；bs=64 为 7.62 vs 8.38，差 9%，这 0.76 ms 是引擎侧调度和采样等 CPU 间隙，不计入 GPU 内核时间。所以小 batch 时 GPU 时间几乎等于端到端，batch 变大后 CPU 间隙开始出现。

**归因小结**

| 阶段 | SharQ 落后 NVFP4 的主要原因 | 量级 |
|---|---|---|
| prefill | 两 GEMM 结构（稀疏 + 稠密残差） | 差值的 80%（+125 ms / 8192 token） |
| prefill | 缺融合：prepare + RMSNorm + SiLU 独立执行 | +26 ms（约 17%） |
| decode bs≤16 | prepare kernel（absmax + prepare + pair-add） | 差值的 100%（GEMM 已持平） |
| decode bs≥64 | 两 GEMM 结构（M > 192 切换路径） | 差值的 80%（bs=256 时 4.42 / 5.46 ms） |

对应的优化优先级也随之明确：**小 batch decode 优先做 prepare 融合与合并 kernel（第 8 节前两条），大 batch 场景优先做两 GEMM 的合并 kernel（第 8 节第三条）。**

### 7.5 执行策略对比：stacked 用到底会怎样

stacked 路径（把稀疏部分展开成稠密、与残差部分拼成 2M 行，做一个 GEMM）在解码阶段效果很明显，自然会问：**为什么不干脆全部用 stacked？** 实测答案是：小 M 时它确实最优，大 M 时反而更差，交叉点在 M≈200～260，当前的按 M 切换策略已经接近最优。

**逐层微基准**（4 个线性层耗时之和，CUDA Graph 计时，µs；`bench_full_stacked.py`）

| M | NVFP4（vLLM） | 两 GEMM（CUTLASS，论文 kernel） | 全 stacked（CUTLASS） | 全 stacked（FlashInfer） | stacked-FI / 两 GEMM |
|---|---|---|---|---|---|
| 1 | 83.0 | 287.3 | 161.4 | **90.9** | 0.32x |
| 4 | 78.6 | 268.9 | 162.8 | **92.6** | 0.34x |
| 16 | 72.3 | 262.7 | 156.7 | **93.0** | 0.35x |
| 64 | 78.1 | 238.9 | 142.7 | **130.5** | 0.55x |
| 128 | 109.8 | 230.5 | 201.5 | 202.5 | 0.88x |
| 192 | 184.7 | 325.1 | 259.4 | **259.0** | 0.80x |
| 256 | 180.8 | **326.7** | 365.5 | 348.9 | 1.07x |
| 512 | 310.6 | **558.6** | 636.4 | 644.8 | 1.15x |
| 1024 | 566.6 | **1033.8** | 1200.6 | 1201.5 | 1.16x |
| 2048 | 1080.5 | **2100.7** | 2578.3 | 2818.3 | 1.34x |
| 4096 | 2146.6 | **4108.9** | 5076.4 | 5112.4 | 1.24x |
| 8192 | 4284.3 | **8495.6** | 10982.2 | 10351.7 | 1.22x |

- **M ≤ 64：stacked 是压倒性的**。M=4 时比两 GEMM 快 2.9 倍（92.6 vs 268.9 µs），因为解码是权重带宽受限，两 GEMM 要读两遍权重。
- **M ≈ 200～260 交叉**，之后两 GEMM 反超，M ≥ 512 时全 stacked 慢 15～34%。
- **stacked 用 CUTLASS 还是 FlashInfer 差别很大**：小 M 时 FlashInfer 快 1.7 倍（91 vs 161 µs），大 M 时两者接近。所以 v3 在 stacked 分支上必须用 FlashInfer。

**为什么大 M 时 stacked 反而更慢**（`gate_up`，N=28672）：

| M | 2M×N 中间结果 | prepare | GEMM（2M 行） | pair-add | 合计 |
|---|---|---|---|---|---|
| 1024 | 117 MB | 14.9 µs | 533.6 µs | 127.5 µs | 653.4 µs |
| 4096 | 470 MB | 46.1 µs | 2070.5 µs | 615.7 µs | 2656.7 µs |
| 8192 | 940 MB | 96.4 µs | 4102.7 µs | 1232.0 µs | 5349.0 µs |

两个原因：

1. **算力从 1.5 倍变成 2 倍**：两 GEMM 是"稀疏（0.5 倍面积）+ 稠密（1 倍）"= 1.5 倍稠密算力；stacked 是 2 个稠密 = 2.0 倍。
2. **多了一份 2M×N 的中间结果**：M=8192 时 gate_up 的输出有 940 MB，pair-add 要把它整个读一遍再写一半，耗时 1232 µs，占 stacked 总时间的 23%，而且只受带宽限制（1.41 GB / 1.23 ms ≈ 1.14 TB/s）。小 M 时这份中间结果只有几 MB，可以忽略。

**端到端验证**：三种策略各起一个 vLLM 引擎（同一份 SharQ 权重）：

- A：当前 v3 策略（M ≤ 192/256 用 stacked + FlashInfer，否则两 GEMM）
- B：全部 stacked + FlashInfer（`SHARQ_STACK_MAX_M=999999`）
- C：全部两 GEMM + CUTLASS（`SHARQ_VLLM_GEMM=cutlass SHARQ_FORCE_MODE=0`）

prefill（单请求，tok/s）：

| 输入长度 | A 当前策略 | B 全 stacked | C 全两 GEMM |
|---|---|---|---|
| 1024 | 23951 | 22024 | **24464** |
| 2048 | 23843 | 21015 | **24056** |
| 4096 | 22033 | 19311 | **22359** |
| 8192 | 19246 | 16976 | **19565** |

decode（每步 ms）：

| batch | A 当前策略 | B 全 stacked | C 全两 GEMM |
|---|---|---|---|
| 1 | 6.40 | **6.35** | 11.22 |
| 4 | 6.51 | **6.45** | 11.05 |
| 16 | 6.88 | **6.83** | 11.32 |
| 64 | 8.38 | **8.10** | 11.76 |
| 256 | **17.55** | 17.91 | 17.58 |

（B 与 A 在 batch ≤ 64 时本应完全相同，实测差 0.5～3% 属于运行间波动。）

延迟（batch=4）与满负载吞吐：

| 指标 | A 当前策略 | B 全 stacked | C 全两 GEMM |
|---|---|---|---|
| 延迟 256（s） | **1.726** | 1.729 | 2.949 |
| 延迟 2048（s） | **16.307** | 16.372 | 26.202 |
| 满负载 256×2048（tok/s） | **14570** | 13589（−6.7%） | 14481（−0.6%） |
| 满负载 512×1024（tok/s） | **8128** | 7698（−5.3%） | 8038（−1.1%） |
| 满负载 1024×512（tok/s） | **5770** | 5556（−3.7%） | 5621（−2.6%） |
| 满负载 2048×256（tok/s） | **3750** | 3659（−2.4%） | 3547（−5.4%） |

结论：

- **全 stacked**：解码维持最优（本来就用它），但 prefill 慢 6～12%、满负载吞吐低 2.4～6.7%。
- **全两 GEMM**：prefill 与当前策略持平，但**解码慢 1.7 倍**（11.05 vs 6.51 ms/步），低并发延迟从 1.73 s 涨到 2.95 s。
- **当前按 M 切换**在两个极端都取到最好的结果，没有明显损失。若只允许一种策略，两 GEMM 在吞吐上更稳（满负载只差 0.6～5.4%），但会牺牲 1.7 倍的解码延迟。
- 想进一步扩大 stacked 的适用范围，需要先解决 pair-add 的带宽开销（写自定义 epilogue 直接读两行相加，省掉 2M×N 的中间结果）。按 M=8192 时 pair-add 占 stacked 总时间 23% 估算，去掉后交叉点会明显后移，但具体位置需要实测。

---

## 8. 剩余差距与后续方向

| 差距来源 | 量级 | 建议方向 |
|---|---|---|
| 解码阶段每个线性层多出的激活处理：absmax 约 1.7 µs + prepare 约 2.0 µs + pair-add 约 0.8 µs，而 NVFP4 约 1 µs 且大多已融合 | 解码步 1.16x | 把 prepare 融合进 RMSNorm（仓库已有 `fused_rmsnorm_sparse_prepare.cu`，可改为跟随当前 stream 的版本）和 SiLU-mul；absmax 可在融合 kernel 内用两遍或 cooperative 方式归约。也可以选择静态校准尺度，但这会改变方法"免校准"的特性 |
| pair-add 是一个单独的 kernel，并且要多写一遍 2M×N 的中间结果 | 每层约 0.8 µs | 写自定义 GEMM epilogue，同时读第 r 行和第 r+M 行并相加 |
| 大 M 时两个 GEMM：稀疏 GEMM 耗时为稠密的 0.65～0.69 倍，合计约 1.7 倍 | 满负载吞吐 0.71～0.89x（低并发 0.77～0.90x） | 写一个持久化 kernel，在同一个累加器里依次跑稀疏和稠密两个主循环，省掉 C 矩阵的一次读写和一次 kernel 启动；调优稀疏 GEMM 的 tile |
| 插件只支持 TP=1 | — | 支持 qkv / gate_up 的分片加载，以及按 TP 切分尺度布局 |

---

## 9. 经验教训

1. **对照组必须和 vLLM 的实际跑法一致**：包括静态尺度、autotune 和算子融合。第 5 节的误判就是因为对照组不一致。
2. **微基准要区分 L2-hot 和 L2-cold**：96 MB 的 L2 能装下单层权重，只测单层会让小 M 的结果偏乐观。
3. **端到端与微基准对不上时，直接看真实引擎**：用 `nsys --cuda-graph-trace=node` 加按 kernel + grid 聚合来归因，比继续堆微基准更快找到问题。
4. **每次重写 kernel 都做逐位一致性校验和压力分布测试**：本项目三次重写 prepare，PPL 始终没变。
5. **看 SASS**：高层库里的类型转换不一定会编译成硬件指令，本项目 prepare 最大的一次提速就来自这里。
6. **环境和构建隔离**：独立的 conda 环境，按 torch ABI 分开构建目录（见第 10 节），不影响机器上已有的服务。
7. **吞吐测试要记录实际并发**：只写"提交多少条请求"不够，真实 batch 由 KV 容量决定。最初 2.25 GiB 的 KV 让"吞吐测试"实际变成了小 batch 测试，而 SharQ 与各 baseline 的相对表现很依赖 batch 大小（W4A8 在小 batch 最快、在满负载反而落后于 SharQ）。现在 `bench_e2e.py --log-stats` 会记录每步平均序列数和抢占次数。

---

## 10. 改动清单与复现

### 仓库改动

| 文件 | 说明 |
|---|---|
| `kernels/src/vllm_ops.cu`（新增） | `sharq_ops_vllm` 模块：absmax、prepare（两 GEMM 版 / stacked 版）、pair-add、CUTLASS GEMM 变体（含 Stream-K）、自动调度 |
| `kernels/CMakeLists.txt` | 新增 `sharq_ops_vllm` 编译目标 |
| `benchmarks/vllm/plugin/` （新增） | vLLM 插件，通过 `vllm.general_plugins` 注册 `sharq` 量化方法，自定义算子名为 `vllm::sharq_linear` |
| `benchmarks/vllm/test_sharq_vllm_ops.py` | 正确性测试：逐位一致性、压力分布、全部执行方式、CUDA Graph |
| ~~`bench_linear_shapes.py`~~ | 逐层微基准（单层 L2-hot，小 M 结论偏乐观，第 3 节）；已被 `bench_decode_fair.py` + `bench_full_stacked.py` 取代，脚本已删除 |
| `bench_decode_fair.py` | 与 vLLM 跑法一致的解码步对照（推荐使用这个） |
| `bench_e2e.py`、`run_all.sh`、`summarize.py e2e` | 端到端测速与结果汇总 |
| `bench_phases.py`、`profile_phases.py`、`run_phases.sh`、`summarize.py phases` | prefill / decode 分阶段测速、分阶段 nsys 采集、按功能类别归因 kernel 耗时 |
| `bench_full_stacked.py` | 跨 M 对比执行策略：全 stacked / 全两 GEMM / 当前自动策略（含 stacked 内部各步拆解） |
| `make_baselines.py`、`smoke_generate.py`（`--baselines`）、`ppl_vllm.py` | baseline 权重生成、生成冒烟测试、PPL 计算 |
| `quantize_gptq.py`、`analysis_ppl_decomp.py` | 精度方向的后续工作：g32 网格上的 GPTQ 权重（PPL 6.7481 → 6.6134，运行时零开销）与"激活侧 / 权重侧"误差归因，见 vLLM README 3.4 节 |

构建目录：
- `kernels/build_cmake_sm120a`：`sharq` 环境，torch 2.9.1。
- `kernels/build_vllm_sm120a`：v1 及 torch 2.11 版的 `sharq_ops`。
- `kernels/build_vllm2_sm120a`：当前 v3，插件默认优先加载这个目录。

v2 的二进制和源码已被 v3 覆盖，没有单独保留；v3-CUTLASS 模式（`SHARQ_VLLM_GEMM=cutlass`）可近似看作"v2 + prepare v3"。

### 复现命令

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate sharq-vllm
cd /root/SharQ

# 编译
cmake -S kernels -B kernels/build_vllm2_sm120a -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DSHARQ_CUDA_ARCH=sm120a \
  -DPython3_EXECUTABLE=$(which python)
cmake --build kernels/build_vllm2_sm120a --target sharq_ops_vllm -j 16
pip install -e benchmarks/vllm/plugin --no-deps --no-build-isolation

# 正确性
CUDA_VISIBLE_DEVICES=0 python benchmarks/vllm/test_sharq_vllm_ops.py

# 解码步对照（与 vLLM 跑法一致）
cd benchmarks/vllm
CUDA_VISIBLE_DEVICES=0 python bench_decode_fair.py

# 端到端（可选方法：BF16 FP8 W4A8 NVFP4 SHARQ SHARQ_CUT）
CUDA_VISIBLE_DEVICES=0 EXTRA_ARGS="--kv-cache-gib 2.25 --tp-prompts 256:256,512:128,1024:64,2048:32" \
  bash run_all.sh <tag> SHARQ NVFP4

# 满负载吞吐（只测吞吐，记录实际并发与抢占次数）
CUDA_VISIBLE_DEVICES=0 EXTRA_ARGS="--kv-cache-gib 40 --gpu-mem-util 0.9 --max-num-seqs 512 --targets throughput \
  --tp-warmup 64 --log-stats --tp-prompts 256:2048,512:1024,1024:512,2048:256" \
  bash run_all.sh fullload SHARQ NVFP4

# prefill / decode 分阶段：测速 + 三组 nsys profiling（prefill / decode bs=4 / bs=64），一个方法一张卡
CUDA_VISIBLE_DEVICES=0 bash run_phases.sh ph1 SHARQ
python3 summarize.py phases        # 生成 7.3 / 7.4 的表（含 kernel 分类归因）

# PPL（需要预先分词好的 /dockerdata/sharq_work/results/ppl/wikitext2_test_llama3_ids.json）
CUDA_VISIBLE_DEVICES=0 python ppl_vllm.py --label SHARQ --model /dockerdata/sharq_work/quantized/sharq/Llama-3.1-8B \
  --tokenizer /dockerdata/sharq_work/models/Llama-3.1-8B

# 汇总表
python3 summarize.py e2e
```

### 原始数据位置（`/dockerdata/sharq_work/`）

| 路径 | 内容 |
|---|---|
| `results/v0/` | 第一轮端到端：baseline + SharQ v1 |
| `results/v2/` | SharQ v2 端到端 |
| `results/r1/` | 最终轮端到端：全部 baseline + SharQ v3 + v3-CUTLASS，在空闲 GPU 上测 |
| `results/fullload/` | 满负载吞吐：全部 baseline + SharQ v3 + v3-CUTLASS（KV 40 GiB） |
| `results/ph1/` | prefill / decode 分阶段测速结果 + `summary.md`（7.3、7.4 两节的数据） |
| `results/stkA/`、`results/stkB/`、`results/stkC/` | 执行策略对比：当前策略 / 全 stacked / 全两 GEMM 的端到端结果 |
| `results/stk/` | 执行策略对比的分阶段测速结果 |
| `results/ppl/` | 各方法的 PPL |
| `results/summary.md` | 端到端汇总表 |
| `prof/` | nsys 采集文件（`prof/ph1/` 是分阶段的 prefill、decode bs=4/64/256 采集） |
| `quantized/` | 各方法的量化权重 |
