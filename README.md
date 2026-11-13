# PEBench standalone evaluation pipeline

This repository contains a standalone command-line implementation of the PEBench direct prompt evaluation workflow. It does not generate videos and does not depend on the BlindBench web application. Original benchmark JSON files remain unchanged; PE outputs are imported separately when available.

## Pipeline

1. `validate` checks all benchmark records, local media references, image decoding and video streams. The `case_key` is `<mode>:<capid>` because capids repeat across task modes.
2. `prepare` freezes a data snapshot and constructs modality-aware facts. T2V needs no visual extraction; I2V uses its single first-frame image; image-only R2V uses the image-facts provider; R2V cases containing a video use the video-facts provider over all ordered assets in that case. Successful evidence is cached with the benchmark, asset, prompt and provider identities.
3. `evaluate` imports PE prompts as JSONL and evaluates each candidate in three separate calls: Semantic Fidelity (SF), Internal Consistency (IC), and Directorial Creation (DC). A failed case or dimension is recorded and does not stop other work. Successful evidence and judgments can be resumed.
4. `summary.json` and `summary.md` report dimension score averages and criterion pass rates over applicable, non-mock judgments. Mock outputs are always excluded from real statistics. No overall score is synthesized because the paper does not fully specify the severity-to-score conversion.

Every judgment retains criterion-level applicability, severity, evidence, rationale, model response, usage and cache identity. The run directory also contains the frozen case/asset manifest and candidate prompts.

## Requirements

- Python 3.10 or later
- `pip install -r requirements.txt`
- `ffmpeg` and `ffprobe` on `PATH` for full R2V media validation
- Evaluation outputs, report files, and generated PE templates are restricted to the PFS data mount `/pfs/pfs-spMCeu`; the default run directory is the project's `runs/`. Paths on `/` (including `/root` and `/tmp`) are rejected.
- Pillow is used for image decoding; if unavailable, validation falls back to checking image signatures

## Offline validation

```bash
python3 -m pebench_eval validate --data-root data
python3 -m unittest discover -s tests -v
python3 -m pebench_eval self-test --data-root data
```

`self-test` uses the actual benchmark records, but its candidate text, evidence and judgments are synthetic fixtures. It verifies the data protocol, asset ordering, persistence, three-dimension flow and checkpoint reuse. Its outputs are never treated as benchmark measurements.

## Candidate file

Create a blank template, fill `pe_prompt`, and keep the four fields unchanged:

```bash
python3 -m pebench_eval make-template --data-root data --method my-pe --output candidates.jsonl
```

Each non-empty JSONL row has this shape:

```json
{"mode":"i2v","capid":"001","method":"my-pe","pe_prompt":"..."}
```

The importer rejects blank PE prompts, duplicated keys, unknown cases and unsafe method names. Partial files are allowed for pilot runs; unprovided cases remain pending.

## Real provider configuration

The evaluator uses three independently configured model roles. For the legacy CLI, secrets are read from environment variables and are never written to run metadata. YAML configuration can also read a separate key file, as described below.

```bash
export PEBENCH_JUDGE_ENDPOINT="https://.../v1/chat/completions"
export PEBENCH_JUDGE_API_KEY="..."
export PEBENCH_JUDGE_MODEL="..."

# Optional: if omitted, image facts use the judge provider configuration.
export PEBENCH_IMAGE_FACTS_ENDPOINT="https://.../v1/chat/completions"
export PEBENCH_IMAGE_FACTS_API_KEY="..."
export PEBENCH_IMAGE_FACTS_MODEL="..."

# Required for R2V cases that contain videos.
export PEBENCH_VIDEO_FACTS_ENDPOINT="https://.../v1/chat/completions"
export PEBENCH_VIDEO_FACTS_API_KEY="..."
export PEBENCH_VIDEO_FACTS_MODEL="..."
export PEBENCH_VIDEO_PUBLIC_BASE_URL="https://host-serving-the-data-directory/"
export PEBENCH_PROVIDER_RPM="30"
```

The real provider expects an OpenAI-compatible chat-completions JSON endpoint. Requests are serialized and paced at 30 RPM per configured provider by default; tune this to the service quota with `PEBENCH_PROVIDER_RPM` (0 disables pacing). Images up to 1280 KiB are sent unchanged as inline data URLs; larger images are resized/re-encoded (with source hash, original dimensions, transform version and output size recorded in the evidence) to keep multimodal requests bounded. Video facts use the `video_url` message extension; the configured public base must map to the `data/` directory and be reachable by the model API. The video endpoint must support that extension. There is deliberately no silent sampled-frame fallback because it would lose temporal/audio evidence and would not match the paper's video-aware stage.

Prepare evidence before PE outputs exist, then evaluate later using the same run ID, case scope, provider configuration and output root:

```bash
python3 -m pebench_eval prepare --run-id paper-models --provider real
python3 -m pebench_eval evaluate --run-id paper-models --provider real --candidate-file candidates.jsonl
python3 -m pebench_eval summarize --run-dir runs/paper-models
```

For a small pilot, add `--mode i2v --limit 20` or another mode selection. Reusing a run ID with a different data fingerprint or case scope is rejected. Use a new run ID for an independent experiment.

## Notes on comparability

- The paper names GPT-5.4 for image facts and the unified evaluator, and Qwen3.5-Omni-Plus for R2V cases with video. Use those exact versions when accessible; otherwise record the replacement models and treat the results as a local reproduction.
- The pipeline emits a 0-100 dimension score requested by the paper's output format, and separately computes criterion pass rates from `none/minor/major` severities. The score conversion is model-reported; the exact paper aggregation formula is not asserted here.
- R2V asset IDs preserve modality-local numbering (`image_1`, `video_1`) and the original input order. The evaluator must not infer facts that are absent from the cached evidence.
- The current real HTTP adapter and video-message extension are structurally tested with mocked transport. A real API integration test still requires configured credentials, an endpoint that supports image/video payloads, and model-accessible media URLs.

## YAML 配置（独立项目）

编辑项目根目录的 `config.yaml`，然后在项目根目录执行：

```bash
python3 -m pebench_eval validate --config config.yaml
python3 -m pebench_eval make-template --config config.yaml
python3 -m pebench_eval prepare --config config.yaml --run-id my-run
# 填写模板中的 pe_prompt 后，设置 paths.candidate_file，或指定 --candidate-file
python3 -m pebench_eval evaluate --config config.yaml --run-id my-run --candidate-file data/my-candidates.jsonl
python3 -m pebench_eval summarize --run-dir runs/my-run
```

配置包含数据、输出、候选和模板路径；模式、case ID、task_category、task_type 和数量限制；运行编号、方法名、素材解码、断点续跑；真实或 mock provider、是否使用系统代理、超时、输出 token 上限、请求速率；以及裁判、图片事实和视频事实各自的 endpoint、模型和密钥来源。路径相对于 YAML 文件，密钥文件路径相对于项目目录。配置中的未知字段和无效值会报错。可选 CLI 参数优先于 YAML 中对应字段；显式 `--provider real` 使用旧版环境变量配置，`--provider mock` 使用模拟模型。没有 `--config` 时旧命令继续可用。

筛选条件取交集，再应用 limit；空列表表示不限制。所有四个命令使用相同筛选逻辑。YAML 评估要求候选文件覆盖所有选中 case（每个 case 至少一个候选），不同方法仍可部分覆盖。默认只选 1 个 T2V 案例，避免意外批量调用。当前数据仅有 `t2v`、`i2v`、`r2v` 模式，不支持 `t2va`、`r2va`；Speech 等内容类别不等于素材一定包含音轨。

默认裁判为网关 `gpt-5.4`；图片事实配置为 null 时继承裁判。环境变量 `PEBENCH_JUDGE_API_KEY` 优先，其次读取项目数据盘 `secrets/judge_api_key`；该文件权限为 0600，已加入忽略列表。不要将密钥写进 YAML 或报告。要切换其他兼容模型，修改 judge 的 endpoint、model 和密钥来源；独立图片模型可设置同样的四项。`video_facts` 目前为 null；接入视频模型时需要填写 endpoint、model、密钥来源与 public_base_url，服务必须支持 `video_url` 并可访问素材。已部署的 Omni 尚未启动或联调，也没有独立音轨输入适配，不能将当前配置视为完整音视频验证。

同一 run ID 恢复时选中的 case 必须与首次一致，且候选内容一旦写入后不能改变；更换范围或 PE 内容请使用新 run ID。`prepare` 后再 `evaluate` 时，候选文件需覆盖准备阶段选中的所有 case。运行产物和缓存位于 PFS 数据盘的输出目录。
