<div align="center">
  <p>
    <img src="assets/lockup-light.svg#gh-light-mode-only" alt="MOSS-Image" width="520">
    <img src="assets/lockup-dark.svg#gh-dark-mode-only" alt="MOSS-Image" width="520">
  </p>
  <p>
    <a href="https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base"><img alt="Hugging Face · MOSS-Image-Base" src="https://img.shields.io/badge/🤗%20Hugging%20Face-MOSS--Image--Base-ffc107?style=flat-square"></a>
    &nbsp;
    <a href="https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT"><img alt="Hugging Face · MOSS-Image-SFT" src="https://img.shields.io/badge/🤗%20Hugging%20Face-MOSS--Image--SFT-ffc107?style=flat-square"></a>
    &nbsp;
    <a href="https://huggingface.co/Comfy-Org/MOSS-Image"><img alt="ComfyUI Models" src="https://img.shields.io/badge/ComfyUI-Models-ffc107?style=flat-square"></a>
    &nbsp;
    <a href="https://modelscope.cn/models/openmoss/MOSS-Image-Base"><img alt="ModelScope · MOSS-Image-Base" src="assets/modelscope-base-badge.svg"></a>
    &nbsp;
    <a href="https://modelscope.cn/models/openmoss/MOSS-Image-SFT"><img alt="ModelScope · MOSS-Image-SFT" src="assets/modelscope-sft-badge.svg"></a>
    &nbsp;
    <a href="https://huggingface.co/spaces/OpenMOSS-Team/MOSS-Image"><img alt="Hugging Face Spaces · Demo" src="https://img.shields.io/badge/🤗%20Spaces-Demo-4f46e5?style=flat-square"></a>
    &nbsp;
    <a href="https://platform.mosi.cn/?utm_source=github&amp;utm_medium=readme&amp;utm_campaign=imageapi"><img alt="MOSI API Platform" src="https://img.shields.io/badge/🔌%20MOSI-API%20Platform-0f766e?style=flat-square"></a>
    &nbsp;
    <a href="https://openmoss.github.io/MOSS-Image/"><img alt="Homepage" src="https://img.shields.io/badge/🌐%20Project-Homepage-2563eb?style=flat-square"></a>
    &nbsp;
    <a href="paper/MOSS_Image_Technical_Report.pdf"><img alt="Technical Report" src="https://img.shields.io/badge/📄%20Paper-Technical_Report-b31b1b?style=flat-square"></a>
    &nbsp;
    <a href="https://arxiv.org/abs/2610.23374"><img alt="arXiv: 2610.23374" src="https://img.shields.io/badge/arXiv-2610.23374-b31b1b?style=flat-square&amp;logo=arxiv&amp;logoColor=white"></a>
    &nbsp;
    <a href="https://x.com/Open_MOSS"><img alt="Follow OpenMOSS on X" src="https://img.shields.io/badge/Follow-OpenMOSS-000000?style=flat-square&amp;logo=x&amp;logoColor=white"></a>
    &nbsp;
    <a href="https://qm.qq.com/q/hphM7VhaDu"><img alt="MOSS-Image QQ 交流群：755535143" src="https://img.shields.io/badge/QQ-755535143-12B7F5?style=flat-square&amp;logo=qq&amp;logoColor=white"></a>
    &nbsp;
    <a href="https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=7a3m2007-815f-409b-9c3d-a4129f5c058e"><img alt="MOSS-Builders Feishu Community" src="assets/feishu-builders-badge.svg"></a>
    &nbsp;
    <a href="https://discord.gg/nXFWEyvbb"><img alt="MOSS-Image Discord Community" src="https://img.shields.io/badge/Discord-Community-5865F2?style=flat-square&amp;logo=discord&amp;logoColor=white"></a>
  </p>
  <p>
    <strong>English</strong> &nbsp; | &nbsp; <a href="README_ZH.md">简体中文</a>
  </p>
</div>


## News

- **[2026-10-20]** 🚀 The **MOSS-Image** series is now available on the [MOSI API Platform](https://platform.mosi.cn/?utm_source=github&utm_medium=readme&utm_campaign=imageapi). Try it in the Playground or integrate it through the API.

- **[2026-10-19]** 📄 The [MOSS-Image Technical Report](https://arxiv.org/abs/2610.23374) is now available on arXiv. If you find our work useful, please support us with an [upvote on Hugging Face Papers](https://huggingface.co/papers/2610.23374)! 🤗


- **[2026-10-16]** 🚀 MOSS-Image has day-0 support in [SGLang](https://docs.sglang.io/cookbook/diffusion/OpenMOSS/MOSS-Image), [vLLM-Omni](https://recipes.vllm.ai/OpenMOSS-Team/MOSS-Image-Base), and [ComfyUI](https://huggingface.co/Comfy-Org/MOSS-Image).

- **[2026-10-16]** 🎉 We are introducing **MOSS-Image-Base** and **MOSS-Image-SFT**! Code and model weights are publicly available.

  **Our key highlights:**

  - 📐 **Layout control:** Control the placement of text and visual elements.
  - ✍️ **Strong text rendering:** Generate clear, accurate text within images.
  - 📜 **Long-prompt support:** Accept text prompts of up to 4K tokens.
  - 🎨 **Versatile visual styles:** Create high-quality anime and photorealistic images.
  - 🛠️ **Easy to fine-tune:** Adapt the base model to custom tasks and domains.
  - ⚡ **Compact and efficient:** Efficient image generation with a 5B backbone.
  - 🖥️ **Heterogeneous GPU training**: Trained on alternative GPU platforms.

<a id="community-projects"></a>

## Community projects

- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) now supports full fine-tuning and LoRA training for **MOSS-Image-Base** and **MOSS-Image-SFT**. See the [training documentation](https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/MOSS-Image.md#model-training) and [example scripts](https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/moss_image/model_training).

- [MOSS-Image-SFT-DeBand](https://huggingface.co/MATLOWAI/MOSS-Image-SFT-DeBand) by **MATLOWAI** provides community LoRAs to reduce vertical color banding in **MOSS-Image-SFT**, together with ComfyUI preset workflows. See the model card for examples, settings, and the trade-offs between stripe reduction and changes to image details.

- [MOSS-Image-SFT-8-steps-dmd](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd) by **lodestones** is a community 8-step DMD distillation of **MOSS-Image-SFT**, with LoRA adapters for ComfyUI. See the model card for the latest checkpoints and usage recommendations.

- [MOSS-Image-FP16-fixed](https://huggingface.co/KBlueLeaf/MOSS-Image-FP16-fixed) by **KBlueLeaf** provides FP16-compatible transformer weights for **MOSS-Image-Base** and **MOSS-Image-SFT**, using residual-stream rescaling to improve overflow headroom. Diffusers and ComfyUI formats are available; see the model card for loading instructions and adapter compatibility.

- [ComfyUI-Spectrum-MOSS](https://github.com/Rkkss/ComfyUI-Spectrum-MOSS) by **Rkkss** adds Spectrum-style hidden-state forecasting to accelerate MOSS-Image sampling in ComfyUI, with manual and experimental adaptive nodes. See the repository for recommended sampling settings and examples.

<a id="user-guide"></a>

## User guide

Start with the **[English user guide](USERGUIDE.md)** or **[中文用户指南](USERGUIDE_ZH.md)**
for recommended inference settings, prompt examples, layout input formats, and
LoRA training guidance.

<a id="comfyui"></a>

## ComfyUI

MOSS-Image is supported natively in [ComfyUI](https://github.com/Comfy-Org/ComfyUI/pull/17095). Update to ComfyUI master/nightly with MOSS-Image support and install its dependencies, then use the built-in loaders with [ComfyUI-format weights](https://huggingface.co/Comfy-Org/MOSS-Image). No MOSS-Image plugin is required for this workflow.

The repository plugin remains available for loading a local Hugging Face / Diffusers directory and for existing plugin workflows.

[Installation and usage](comfyui/README.md) · [HF loader plugin workflow](comfyui/moss-image/example_workflows/moss-image.json)

<a id="prompt-enhancer"></a>

## Prompt enhancer skill

The [MOSS-Image Prompt Enhancer](skills/moss-image-prompt-enhancer/SKILL.md) turns
image ideas into detailed captions or spatial Layout prompts. It is designed for
GPT-6-family agents with local file and Python tools. The bundled script converts
pixel-coordinate drafts into compact JSON with integer centered coordinates;
it does not call an image-generation service.

Install the [skill folder](skills/moss-image-prompt-enhancer) in your agent's skills
directory, then invoke `$moss-image-prompt-enhancer` with your image idea, desired
caption or Layout output, and any canvas dimensions. Paste the compiled result
into MOSS-Image's positive prompt; for Layout, use the same generation dimensions.

![MOSS-Image-Base and MOSS-Image-SFT](assets/teaser.webp)

## Contents

- [News](#news)
- [Community projects](#community-projects)
- [User guide](#user-guide)
- [ComfyUI](#comfyui)
- [Prompt enhancer skill](#prompt-enhancer)
- [Models](#models)
- [Quick start](#quick-start)
  - [Installation](#installation)
  - [Python](#python)
  - [Gradio](#gradio)
  - [Sampling settings](#sampling-settings)
  - [Memory and performance](#memory-and-performance)
- [Layout-to-image generation](#layout-generation)
- [Training](#training)
- [Showcase](#showcase)
- [Citation](#citation)
- [License and acknowledgements](#license)
- [Star History](#star-history)

<a id="models"></a>

## 🤗 Models

| Model | Description |
| --- | --- |
| [MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) · [ModelScope](https://modelscope.cn/models/openmoss/MOSS-Image-Base) | Base text-to-image model |
| [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) · [ModelScope](https://modelscope.cn/models/openmoss/MOSS-Image-SFT) | Fine-tuned from MOSS-Image-Base on curated data |

### Base vs. SFT

In this example, SFT produces clearer text grouping and a more organized whiteboard layout.

| MOSS-Image-Base | MOSS-Image-SFT |
| --- | --- |
| ![MOSS-Image-Base](assets/comparison/base.webp) | ![MOSS-Image-SFT](assets/comparison/sft.webp) |


<!-- MOSS-Image-Base uses a 42-layer diffusion transformer, a Qwen3.5-9B text encoder,
and a FLUX.2 VAE. -->

<a id="quick-start"></a>

## 🚀 Quick start

<a id="installation"></a>

### 🛠️ Installation

```bash
git clone https://github.com/OpenMOSS/MOSS-Image.git
cd MOSS-Image
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Apple Silicon / MPS

[Apple silicon Mac setup and limitations](docs/macos.md) · M1 Max / 32GB

<a id="python"></a>

### 🐍 Python

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "OpenMOSS-Team/MOSS-Image-Base",
    custom_pipeline="pipeline_moss_image",
    trust_remote_code=True,
    torch_dtype={"default": torch.bfloat16, "vae": torch.float32},
)
pipe.scheduler = type(pipe.scheduler).from_config(pipe.scheduler.config)
pipe.to("cuda")
image = pipe(
    "A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small sticky note on the refrigerator door is handwritten in Chinese:“\n记得买牛奶和鸡蛋！\n牛奶两盒，鸡蛋一打。\n顺便带一袋面包。”",
    height=1456,
    width=1456,
    num_inference_steps=50,
    guidance_scale=3.0,
    uncond_skip_layers=[22],
    generator=torch.Generator("cpu").manual_seed(42),
).images[0]
image.save("refrigerator_note.png")
```

<a id="gradio"></a>

### 🎨 Gradio

```bash
pip install -e '.[ui]'
python gradio_app.py --model OpenMOSS-Team/MOSS-Image-Base
```

<a id="sampling-settings"></a>

### 🎛️ Sampling settings

| Setting | Default |
| --- | --- |
| Sampler | Flow-matching Euler |
| Steps | 50 |
| CFG | 3 |
| Shift | 5 |
| Layer skipping | Negative-branch skipping at layer 22 |
| Negative prompt (Python API) | Empty |

See the [user guide](USERGUIDE.md) for sampling and prompt recommendations.

<a id="memory-and-performance"></a>

### ⚡ Memory and performance

Select Memory mode in Gradio's Advanced settings:

| Mode | Behavior |
| --- | --- |
| GPU resident | Keep the entire model on GPU |
| Component | Move model components between CPU and GPU as needed |
| Sequential · fallback | Lower memory use with slower inference |

If GPU memory is insufficient, select Sequential or reduce image size.
Compile DiT is disabled by default and unavailable in Sequential mode.

<a id="layout-generation"></a>

## 📐 Layout-to-image generation

MOSS-Image uses JSON layout inputs to control text placement and object arrangement
in complex text-to-image compositions. The image below was generated by MOSS-Image
using layout control.

Layouts can also support controlled editing and user-friendly layout tools.
By editing text, element descriptions, and bounding boxes, users can adjust
image content and composition in a structured way.

[![Layout bounding boxes](assets/showcase/layout-bboxes.webp)](assets/showcase/layout-bboxes.svg)

Paste the layout JSON into Gradio’s Prompt field to specify the scene, text,
objects, and their positions. The following example uses a `summary`/`items` layout for a **1024×1024**
canvas. Save it as `layout.json`:

```json
{
  "summary": {
    "meta": ["poster"],
    "canvas": "single",
    "caption": "A minimalist travel poster with a cream background, a blue mountain illustration, and the title EXPLORE at the top."
  },
  "items": [
    {
      "label": "text",
      "caption": "Large dark-blue uppercase title centered at the top.",
      "text": "EXPLORE",
      "bbox": [-416, -384, -288, 384]
    },
    {
      "label": "object",
      "caption": "A flat blue mountain illustration in the lower center.",
      "bbox": [-192, -288, 384, 288]
    }
  ],
  "bbox_coord_space": "abs_centralized",
  "bbox_format": "yxyx"
}
```

- `summary.caption` describes the overall scene and style.
- `items` describes individual elements: `label`, `caption`, and `bbox`, with
  `text` for the words to render.
- `bbox_format: "yxyx"` specifies `[y_min, x_min, y_max, x_max]`.
- `bbox_coord_space: "abs_centralized"` uses pixel coordinates relative to the
  canvas center. On a 1024×1024 canvas, each axis runs from −512 to +512;
  x increases to the right and y increases downward. For a canvas of width W
  and height H, subtract W/2 from ordinary x coordinates and H/2 from y coordinates.

Keep the coordinates consistent with the requested output dimensions. In Gradio,
paste the JSON into the prompt field and set the matching width and height. With
the Python API, pass `json.dumps(layout, ensure_ascii=False)` as the prompt string.

<a id="training"></a>

## Training

LoRA and full fine-tuning train the DiT only. The Qwen3.5-9B text encoder and the
FLUX.2 VAE stay frozen: precache writes their outputs, and the training step never
loads them. Edit examples are the same loop. A reference image is packed as clean
tokens with RoPE reference index `1`, `2`, … and the loss stays on the noised
target, which keeps `r = 0`.

```bash
pip install -e '.[train]'

python -m training.precache --bundle /path/to/MOSS-Image-Base --images /path/to/captions --out cache/t2i
python -m training.train --config configs/t2i_lora.yaml
python -m training.train --config configs/t2i_full.yaml

python -m training.precache --bundle /path/to/MOSS-Image-Base --manifest edits.jsonl --out cache/edit
python -m training.train --config configs/edit_lora.yaml
python -m training.train --config configs/edit_full.yaml
```

Set `model` in the YAML to the local bundle or a dense ComfyUI diffusion-model
file. For ComfyUI component files, precache/sample also accept `--text-encoder`
and `--vae`; `--bundle` supplies tokenizer assets and component configs.
An edit manifest row is
`{"target": "after.png", "refs": ["before.png"], "instruction": "make it autumn"}`.
Paths are relative to the manifest. Text-to-image data is an image plus a `.txt`
caption with the same file stem.

`configs/t2i_lora.yaml` trains adapter weights on a frozen bf16 base. Training
memory options and current verification limits are documented in
[training validation](training/VALIDATION.md).
Packed batches, selective checkpointing, activation/gradient offload, quantized
CUDA kernels, fixed validation and optimizer controls are described in
[training performance and controls](training/PERFORMANCE.md).
For CPU block offloading, start with `configs/low_vram_lora.yaml` or
`configs/low_vram_full.yaml`, set `model` and `cache_dir`, and read the training documentation's
validation scope and host-memory considerations. Their learning rates still need
dataset-specific tuning.

For LoRA that fits on your GPU, set `blocks_to_swap: 0` to keep all weights
resident. For asynchronous frozen-base streaming, set `blocks_to_swap: 42`
and `block_swap_mode: async`. This uses pinned CPU masters, a dedicated CUDA
copy stream and two GPU staging buffers; each block is one packed H2D copy.
Adapters and optimizer state stay on CUDA. Streamed blocks always use gradient
checkpointing. Full fine-tuning supports synchronous CPU masters, or
`block_swap_mode: async_full` with `optimizer: adafactor_fused` for bidirectional
streaming. The latter updates during backward and requires `grad_clip: 0`.

Full fine-tuning defaults to fp32 parameters:
weights, gradients and AdamW moments require separate storage in addition to
activations, distributed across CPU/GPU when blocks are swapped.
Adafactor reduces optimizer-state memory; the fused path also supports opt-in
BF16 parameter storage. `optimizer: adamw8bit` needs bitsandbytes. The noise
draw is the final pretraining interval from the technical report (logit-normal
`μ = -1.09`, `s = 0.95`, cutoff 0.95, 1% uniform, `T = 1000`).

```bash
python -m training.sample --bundle /path/to/MOSS-Image-Base \
  --lora runs/t2i-lora/step_001000/lora.safetensors \
  --prompt "a fox in the snow" --out fox.png

python -m training.sample --bundle /path/to/MOSS-Image-Base \
  --transformer runs/edit-full/step_002000 \
  --ref source.png --prompt "make it autumn" --out edited.png
```

LoRA files use ComfyUI keys (`diffusion_model.<layer>.lora_A.weight`). Sampling
uses 50 steps, CFG 3, shift 5, and negative-branch skipping at block 22. Both
guidance branches see the reference images. `python -m training.train --smoke`
runs a tiny CPU check without weights.

Sampling accepts `--blocks-to-swap 42` for the default architecture and optional
`--block-swap-mode async` for frozen-weight CUDA streaming, plus
`--base-quantization int8` or `int8_convrot`. The VAE runs separately from DiT
denoising to reduce peak VRAM. Quantization changes outputs approximately.
Current native ComfyUI supports MOSS LoRA loading, but does not implement this
trainer's reference-image conditioning; use `training.sample --ref` for edits.

<a id="showcase"></a>

## 🖼️ Showcases

![Examples overview](assets/showcase/overview.webp)

![Examples overview 2](assets/showcase/overview2.webp)

<a id="citation"></a>

## 📝 Citation

If you use MOSS-Image in your research, please cite our technical report:

```bibtex
@misc{ai2026mossimagetechnicalreport,
      title={MOSS-Image Technical Report},
      author={MOSI. AI},
      year={2026},
      eprint={2610.23374},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.23374},
}
```

<a id="license"></a>

## 📄 License and acknowledgements

The standalone pipeline and training code are provided under the [Apache License 2.0](LICENSE).

We thank the Diffusers, Qwen and FLUX teams.

<a id="star-history"></a>

## ⭐ Star History

<a href="https://www.star-history.com/?repos=OpenMOSS%2FMOSS-Image&amp;type=date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=OpenMOSS/MOSS-Image&amp;type=date&amp;theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=OpenMOSS/MOSS-Image&amp;type=date" />
    <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=OpenMOSS/MOSS-Image&amp;type=date" />
  </picture>
</a>
