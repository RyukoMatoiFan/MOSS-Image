# MOSS-Image-Base 用户指南

## 推理参数配置

作为纯粹的 Base Model, MOSS-Image-Base 没有做过任何后训练, 也没有做 CFG 等推理参数的蒸馏. 我们的本意是完整保留训练信息, 方便社区训练 LoRA; 但同时, 这也要求用户需要使用合适的参数配置才能发挥出最佳效果. 以下是我们的推荐配置:

* **分辨率 (Resolution)**: 我们推荐使用 **1M、1.5M、2M** 的总像素数输出.此外, 模型可以外推至 0.25M (`512×512`) 到 4M（2K）的任意分辨率输出.
* **长宽比 (Aspect Ratio)**: 我们支持任意长宽比。Gradio 的分辨率和长宽比控件会计算对齐到 16 倍数的尺寸；使用 Python API 或手动输入宽高时，必须提供 16 的正整数倍，否则会报错。
* **推理步数 (Steps)**: **≥ 50** 步时效果最佳.
* **CFG**: 默认 **3.0**（推荐范围 3.0 - 6.0）. 常规长 Prompt 用 3.0 即可; 短 Prompt 建议适当调高.
* **Shift**: 默认 **5.0**.
* **采样器 (Sampler)**: 默认推荐 **Euler**.
* **SLG (Skip Layer Guidance)**: 我们实现了两种方式:
  * **标准 SLG**: scale 3, 0.01-1.0.
  * **二次采样 SLG**: 推荐使用 `uncond skip`, skip layer 推荐为 **22**（0-index）. layer 22, 24, 26 都对改善图像质量有帮助.

  关闭 SLG 会使输出的图像非常真实, 但是可能会造成细节的崩坏.

## Prompt 规范

MOSS-Image-Base 支持自然语言、Tag 串以及 JSON Layout 等多种输入格式.

* **语言**: 默认推荐**英文**（训练量最多）, 同时也支持中文等多语言输入.
* **负面提示词 (Negative Prompt)**: 经过测试, 我们推荐下面这些有效的负面 tag:
  ```text
  nsfw, worst quality, bad anatomy, watermark, jpeg artifact
  ```
* **Tag 推荐顺序**: 为了在 tag 串中实现最好的性能, 我们推荐使用 Quality -> Score -> Meta -> Year -> Character -> Series -> Artist -> General 的顺序构造. 以下是一个 Example:
  ```text
  best quality, highres, hatsune miku, vocaloid, @foo bar, 1girl, blue hair
  ```
* **文字排版**: 模型支持生成文字并自动换行, 但未必完全符合预期. 如过需要追求文本的正确率, 或者精确控制换行, 请在 prompt 中手动添加 `\n` 或者直接换行.

推荐的 prompt examples:

```text
prompt:
一个穿着"MOSI"标志的T恤的中国美女正拿着黑色的马克笔面向镜头微笑。她身后的玻璃板上手写体写着 "一、MOSS-Image的技术路线：5B 的单流 DiT 模型，基于 Muon 训练。二、MOSS-Image的模型特色：1、复杂文字渲染。支持中英渲染、自动布局； 2、使用 Layout 进行精准图像控制，支持文字编辑和物体增减等自由编辑。三、MOSS-Image的后续发展：探索 R2I 参考生图、助力生成式AI发展。"

negative prompt:
worst quality, bad anatomy
```

```text
prompt:
A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small sticky note on the refrigerator door is handwritten in Chinese:
“记得买牛奶和鸡蛋！
牛奶两盒，鸡蛋一打。
顺便带一袋面包。”

negative prompt:

```

### Layout Prompt（实验性功能）


MOSS-Image-Base 探索了文生图中的布局控制，目前支持通过以下 JSON 格式输入:

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
其中,
- `summary.caption`: 整体场景与风格描述。
- `items`：各元素的类别（label）、描述（caption）和位置框（bbox）。生成文字时需用 text 指定内容。
- `bbox_format: "yxyx"`：坐标顺序为 `[y_min, x_min, y_max, x_max]`。
- `bbox_coord_space: "abs_centralized"`：以画布中心为原点的绝对像素坐标（x 向右、y
  向下递增）。例如 1024×1024 画布的范围是 -512 到 +512。对于宽 W、高 H 的画布，将常规左上角原点的 x、y 分别减去 W/2 和 H/2 即可转换。

> 布局控制使用绝对像素坐标，适用于不同分辨率和长宽比。

模型支持自然语言和结构化语言的混合输入。

## Prompt Enhancer and Layout Enhancer

MOSS-Image 系列原生支持 layout, 这给我们的模型带来了很多有趣的功能, 提升了模型能力的上限, 同时也变相增加了用户使用的困难.

布局增强可以使用 LLM 配合布局规划 skill，也可以使用微调后的开源 LLM 作为 layout enhancer。

## 后训练

### LoRA

监督微调可以让 MOSS-Image-Base 支持 GIF 生成和参考图生成等功能。

本仓库带有 LoRA、全量微调，以及参考图编辑训练。请保持 Text Encoder 冻结。用法见 README 的 Training 一节。

### 图片标注

如果你要自己后训练 MOSS-Image-Base, 可以参考下面的图片标注方法.

#### Caption Prompt

```text
You are an image recaptioner. Return exactly one JSON object with this schema:
{
  "caption": {
    "long": "a detailed image description, including all non-watermark visible text",
    "medium": "a one-paragraph image description",
    "short": "a one-sentence image description"
  },
  "watermark": [
    {"box_2d": [y_min, x_min, y_max, x_max], "label": "watermark text"}
  ],
  "borders": ["none"]
}

Rules:
- caption.long:
    - Describe the composition, key subjects, actions, setting, colors, lighting, and artistic style in detail. Include only information visible in the image.
    - Preserve visible text exactly, excluding watermarks. Do not translate, paraphrase, romanize, or normalize image text.
    - Describe text layout as much as possible: line breaks, ordering, relative position, alignment, size hierarchy, font/style, color, spacing, and how text blocks are arranged.
- caption.medium: compress long into one paragraph.
- caption.short: summarize the main subject, action, or relationship in one sentence.
- Do not describe watermarks in any caption field. Record them only in watermark.
- watermark:
    - Use normalized 0–1000 [y_min, x_min, y_max, x_max] coordinates relative to the full image.
    - Set label to the visible watermark text, or a short description if unreadable.
    - Use [] if there are no watermarks.
- borders:
    - Record evident borders or margins using one or more of "left", "right", "up", and "down".
    - Use ["none"] if there are no evident borders or margins. Do not combine "none" with other directions.
- All three caption fields must be non-empty strings.
- Treat the original caption as a reference. Correct any claims that are not supported by the image.
- Return only the JSON object, without Markdown or explanations.

Here is the original caption reference:
{{reference}}
```

#### Caption Model

MOSS-Image-Base 的推荐标注模型包括 Gemma 4、Qwen 3.5 27B 和 Qwen 3.5 35B-A3B。可以使用以下 chat template 进行标注：

```python
body = {
    "model": model_name,
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ],
    "n": 1,
    "stream": False,
    "max_tokens": 16384,
    "temperature": 1,
    "top_p": 0.95,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
}
```

我们推荐开启 thinking 模式, 以获得更好的标注精度.

#### Tips

1. 适当的世界知识注入可以获得更好的 caption.
2. Agentic caption pipeline 可以实现相当高的标注精度.
3. 理解促进生成.

### 少步蒸馏
