# MOSS-Image-Base User Guide

## Inference Parameters

As a base model, MOSS-Image-Base has not undergone any post-training or distillation of inference parameters such as CFG. Our goal is to preserve the training information in full and make it easier for the community to train LoRAs. This also means that users need suitable inference settings to get the best results. Here are our recommendations:

* **Resolution**: We recommend outputs with **1M, 1.5M, or 2M** total pixels. The model can also extrapolate to arbitrary resolutions from 0.25M (`512Г—512`) to 4M (2K).
* **Aspect Ratio**: We support arbitrary aspect ratios. GradioвЂ™s resolution and aspect-ratio controls calculate dimensions aligned to multiples of 16. In the Python API, or when entering dimensions manually, use positive multiples of 16; invalid dimensions raise an error.
* **Inference Steps**: Best results are achieved with **в‰Ґ 50** steps.
* **CFG**: Default **3.0** (recommended range: 3.0вЂ“6.0). A value of 3.0 is generally sufficient for long prompts; for short prompts, we recommend increasing it as needed.
* **Shift**: Default **5.0**.
* **Sampler**: **Euler** is the recommended default.
* **SLG (Skip Layer Guidance)**: We have implemented two variants:
  * **Standard SLG**: scale 3, 0.01-1.0.
  * **Two-Pass Sampling SLG**: We recommend `uncond skip`, with **22** as the recommended skip layer (0-indexed). Layers 22, 24, and 26 can all help improve image quality.
  
  Turning off SLG will make the output image look very realistic, but it may also cause some details to collapse.

## Prompt Guidelines

MOSS-Image-Base supports multiple input formats, including natural language, tag sequences, and JSON layouts.

* **Language**: We recommend **English**, which has the most training data. Chinese and other languages are also supported.
* **Negative Prompt**: Based on our tests, we recommend the following effective negative tags:
  ```text
  nsfw, worst quality, bad anatomy, watermark, jpeg artifact
  ```
* **Recommended Tag Order**: For the best results with tag sequences, we recommend the order Quality -> Score -> Meta -> Year -> Character -> Series -> Artist -> General. For example:
  ```text
  best quality, highres, hatsune miku, vocaloid, @foo bar, 1girl, blue hair
  ```
* **Text Layout**: The model can render text and insert line breaks automatically, but the result may not always match your expectations. To improve text accuracy or control line breaks precisely, insert `\n` or actual line breaks directly into the prompt.

Here are some recommended example prompts:

```text
prompt:
дёЂдёЄз©їзќЂ"MOSI"ж ‡еї—зљ„TжЃ¤зљ„дё­е›ЅзѕЋеҐіж­Јж‹їзќЂй»‘и‰Ізљ„й©¬е…‹з¬”йќўеђ‘й•ње¤ґеѕ®з¬‘гЂ‚еҐ№иє«еђЋзљ„зЋ»з’ѓжќїдёЉж‰‹е†™дЅ“е†™зќЂ "дёЂгЂЃMOSS-Imageзљ„жЉЂжњЇи·Їзєїпјљ5B зљ„еЌ•жµЃ DiT жЁЎећ‹пјЊеџєдєЋ Muon и®­з»ѓгЂ‚дєЊгЂЃMOSS-Imageзљ„жЁЎећ‹з‰№и‰Іпјљ1гЂЃе¤Ќжќ‚ж–‡е­—жёІжџ“гЂ‚ж”ЇжЊЃдё­и‹±жёІжџ“гЂЃи‡ЄеЉЁеёѓе±Ђпј› 2гЂЃдЅїз”Ё Layout иї›иЎЊзІѕе‡†е›ѕеѓЏжЋ§е€¶пјЊж”ЇжЊЃж–‡е­—зј–иѕ‘е’Њз‰©дЅ“еўће‡Џз­‰и‡Єз”±зј–иѕ‘гЂ‚дё‰гЂЃMOSS-Imageзљ„еђЋз»­еЏ‘е±•пјљжЋўзґў R2I еЏ‚иЂѓз”џе›ѕгЂЃеЉ©еЉ›з”џж€ђејЏAIеЏ‘е±•гЂ‚"

negative prompt:
worst quality, bad anatomy
```

```text
prompt:
A wide view of a refrigerator beside a kitchen counter with a fruit bowl. A small sticky note on the refrigerator door is handwritten in Chinese:
вЂњи®°еѕ—д№°з‰›еҐ¶е’ЊйёЎи›‹пјЃ
з‰›еҐ¶дё¤з›’пјЊйёЎи›‹дёЂж‰“гЂ‚
йЎєдѕїеё¦дёЂиў‹йќўеЊ…гЂ‚вЂќ

negative prompt:

```

### Layout Prompts (Experimental)

MOSS-Image-Base explores layout control for text-to-image generation and currently supports input in the following JSON format:

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

The fields are defined as follows:

- `summary.caption`: A description of the overall scene and style.
- `items`: Each element's category (`label`), description (`caption`), and bounding box (`bbox`). Use `text` to specify the content when rendering text.
- `bbox_format: "yxyx"`: Coordinates are ordered as `[y_min, x_min, y_max, x_max]`.
- `bbox_coord_space: "abs_centralized"`: Absolute pixel coordinates with the canvas center as the origin (x increases to the right and y increases downward). For example, a 1024Г—1024 canvas ranges from -512 to +512 on both axes. For a canvas of width W and height H, subtract W/2 and H/2 from conventional top-left-origin x and y coordinates, respectively, to convert to this coordinate system.

> Layout control uses absolute pixel coordinates across resolutions and aspect ratios.

The model supports combining natural language and structured input.

## Prompt Enhancer and Layout Enhancer

The MOSS-Image series natively supports layouts, enabling many interesting applications and expanding the models' capabilities. This also makes the models more difficult to use.

Layout enhancement can use an LLM with a layout-planning skill or a fine-tuned open-source LLM as a layout enhancer.

## Post-Training

### LoRA Training Recommendations

Supervised fine-tuning can enable GIF generation and reference-based generation in MOSS-Image-Base.

The trainer in this repository is that post-training path: LoRA or full fine-tuning, and reference-image edit training on the reserved RoPE reference axis. Keep the text encoder frozen. Commands are in the Training section of the README.

### Image Captioning

If you post-train MOSS-Image-Base yourself, you can use the following image captioning approach as a reference.

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
    - Use normalized 0вЂ“1000 [y_min, x_min, y_max, x_max] coordinates relative to the full image.
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

Recommended captioning models for MOSS-Image-Base include Gemma 4, Qwen 3.5 27B, and Qwen 3.5 35B-A3B. Use the following request format:

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

We recommend enabling thinking mode for more accurate captions.

#### Tips

1. Adding relevant world knowledge can improve captions.
2. An agentic captioning pipeline can achieve high caption accuracy.
3. Better understanding helps improve generation.

### Few-Step Distillation
