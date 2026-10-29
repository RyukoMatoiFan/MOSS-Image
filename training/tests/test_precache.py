"""Cache completion must release input mappings and stream completed examples."""

from pathlib import Path

import pytest
import torch

from training import precache as module
from training.cache import read_example, read_index


def test_precache_streams_images_then_contexts(tmp_path, monkeypatch):
    class FakeVAE:
        def eval(self):
            return self

        def to(self, device):
            return self

    monkeypatch.setattr(module, "bundle_paths", lambda path: (path, path, path))
    monkeypatch.setattr(module.AutoencoderKLFlux2, "from_pretrained", lambda *a, **k: FakeVAE())
    monkeypatch.setattr(module, "_encode_image", lambda *a: (torch.ones(4, 2, 2), 32, 32))
    output = tmp_path / "cache"

    def encode(prompts, *args, **kwargs):
        assert prompts == ["first", "second", ""]
        assert len(list(output.glob("*.safetensors"))) == 2
        for index in range(3):
            if index:
                previous = read_example(output / f"{index - 1:06d}.safetensors")
                assert previous["context"].shape == (index, 16)
            yield torch.full((index + 1, 16), float(index))

    monkeypatch.setattr(module, "iter_encode_prompts", encode)
    jobs = [{"target": Path("target.png"), "refs": [Path("ref.png")], "prompt": prompt}
            for prompt in ("first", "second")]
    module.precache(tmp_path, output, jobs, 32, torch.device("cpu"), 0)
    assert len(read_index(output)) == 2
    for index, prompt in enumerate(("first", "second")):
        example = read_example(output / f"{index:06d}.safetensors")
        assert example["prompt"] == prompt
        assert len(example["refs"]) == 1
        assert example["context"].shape == (index + 1, 16)


def test_precache_preserves_an_existing_cache(tmp_path):
    index = tmp_path / "index.jsonl"
    index.write_text("existing index", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Cache already exists"):
        module.precache(tmp_path / "missing-bundle", tmp_path, [], 32, torch.device("cpu"), 0)
    assert index.read_text(encoding="utf-8") == "existing index"
