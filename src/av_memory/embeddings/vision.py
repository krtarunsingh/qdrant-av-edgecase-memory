import os
from functools import lru_cache

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel

from ..config import SETTINGS


DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


@lru_cache(maxsize=1)
def _load_clip() -> tuple[CLIPImageProcessor, CLIPModel, torch.device]:
    model_id = os.getenv("AV_VISION_MODEL_ID", DEFAULT_CLIP_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = CLIPImageProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.eval()
    model.to(device)
    return processor, model, device


def vision_embed(img: Image.Image) -> list[float]:
    processor, model, device = _load_clip()
    inputs = processor(images=img.convert("RGB"), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.inference_mode():
        outputs = model.get_image_features(pixel_values=pixel_values)
        features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
        features = F.normalize(features, p=2, dim=-1)

    v = features.squeeze(0).detach().cpu().to(dtype=torch.float32).numpy()
    if v.shape[0] != SETTINGS.vision_dim:
        raise ValueError(
            f"CLIP image embedding dim={v.shape[0]} does not match SETTINGS.vision_dim={SETTINGS.vision_dim}. "
            "Update the configured dimension and recreate the collection."
        )
    return v.tolist()

