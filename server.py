from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import time
import uuid
import sys
from pathlib import Path

# Set path for loading the V6 model
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config_v6 import get_config
from models.model_v6 import build_model_v6

app = FastAPI()

# [Initialization] Load and warmup the V6 model
print("Loading V6 segmentation model...")

device = torch.device("cpu")
checkpoint_path = PROJECT_ROOT / "best_phase2.pth"

# Build model architecture and load weights
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint["model_state_dict"]

# Initialize configuration
run_phase = checkpoint.get("phase", 2)
backbone = checkpoint.get("backbone", "convnext_tiny")
cfg = get_config(phase=run_phase, backbone=backbone)

# Infer and overwrite the number of head classes
part_key = next(k for k in state_dict.keys() if "part_head.classifier.weight" in k)
damage_key = next(k for k in state_dict.keys() if "damage_head.classifier.weight" in k)
cfg.part_head.num_classes = state_dict[part_key].shape[0]
cfg.damage_head.num_classes = state_dict[damage_key].shape[0]

# Load injection settings saved in the checkpoint
cfg.damage_head.part_injection = checkpoint.get("part_injection", "gate")
cfg.part_head.viewpoint_injection_mode = checkpoint.get("viewpoint_injection_mode", "concat")
cfg.damage_head.part_feature_injection_mode = checkpoint.get("part_feature_injection_mode", "gate")
cfg.part_head.enable_viewpoint_injection = bool(checkpoint.get("enable_viewpoint_injection", True))
cfg.damage_head.enable_part_feature_injection = bool(checkpoint.get("enable_part_feature_injection", True))

# Build V6 model
model = build_model_v6(cfg)
model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()}, strict=False)
model.to(device).eval()

# Preprocessing settings (ImageNet)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
IMAGE_SIZE = 512

# Damage class name mapping
DAMAGE_CLASSES = ["bg", "dent", "scratch", "crack", "broken_glass", "broken_light", "broken_part", "tire_flat"]

print("Server ready and V6 model loaded.")


# [Endpoint] Synchronous single communication for latency measurement
@app.post("/infer", response_class=JSONResponse)
async def infer_v6(file: UploadFile = File(...)):
    server_start_time = time.perf_counter()
    
    # 1.Decode
    decode_start = time.perf_counter()
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    original_size = image.size
    decode_ms = (time.perf_counter() - decode_start) * 1000
    
    # 2.Preprocess
    pre_start = time.perf_counter()
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    img_np = np.array(resized, copy=True).transpose(2, 0, 1)
    x = torch.from_numpy(img_np).float().to(device) / 255.0
    x = (x - MEAN) / STD
    x = x.unsqueeze(0)
    pre_ms = (time.perf_counter() - pre_start) * 1000
    
    # 3.Inference
    infer_start = time.perf_counter()
    with torch.no_grad():
        out = model(x, phase=run_phase)
    infer_ms = (time.perf_counter() - infer_start) * 1000
    
    # 4.Postprocess
    post_start = time.perf_counter()
    damage_summary = []
    
    if "damage_logits" in out:
        logits = F.interpolate(out["damage_logits"], size=(original_size[1], original_size[0]), mode="bilinear", align_corners=False)
        damage_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        
        total_pixels = max(damage_mask.size, 1)
        values, counts = np.unique(damage_mask, return_counts=True)
        
        for val, count in zip(values, counts):
            if val == 0: continue 
            class_name = DAMAGE_CLASSES[val] if val < len(DAMAGE_CLASSES) else f"class_{val}"
            damage_summary.append({
                "class_name": class_name,
                "pixels": int(count),
                "ratio": round(float(count / total_pixels) * 100, 2) 
            })
            
        damage_summary.sort(key=lambda x: x["pixels"], reverse=True)

    post_ms = (time.perf_counter() - post_start) * 1000
    server_total_ms = (time.perf_counter() - server_start_time) * 1000

    response_data = {
        "status": "ok",
        "request_id": str(uuid.uuid4()),
        "server_timing_ms": {
            "decode_ms": round(decode_ms, 2),
            "pre_ms": round(pre_ms, 2),
            "infer_ms": round(infer_ms, 2),
            "post_ms": round(post_ms, 2),
            "total_ms": round(server_total_ms, 2)
        },
        "result": {
            "topk": damage_summary[:3],
            "all_damages": damage_summary
        }
    }
    
    return response_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)