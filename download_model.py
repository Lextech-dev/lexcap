from huggingface_hub import hf_hub_download
from onnxruntime.quantization import quantize_dynamic, QuantType
import shutil
import os

os.makedirs("./model", exist_ok=True)

# ── Download ──────────────────────────────────────────────────────────────────
configs = [
    {"repo_id": "toandev/OCR-for-Captcha", "repo_type": "model"},
    {"repo_id": "toandev/OCR-for-Captcha", "repo_type": "space"},
    {"repo_id": "toandev/OCR-for-Captcha", "repo_type": "dataset"},
]

model_path = None
for cfg in configs:
    try:
        print(f"Trying repo_type='{cfg['repo_type']}'...")
        model_path = hf_hub_download(
            repo_id=cfg["repo_id"],
            filename="model.onnx",
            repo_type=cfg["repo_type"]
        )
        print(f"✅ Found with repo_type='{cfg['repo_type']}'")
        break
    except Exception as e:
        print(f"❌ Failed: {e}")

if not model_path:
    raise RuntimeError("Could not find model.onnx in any repo type")

print("Saving original model...")
shutil.copy(model_path, "./model/model_original.onnx")

# QUInt8 — reverted from QInt8 since ConvInteger nodes are not supported
# by CPUExecutionProvider on this model architecture in onnxruntime 1.23+
print("Quantizing to INT8...")
quantize_dynamic(
    model_input="./model/model_original.onnx",
    model_output="./model/model.onnx",
    weight_type=QuantType.QUInt8,
)

os.remove("./model/model_original.onnx")
print("✅ Quantized ONNX model saved to ./model/model.onnx")
