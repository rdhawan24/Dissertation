from pathlib import Path
from gliner import GLiNER
import onnxruntime as ort

print("ONNX-RT device:", ort.get_device())      # sanity check (CPU / GPU)

model_dir = Path(
    "/home/mohan/Software/Personal/Dissertation/hf_models/"
    "pii_model/gliner_multi_pii_onnx"           # ABSOLUTE path → safer
)

model = GLiNER.from_pretrained(
    model_dir.as_posix(),
    load_onnx_model=True,
    load_tokenizer=True,
    onnx_model_file="onnx/model.onnx",      # <— no leading “./”
    local_files_only=True                       # stays offline
)

text   = "Call me at +1-415-555-0123 or send ₹10,000 to ACME Corp."
labels = ["phone number", "currency", "organisation"]

for ent in model.predict_entities(text, labels, threshold=0.3):
    print(f"{ent['text']} → {ent['label']}")
