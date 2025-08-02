from pathlib import Path
from gliner import GLiNER
import onnxruntime as ort

print("ONNX-RT device:", ort.get_device())      # sanity check (CPU / GPU)

model_dir = Path("../hf_models/pii_model/gliner_multi_pii_onnx/")

model = GLiNER.from_pretrained(
    model_dir.as_posix(),
    load_onnx_model=True,
    load_tokenizer=True,
    onnx_model_file="onnx/model.onnx",      # <— no leading “./”
    local_files_only=True                       # stays offline
)

labels = [
    "person",
    "organization",
    "phone number",
    "address",
    "passport number",
    "email",
    "credit card number",
    "social security number",
    "health insurance id number",
    "date of birth",
    "mobile phone number",
    "bank account number",
    "medication",
    "cpf",
    "driver's license number",
    "tax identification number",
    "medical condition",
    "identity card number",
    "national id number",
    "ip address",
    "email address",
    "iban",
    "credit card expiration date",
    "username",
    "health insurance number",
    "registration number",
    "student id number",
    "insurance number",
    "flight number",
    "landline phone number",
    "blood type",
    "cvv",
    "reservation number",
    "digital signature",
    "social media handle",
    "license plate number",
    "cnpj",
    "postal code",
    "passport_number",
    "serial number",
    "vehicle registration number",
    "credit card brand",
    "fax number",
    "visa number",
    "insurance company",
    "identity document number",
    "transaction number",
    "national health insurance number",
    "cvc",
    "birth certificate number",
    "train ticket number",
    "passport expiration date",
    "social_security_number"
]


text   = "Call me at +1-415-555-0123 or send ₹10,000 to ACME Corp.   Visa 4128 0033 2341 1978 exp  10/30"
# labels = ["phone number", "currency", "organisation"]

for ent in model.predict_entities(text, labels, threshold=0.3):
    print(f"{ent['text']} → {ent['label']}")
