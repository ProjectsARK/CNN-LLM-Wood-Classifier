from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import BertForSequenceClassification, AutoTokenizer
import io
import json

app = FastAPI()

# CORS (opsional, untuk frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== DEVICE SETUP ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== LOAD CLASS CONFIG ==========
with open("class_names.json") as f:
    class_names = json.load(f)
num_labels = len(class_names)

with open("id2label.json") as f:
    id2label = json.load(f)

with open("label2id.json") as f:
    label2id = json.load(f)

# ========== LOAD TOKENIZER ==========
tokenizer = AutoTokenizer.from_pretrained("distilbert_tokenizer")

# ========== CNN MODEL ==========
cnn_model = models.mobilenet_v3_large(pretrained=False)
cnn_model.classifier[3] = nn.Linear(cnn_model.classifier[3].in_features, num_labels)
cnn_model.load_state_dict(torch.load("mobilenet_v3.pth", map_location=device))
cnn_model.eval().to(device)

# ========== LLM MODEL ==========
llm_model = BertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels,
    problem_type="multi_label_classification",
    id2label=id2label,
    label2id=label2id
)
llm_model.load_state_dict(torch.load("distilbert.pth", map_location=device))
llm_model.eval().to(device)

# ========== IMAGE TRANSFORM ==========
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ========== PREDICT ROUTE ==========
@app.post("/predict")
async def predict(image: UploadFile = File(...), text: str = Form("")):
    try:
        # Proses gambar
        img_bytes = await image.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = img_transform(img).unsqueeze(0).to(device)
    except:
        raise HTTPException(status_code=400, detail="Gambar tidak valid.")

    use_text = text.strip() != ""
    if use_text:
        tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        # CNN inference
        logits_cnn = cnn_model(img_tensor)
        probs_cnn = F.softmax(logits_cnn, dim=1)

        if use_text:
            # LLM inference
            logits_llm = llm_model(**tokens).logits
            probs_llm = torch.sigmoid(logits_llm)

            # Gabungkan probabilitas
            weight_cnn = 0.75
            weight_llm = 0.25
            combined = (weight_cnn * probs_cnn) + (weight_llm * probs_llm)
        else:
            combined = probs_cnn  # fallback ke CNN full

        combined = combined.squeeze()
        probs_list = combined.cpu().tolist()

        # Validasi: jika semua rendah, anggap tidak relevan
        if max(probs_list) < 0.2:
            raise HTTPException(status_code=422, detail="Tidak ada label yang relevan terdeteksi.")

        # Ambil hanya 1 label dengan probabilitas tertinggi
        top_idx = torch.argmax(combined).item()
        top_label = id2label[str(top_idx)]
        top_prob = round(probs_list[top_idx], 4)

    return {
        "final_label": top_label,
        "probability": top_prob,
        "used_text_input": use_text
    }
