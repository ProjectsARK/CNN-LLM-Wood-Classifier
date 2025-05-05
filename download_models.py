import gdown
import os

# Folder tujuan
os.makedirs("models", exist_ok=True)

# Link Google Drive (gunakan file ID)
cnn_id = "1BzKXslQz5kFvDRY9BQGziYZs3Uj1c8qV"
llm_id = "1AbCdEfGhIJkLmnOpQrSTuVWxyz123456"

cnn_output = "models/model_cnn.pth"
llm_output = "models/distilbert.pth"

if not os.path.exists(cnn_output):
    gdown.download(f"https://drive.google.com/uc?id={cnn_id}", cnn_output, quiet=False)

if not os.path.exists(llm_output):
    gdown.download(f"https://drive.google.com/uc?id={llm_id}", llm_output, quiet=False)