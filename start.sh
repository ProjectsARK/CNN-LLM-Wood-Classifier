#!/bin/bash

# 1. Download model dan tokenizer dari Google Drive jika belum ada
python download_models.py

# 2. Jalankan server FastAPI dengan Uvicorn (Render default port: 10000)
uvicorn main:app --host 0.0.0.0 --port 10000
