
# ---- Base Image ----
    FROM python:3.10-slim

    # ---- Set Workdir ----
    WORKDIR /workspace
    
    # ---- Install Linux packages (if needed) ----
    RUN apt-get update && apt-get install -y \
        git \
        && rm -rf /var/lib/apt/lists/*
    
    # ---- Install Python dependencies early (cache benefit) ----
    COPY requirements.txt ./
    RUN pip install --upgrade pip && pip install -r requirements.txt

    