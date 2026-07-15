# download_model.py
from huggingface_hub import hf_hub_download
import shutil, os

os.makedirs("models", exist_ok=True)

path = hf_hub_download(
    repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    local_dir="models",
)
print("Downloaded to:", path)