import urllib.request
from huggingface_hub import snapshot_download
import os

print("Downloading files...")

url = "https://huggingface.co/MohamedRashad/Arabic-Orpo-Llama-3-8B-Instruct-GGUF/resolve/main/arabic-orpo-llama-3-8b-instruct.Q4_K_M.gguf?download=true"
output_file = "arabic-orpo-llama-3-8b-instruct.Q4_K_M.gguf"

if not os.path.exists(output_file):
    print(f"Downloading file from {url}...")
    urllib.request.urlretrieve(url, output_file)
    print(f"File downloaded and saved as {output_file}")

url = "https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-gguf/resolve/main/llava-llama-3-8b-v1_1-int4.gguf?download=true"
output_file = "llava-llama-3-8b-v1_1-int4.gguf"

if not os.path.exists(output_file):
    print(f"Downloading file from {url}...")
    urllib.request.urlretrieve(url, output_file)
    print(f"File downloaded and saved as {output_file}")

url = "https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-gguf/resolve/main/llava-llama-3-8b-v1_1-mmproj-f16.gguf?download=true"
output_file = "llava-llama-3-8b-v1_1-mmproj-f16.gguf"

if not os.path.exists(output_file):
    print(f"Downloading file from {url}...")
    urllib.request.urlretrieve(url, output_file)
    print(f"File downloaded and saved as {output_file}")

repo_id = "openai/whisper-large-v3"
local_dir = "whisper-large-v3"
if not os.path.exists(local_dir):
    print(f"Downloading model {repo_id} from Hugging Face Hub...")
    snapshot_download(repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
    print(f"Model downloaded and saved as {local_dir}")

repo_id = "Helsinki-NLP/opus-mt-tc-big-en-ar"
local_dir = "opus-mt-tc-big-en-ar"
if not os.path.exists(local_dir):
    print(f"Downloading model {repo_id} from Hugging Face Hub...")
    snapshot_download(repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
    print(f"Model downloaded and saved as {local_dir}")

repo_id = "Shularp/model-translate-ar-to-en-from-320k-dataset-ar-en-th2301191019"
local_dir = "model-translate-ar-to-en-from-320k-dataset-ar-en-th2301191019"
if not os.path.exists(local_dir):
    print(f"Downloading model {repo_id} from Hugging Face Hub...")
    snapshot_download(repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
    print(f"Model downloaded and saved as {local_dir}")

print("Setup completed.")
