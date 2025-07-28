from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

api = HfApi()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set.")
print(api.whoami(token=hf_token))