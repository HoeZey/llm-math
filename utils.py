import os
from huggingface_hub import login
from dotenv import load_dotenv


def login_to_hf_from_env_token():
    load_dotenv('.env')
    login(token=os.getenv('HF_ACCESS_TOKEN'))
