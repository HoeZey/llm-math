import os
import torch
from huggingface_hub import login
from dotenv import load_dotenv
from smartboi.models import Gemma


def main():
    load_dotenv('.env')
    login(token=os.getenv('HF_ACCESS_TOKEN'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = Gemma()
    # out = model.generate('hello!', max_new_tokens=20)
    # print(out)


if __name__ == '__main__':
    main()
