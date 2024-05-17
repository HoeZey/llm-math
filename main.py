import torch
from smartboi.models import Gemma


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Gemma()
    out = model.generate('hello!', max_new_tokens=20)
    print(out)


if __name__ == '__main__':
    main()
