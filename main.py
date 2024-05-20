import torch
from smartboi.models import Gemma
from smartboi.prompt import PromptFormatter
from utils import login_to_hf_from_env_token


def main():
    login_to_hf_from_env_token()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    formatter = PromptFormatter(
        problem_prompt='You are a mathematical question answering assistant solving difficult math problems.\nAnswer the following question. Give a step-by-step explanation and then give your answer where the <answer> token is.',
        question_format='Question: <question>. Answer: <answer>',
        replace_answer_token=False
    )
    prompt = formatter.insert_question('Let $a=1$ and $b=2$. What is $a+b$?')
    print(prompt)
    # model = Gemma()
    # out = model.generate('hello!', max_new_tokens=20)
    # print(out)


if __name__ == '__main__':
    main()
