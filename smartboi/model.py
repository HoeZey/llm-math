import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Union


class Gemma(torch.nn.Module):
    def __init__(self) -> None:
        super.__init__()
        self.model = AutoModelForCausalLM.from_pretrained("google/gemma-7b", device_map='auto', revision='float16')
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")

    def forward(self, prompts: Union[str, list[str]], **gen_kwargs) -> dict:
        return model.generate(**self.tokenizer(prompts, return_tensors), **gen_kwargs)


class Llama3(torch.nn.Module):
    def __init__(self) -> None:
        super.__init__()
        pipeline = transformers.pipeline(
            "text-generation",
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="cuda",
        )

        # messages = [
        #     {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        #     {"role": "user", "content": "Who are you?"},
        # ]

        # prompt = pipeline.tokenizer.apply_chat_template(
        #         messages, 
        #         tokenize=False, 
        #         add_generation_prompt=True
        # )

        # terminators = [
        #     pipeline.tokenizer.eos_token_id,
        #     pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        # ]

        # outputs = pipeline(
        #     prompt,
        #     max_new_tokens=256,
        #     eos_token_id=terminators,
        #     do_sample=True,
        #     temperature=0.6,
        #     top_p=0.9,
        # )
        # print(outputs[0]["generated_text"][len(prompt):])


    def forward(self, x):
        pass
