import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Union
from smartboi.prompt import Prompt


class Gemma:
    def __init__(self, variant='7b-it', system_prompt=None) -> None:
        if variant == '7b-it':
            model_name = 'google/gemma-7b-it'
        elif variant == '2b-it':
            model_name = 'google/gemma-2b-it'
        else:
            NotImplementedError('This model variant is not implemented.')

        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.system_prompt = system_prompt

    def generate(self, prompts: list[Prompt], **gen_kwargs) -> dict:
        out = self.model.generate(
            **self.tokenizer(prompts, return_tensors='pt').to(self.model.device), 
            **gen_kwargs
        )
        return self.tokenizer.decode(out[0])


class DeepSeekMathIt:
    def __init__(self, system_prompt=None) -> None:
        model_name = 'deepseek-ai/deepseek-math-7b-instruct'
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map='auto', 
            torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.generation_config = GenerationConfig.from_pretrained(model_name)
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        self.system_prompt

    def generate(self, prompts: list[Prompt], **gen_kwargs):
        messages = [{'role': 'system', 'content': self.system_prompt}] if self.system_prompt is not None else []
        messages.append({'role': 'user', 'content': prompt})

        input_tensor = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt').to(self.model.device)
        outputs = self.model.generate(input_tensor, **gen_kwargs)

        return self.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)


# class Llama3(torch.nn.Module):
#     def __init__(self) -> None:
#         super.__init__()
#         pipeline = transformers.pipeline(
#             'text-generation',
#             model='meta-llama/Meta-Llama-3-8B-Instruct',
#             model_kwargs={'torch_dtype': torch.bfloat16},
#             device_map='auto',
#         )

        # messages = [
        #     {'role': 'system', 'content': 'You are a pirate chatbot who always responds in pirate speak!'},
        #     {'role': 'user', 'content': 'Who are you?'},
        # ]

        # prompt = pipeline.tokenizer.apply_chat_template(
        #         messages, 
        #         tokenize=False, 
        #         add_generation_prompt=True
        # )

        # terminators = [
        #     pipeline.tokenizer.eos_token_id,
        #     pipeline.tokenizer.convert_tokens_to_ids('<|eot_id|>')
        # ]

        # outputs = pipeline(
        #     prompt,
        #     max_new_tokens=256,
        #     eos_token_id=terminators,
        #     do_sample=True,
        #     temperature=0.6,
        #     top_p=0.9,
        # )
        # print(outputs[0]['generated_text'][len(prompt):])


    # def forward(self, x):
    #     pass
