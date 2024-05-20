import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import Union


class Gemma:
    def __init__(self, variant='7b-it') -> None:
        if variant == '7b-it':
            model_name = 'google/gemma-7b-it'
        elif variant == '2b-it':
            model_name = 'google/gemma-2b-it'
        else:
            NotImplementedError('This model variant is not implemented.')

        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, prompts: list[str], **gen_kwargs) -> dict:
        out = self.model.generate(
            **self.tokenizer(prompts, return_tensors='pt').to(self.model.device), 
            **gen_kwargs
        )
        return self.tokenizer.decode(out[0])


class DeepSeekMathIt:
    def __init__(self) -> None:
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

    def generate(self, prompts: list[str], **gen_kwargs):
        messages = [{'role': 'system', 'content': self.system_prompt}] if self.system_prompt is not None else []
        messages.append({'role': 'user', 'content': prompt})

        input_tensor = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt').to(self.model.device)
        outputs = self.model.generate(input_tensor, **gen_kwargs)

        return self.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
