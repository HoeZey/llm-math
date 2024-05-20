class PromptFormatter:
    def __init__(self, problem_prompt: str, question_format: str, replace_answer_token: bool=False) -> None:
        self.problem_prompt = problem_prompt
        self.question_format = question_format
        self.replace_answer_token = replace_answer_token

    def insert_question(self, question: str) -> str:
        question_formatted = self.question_format.replace('<question>', question)
        if self.replace_answer_token:
            question_formatted = question_formatted.replace('<answer>', '')
        return f'{self.problem_prompt}\n{question_formatted}'
        
    def insert_question_few_shot(self, question: str, fewshot_examples: dict[str, str]) -> str:
        examples = '\n'.join([self._create_fs_example(q, a) for q, a in fewshot_examples.items()])
        question_formatted = self.question_format.replace('<question>', question)
        if self.replace_answer_token:
            question_formatted = question_formatted.replace('<answer>', '')
        return f'{self.problem_prompt}\n{examples}\n{question_formatted}'
    
    def _create_fs_example(self, question: str, answer: str) -> str:
        return self.question_format.replace('<question>', question).replace('<answer>', answer)
    