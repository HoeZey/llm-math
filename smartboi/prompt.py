class Prompt:
    def __init__(self, problem_prompt: str, question_format: str) -> None:
        self.problem_prompt = problem_prompt
        self.question_format = question_format

    def insert_question(self, question: str) -> str:
        return f'{self.problem_prompt}\n{self.question_format.replace('<question>', question)}'
        
    def insert_question_few_shot(self, question: str, fewshot_examples: dict[str, str]) -> str:
        examples = '\n'.join([self._create_fs_example(q, a) for q, a in fewshot_examples.items()])
        question_formatted = self.question_format.replace('<question>', question).replace('<answer>', '')
        return f'{self.problem_prompt}\n{examples}\n{question_formatted}'
    
    def _create_fs_example(self, question: str, answer: str) -> str:
        return self.question_format.replace('<question>', question).replace('<answer>', answer)
    