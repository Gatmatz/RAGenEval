from datasets import load_dataset


class QA_Selector:
    """
    Utility class to select questions from the HotpotQA dataset.
    """
    def __init__(self, num_questions, seed=42):
        self.num_questions = num_questions
        self.seed = seed
        self.dataset = load_dataset("hotpot_qa", "distractor")

    def get_question_ids(self):
        """
        Returns a list of question IDs from the HotpotQA dataset.
        """
        shuffled = self.dataset['train'].shuffle(seed=self.seed)
        questions = shuffled.select(range(self.num_questions))
        return [question['id'] for question in questions]

    def get_dataset(self):
        return self.dataset['train']