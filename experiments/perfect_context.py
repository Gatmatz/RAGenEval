import yaml

from src.evaluation.LLMJudge import LLMJudge
from src.generator import MockGenerator
from src.retriever.PerfectContext import PerfectContext
from src.utils.QA_Selector import QA_Selector

# Load settings from YAML file
with open("../configs/perfect_context.yaml", "r") as f:
    settings = yaml.safe_load(f)

selector = QA_Selector(settings.get("number_of_questions"))
models = settings.get("models")
output_file_base = settings.get("output_file")

retriever = PerfectContext()

for model in models:
    generator = MockGenerator()

    output_file = f"../output/{output_file_base}_{model.get('name')}.json"

    judge = LLMJudge()
    results = judge.evaluate_dataset(
        dataset=selector.get_dataset(),
        qa_ids=selector.get_question_ids(),
        generator=generator,
        retriever=retriever,
        output_file=output_file
    )
