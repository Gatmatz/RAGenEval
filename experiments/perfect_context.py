import yaml
from time import sleep

from src.evaluation.LLMJudge import LLMJudge
from src.generator.GroqGenerator import GroqGenerator
from src.retriever.PerfectContext import PerfectContext
from src.utils.QA_Selector import QA_Selector

# Load experiment settings from YAML file
with open("../configs/perfect_context.yaml", "r") as f:
    settings = yaml.safe_load(f)

# Initialize components based on settings
selector = QA_Selector(settings.get("number_of_questions"))
models = settings.get("models")
output_file_base = settings.get("output_file")

retriever = PerfectContext()

for model in models:
    generator = GroqGenerator(model_name=model.get("name"))

    output_file = f"../output/perfect_context/{output_file_base}_{model.get('name')}.json"

    judge = LLMJudge()
    results = judge.evaluate_dataset(
        dataset=selector.get_dataset(),
        qa_ids=selector.get_question_ids(),
        generator=generator,
        retriever=retriever,
        output_file=output_file
    )
    sleep(10)  # Pause between model evaluations
    
