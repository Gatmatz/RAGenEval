import yaml
from time import sleep

from src.evaluation.LLMJudge import LLMJudge
from src.generator import OpenAICompatibleGenerator, GoogleGenerator
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

# Generator factory
def create_generator(generator_type, model_name):
    if generator_type == "openai_compatible":
        return OpenAICompatibleGenerator(model_name=model_name)
    elif generator_type == "google":
        return GoogleGenerator(model_name=model_name)
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")

for model in models:
    generator = create_generator(model.get("generator"), model.get("name"))

    output_file = f"../output/perfect_context/{output_file_base}_{model.get('generator')}_{model.get('name').replace('/', '_')}.json"

    judge = LLMJudge()
    results = judge.evaluate_dataset(
        dataset=selector.get_dataset(),
        qa_ids=selector.get_question_ids(),
        generator=generator,
        retriever=retriever,
        output_file=output_file
    )
    sleep(10)  # Pause between model evaluations
    
