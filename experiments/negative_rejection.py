import yaml

from src.evaluation.LLMJudge import LLMJudge
from src.evaluation.NegativeJudge import NegativeJudge
from src.generator import OpenAICompatibleGenerator, GoogleGenerator
from src.retriever.NegativeRejection import NegativeRejection
from src.retriever.NoiseRobustness import NoiseRobustness
from src.retriever.PerfectContext import PerfectContext
from src.utils.QA_Selector import QA_Selector

# Load settings from YAML file
with open("../configs/negative_rejection.yaml", "r") as f:
    settings = yaml.safe_load(f)

selector = QA_Selector(settings.get("number_of_questions"))
models = settings.get("models")
output_file_base = settings.get("output_file")

retriever = NegativeRejection()

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

    output_file = f"../output/negative_rejection/{output_file_base}_{model.get('generator')}_{model.get('name').replace('/', '_')}.json"
    judge = NegativeJudge()

    results = judge.evaluate_dataset(
        dataset=selector.get_dataset(),
        qa_ids=selector.get_question_ids(),
        generator=generator,
        retriever=retriever,
        output_file=output_file
    )
