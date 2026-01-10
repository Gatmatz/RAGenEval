import yaml

from src.evaluation.LLMJudge import LLMJudge
from src.generator import OpenAICompatibleGenerator, GoogleGenerator, GroqGenerator
from src.retriever.NoiseRobustness import NoiseRobustness
from src.utils.QA_Selector import QA_Selector

# Load settings from YAML file
with open("../configs/noise_robustness.yaml", "r") as f:
    settings = yaml.safe_load(f)

selector = QA_Selector(settings.get("number_of_questions"))
models = settings.get("models")
output_file_base = settings.get("output_file")
noise_ratios = settings.get("noise_ratios", [0.5])  # Default to [0.5] if not specified

for noise_ratio in noise_ratios:
    retriever = NoiseRobustness(noise_ratio=noise_ratio)

    for model in models:
        generator = GroqGenerator(model_name=model.get("name"))

        output_file = f"../output/noise_robustness/{output_file_base}_{model.get('name')}_noise_{noise_ratio}.json"

        judge = LLMJudge()
        results = judge.evaluate_dataset(
            dataset=selector.get_dataset(),
            qa_ids=selector.get_question_ids(),
            generator=generator,
            retriever=retriever,
            output_file=output_file
        )
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

    output_file = f"../output/noise_robustness/{output_file_base}_{model.get('generator')}_{model.get('name').replace('/', '_')}.json"

