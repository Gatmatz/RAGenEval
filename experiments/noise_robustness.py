import yaml

from src.evaluation.LLMJudge import LLMJudge
from src.generator.GroqGenerator import GroqGenerator
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

