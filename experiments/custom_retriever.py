import yaml
import json
from src.evaluation.LLMJudge import LLMJudge
from src.generator import OpenAICompGenerator, GemmaGenerator, OllamaGenerator

# Load experiment settings from YAML file
with open("../configs/custom_retriever.yaml", "r") as f:
    settings = yaml.safe_load(f)

# Load data from JSON file
json_file_path = "../data/retrieval_results.json"
with open(json_file_path, "r") as f:
    data = json.load(f)

# Create evaluation lists from JSON data
questions = [item["question"] for item in data]
contexts_list = [item["retrieved_context"] for item in data]
ground_truths = [item["ground_truths"] for item in data]
question_ids = [item["id"] for item in data]

models = settings.get("models")
output_file_base = settings.get("output_file")

# Generator factory
def create_generator(generator_type, provider_name, model_name):
    if generator_type == "openai_compatible":
        return OpenAICompGenerator(provider_name=provider_name, model_name=model_name)
    elif generator_type == "google":
        return GemmaGenerator(model_name=model_name)
    elif generator_type == "ollama":
        return OllamaGenerator(model_name=model_name)
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")

for model in models:
    generator = create_generator(model.get("generator"), model.get("provider"), model.get("name"))

    output_file = f"../output/custom_retriever/{output_file_base}_{model.get('name').replace('/', '_')}.json"

    judge = LLMJudge()
    results = judge.bulk_evaluation(
        questions=questions,
        contexts_list=contexts_list,
        ground_truths=ground_truths,
        question_ids=question_ids,
        generator=generator,
        output_file=output_file
    )
