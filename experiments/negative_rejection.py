import yaml
from time import sleep
from src.evaluation.NegativeJudge import NegativeJudge
from src.generator import OpenaiCompGenerator, GemmaGenerator
from src.retriever.NegativeRejection import NegativeRejection
from src.utils.QA_Selector import QA_Selector

# Load settings from YAML file
with open("../configs/negative_rejection.yaml", "r") as f:
    settings = yaml.safe_load(f)

selector = QA_Selector(settings.get("number_of_questions"))
models = settings.get("models")
output_file_base = settings.get("output_file")
similarity_threshold = settings.get("similarity_threshold")

retriever = NegativeRejection()


# Generator factory
def create_generator(generator_type, provider_name, model_name):
    if generator_type == "openai_compatible":
        return OpenaiCompGenerator(provider_name=provider_name, model_name=model_name)
    elif generator_type == "google":
        return GemmaGenerator(model_name=model_name)
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")


for model in models:
    generator = create_generator(model.get("generator"), model.get("provider"), model.get("name"))

    output_file = f"../output/negative_rejection/{output_file_base}_{model.get('name')}.json"
    judge = NegativeJudge(similarity_threshold=similarity_threshold)
    output_file = f"../output/negative_rejection/{output_file_base}_{model.get('generator')}_{model.get('name').replace('/', '_')}.json"

    results = judge.evaluate_dataset(
        dataset=selector.get_dataset(),
        qa_ids=selector.get_question_ids(),
        generator=generator,
        retriever=retriever,
        output_file=output_file
    )
    sleep(5)

