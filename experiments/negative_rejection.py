import yaml

from src.datasets.UniversalDataset import UniversalDataset
from src.evaluation.NegativeJudge import NegativeJudge
from src.generator import OpenAICompGenerator, GemmaGenerator, OllamaGenerator
from src.retriever.NegativeRejection import NegativeRejection
from src.utils.QA_Selector import QA_Selector

# Load experiment settings from YAML file
with open("../configs/negative_rejection.yaml", "r") as f:
    settings = yaml.safe_load(f)

# Initialize components based on settings
if settings.get("dataset_source") == "huggingface":
    dataset = UniversalDataset.from_huggingface(
        dataset_path=settings.get("dataset_path"),
        dataset_name=settings.get("dataset_name"),
        split=settings.get("split")
    )
else:
    dataset = UniversalDataset.from_json(
        json_path=settings.get("file_path"),
        id_field=settings.get("id_field"),
        question_field=settings.get("question_field"),
        answer_field=settings.get("answer_field")
    )

selector = QA_Selector(settings.get("number_of_questions"), dataset, seed = settings.get("random_seed"))

models = settings.get("models")
output_file_base = settings.get("output_file")

retriever = NegativeRejection()
similarity_threshold = settings.get("similarity_threshold", 0.8)

# Generate evaluation lists once (they're the same for all models)
questions, contexts_list, ground_truths, question_ids = selector.generate_evaluation_lists(
    retriever=retriever,
    top_k=5
)

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

    output_file = f"../output/negative_rejection/{output_file_base}_{model.get('name').replace('/', '_')}.json"

    judge = NegativeJudge(similarity_threshold=similarity_threshold)
    results = judge.bulk_evaluation(
        questions=questions,
        contexts_list=contexts_list,
        question_ids=question_ids,
        generator=generator,
        output_file=output_file
    )
