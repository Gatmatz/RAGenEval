import yaml

from src.datasets.UniversalDataset import UniversalDataset
from src.evaluation.LLMJudge import LLMJudge
from src.generator import OpenAICompGenerator, GemmaGenerator, OllamaGenerator
from src.retriever.NoiseRobustness import NoiseRobustness
from src.utils.QA_Selector import QA_Selector

# Load experiment settings from YAML file
with open("../configs/noise_robustness.yaml", "r") as f:
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
noise_ratios = settings.get("noise_ratios")
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

# Loop through each noise ratio
for noise_ratio in noise_ratios:
    print(f"\n{'='*60}")
    print(f"Processing noise ratio: {noise_ratio}")
    print(f"{'='*60}\n")

    retriever = NoiseRobustness(noise_ratio)

    # Generate evaluation lists for this noise ratio
    questions, contexts_list, ground_truths, question_ids = selector.generate_evaluation_lists(
        retriever=retriever,
        top_k=5
    )

    # for i in range(len(questions)):
    #     print(f"Q: {questions[i]}")
    #     for j, context in enumerate(contexts_list[i]):
    #         print(f"Context {j+1}: {context}")
    #     print(f"GT: {ground_truths[i]}\n")


    # Loop through each model
    for model in models:
        generator = create_generator(model.get("generator"), model.get("provider"), model.get("name"))

        output_file = f"../output/noise_robustness/{output_file_base}_noise_{noise_ratio}_{model.get('name').replace('/', '_')}.json"

        print(f"Evaluating model: {model.get('name')} with noise ratio: {noise_ratio}")

        judge = LLMJudge()
        results = judge.bulk_evaluation(
            questions=questions,
            contexts_list=contexts_list,
            ground_truths=ground_truths,
            question_ids=question_ids,
            generator=generator,
            output_file=output_file
        )
