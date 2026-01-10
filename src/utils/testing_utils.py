"""
Test script for different generators
"""

from src.generator import OpenaiCompGenerator, GemmaGenerator
def test_generator(generator_class, provider_name, model_name, question, contexts):
    """Test a generator with sample data"""
    try:
        if provider_name is not None:
            generator = generator_class(provider_name=provider_name, model_name=model_name)
        else:
            generator = generator_class(model_name=model_name)
        #generator = generator_class(provider_name=provider_name, model_name=model_name)
        answer = generator.generate(question, contexts)
        print(f"\n{generator_class.__name__} ({model_name}):")
        print(f"Answer: {answer}")
        return True
    except Exception as e:
        print(f"\n{generator_class.__name__} ({model_name}): Error - {e}")
        return False

def main():
    # Sample question and contexts
    question = "What is the capital of France?"
    contexts = [
        "Paris is the capital and most populous city of France.",
        "France is a country in Western Europe.",
        "The population of Paris is approximately 2.2 million people."
    ]

    print("Testing Generators with Different Models")
    print("=" * 60)


    # Test GPT-OOS models via OpenAICompatibleGenerator
    test_generator(OpenaiCompGenerator, "openrouter", "openai/gpt-oss-20b:free", question, contexts)
    test_generator(OpenaiCompGenerator, "openrouter", "openai/gpt-oss-120b:free", question, contexts)

    print("=" * 60)

    # Test Qwen models via OpenAICompatibleGenerator
    test_generator(OpenaiCompGenerator, "openrouter", "qwen/qwen3-4b:free", question, contexts)
    test_generator(OpenaiCompGenerator, "cerebras", "qwen-3-32b", question, contexts)
    
    print("=" * 60)

    # Test Google Gemma models via GemmaGenerator
    test_generator(GemmaGenerator,None, "gemma-3-4b-it", question, contexts)
    test_generator(GemmaGenerator,None, "gemma-3-27b-it", question, contexts)


if __name__ == "__main__":
    main()
