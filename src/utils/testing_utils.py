"""
Test script for different generators
"""

from src.generator import OpenAICompGenerator, GemmaGenerator, OllamaGenerator
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
    # question = "What is the capital of France?"
    # contexts = [
    #     "Paris is the capital and most populous city of France.",
    #     "France is a country in Western Europe.",
    #     "The population of Paris is approximately 2.2 million people."
    # ]
    question = "What is men's year-end No 1 in tennis in 2019?"
    contexts = ["Nov 14, 2019 ... Novak Djokovic has clinched the year-end No. 1 ATP Ranking for a fifth time, following today's results at the Nitto ATP Finals.", "Novak Djokovic finished the year as world No. 1 for the fifth time in his career. Details. Duration, 29 December 2018 – 24 November 2019. Edition, 50th.", "Nov 25, 2019 ... Novak Djokovic Ends the Season in a Familiar Place: On Top. Nadal won two majors and the Davis Cup and finished No. 1 in the rankings. But in 2019 ..."]
    print("Testing Generators with Different Models")
    print("=" * 60)


    # Test GPT-OOS models via OpenAICompatibleGenerator
    # test_generator(OpenAICompGenerator, "openrouter", "qwen/qwen3-4b:free", question, contexts)
    # test_generator(OpenAICompGenerator, "openrouter", "openai/gpt-oss-120b:free", question, contexts)
    #
    # print("=" * 60)
    #
    # # Test Qwen models via OpenAICompatibleGenerator
    # test_generator(OpenAICompGenerator, "openrouter", "qwen/qwen3-4b:free", question, contexts)
    # test_generator(OpenAICompGenerator, "cerebras", "qwen-3-32b", question, contexts)
    #
    # print("=" * 60)
    #
    # # Test Google Gemma models via GemmaGenerator
    # test_generator(GemmaGenerator,None, "gemma-3-4b-it", question, contexts)
    # test_generator(GemmaGenerator,None, "gemma-3-27b-it", question, contexts)
    test_generator(OllamaGenerator,None, "qwen3:0.6b", question, contexts)


if __name__ == "__main__":
    main()
