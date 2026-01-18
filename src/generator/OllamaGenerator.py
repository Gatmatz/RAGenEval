import ollama
from typing import List
from dotenv import load_dotenv
from src.instructors.SystemInstructor import SystemInstructor

load_dotenv()


class OllamaGenerator:
    """Generator using local Ollama models"""

    def __init__(self, model_name: str = "qwen3:4b"):
        """
        Initialize Ollama generator

        Args:
            model_name: Name of the model (default: qwen3:4b)
        """
        self.model_name = model_name

    def generate(self, question: str, contexts: List[str]) -> str:
        """Generate answer based on question and contexts"""
        context_text = "\n\n".join(contexts)
        system_instruction = SystemInstructor()

        prompt = f"""
        Context: {context_text}
        Question: {question}
        Answer:"""

        # Using the native ollama.chat method
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_instruction.get_counterfactual_instructions()
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0.5,
                "num_predict": -1,
            }
        )

        content = response['message']['content']
        return content