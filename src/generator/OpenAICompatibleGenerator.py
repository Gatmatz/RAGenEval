from typing import List
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

class OpenaiCompGenerator:
    """Generator using OpenOpenRouter, Groq, and Cerebras API"""
    
    def __init__(self, provider_name: str, model_name: str):
        endpoints = {
            "openrouter": "https://openrouter.ai/api/v1",
            "groq": "https://api.groq.com/openai/v1",
            "cerebras": "https://api.cerebras.ai/v1"
        }
        self.model_name = model_name
        API_KEY = os.getenv(f"{provider_name.upper()}_API_KEY")
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=endpoints.get(provider_name.lower())
        )

    def generate(self, question: str, contexts: List[str]) -> str:
        """Generate answer based on question and contexts"""
        context_text = "\n\n".join(contexts)
        prompt = f"""
        Context: {context_text}
        Question: {question}
        Answer:"""

        chat_completion = self.client.chat.completions.create(
            messages=[
                    {"role": "system", "content": "Based on the following context, answer the question."},
                    {"role": "user", "content": prompt}
            ],
            model=self.model_name,
            temperature=0.7,
            max_tokens=256,
        )

        return chat_completion.choices[0].message.content