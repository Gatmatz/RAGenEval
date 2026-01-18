from typing import List
import os
from openai import OpenAI
from dotenv import load_dotenv
from src.utils.llm_utils import remove_thinking

from src.instructors.SystemInstructor import SystemInstructor

load_dotenv()

class OpenAICompGenerator:
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
        system_instruction = SystemInstructor()

        prompt = f"""
        Context: {context_text}
        Question: {question}
        Answer:"""

        chat_completion = self.client.chat.completions.create(
            messages=[
                    {"role": "system", "content": system_instruction.get_general_instructions()},
                    {"role": "user", "content": prompt}
            ],
            model=self.model_name,
            temperature=0.5,
            max_tokens=512,
        )

        return remove_thinking(chat_completion.choices[0].message.content)