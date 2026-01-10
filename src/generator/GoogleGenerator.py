from typing import List
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

from src.instructors.SystemInstructor import SystemInstructor

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class GemmaGenerator:
    """Generator using official Google Gemini API"""

    def __init__(self, model_name: str):
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model_name = model_name

    def generate(self, question: str, contexts: List[str]) -> str:
        """Generate answer based on question and contexts"""
        context_text = "\n\n".join(contexts)

        system_instruction = SystemInstructor()
        user_prompt = f"{system_instruction.get_instructions()}\n\nContext: {context_text}\n\nQuestion: {question}"

        config = types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=256,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=config,
        )

        return response.text
