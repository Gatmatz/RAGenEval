from typing import List

from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class GroqGenerator:
    """Generator using Groq API"""

    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.model_name = model_name
        self.client = Groq(api_key=GROQ_API_KEY)

    def generate(self, question: str, contexts: List[str]) -> str:
        """Generate answer based on question and contexts"""
        context_text = "\n\n".join(contexts)
        prompt = f"""Based on the following context, answer the question. 
        If the answer is not contained within the context, respond with "I cannot find the answer in the provided context." and nothing more.

Context:
{context_text}

Question: {question}

Answer:"""

        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_name,
            temperature=0.7,
            max_tokens=256,
        )

        return chat_completion.choices[0].message.content