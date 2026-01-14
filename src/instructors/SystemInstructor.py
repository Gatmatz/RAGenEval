from abc import ABC

from typing_extensions import override

from .Instructor import Instructor


class SystemInstructor(Instructor, ABC):
    @override
    def get_general_instructions(self):
        return ("You are a helpful assistant who answers questions based solely on the provided context. "
                "Read the context carefully."
                "Answer the user's question using only information found within the context."
                "Do not include any thinking process or reasoning tags in your response. Provide only the final answer."
                "If the answer is not in the context respond with 'I don't know'."
                "Do not use prior knowledge."
                "Be concise and accurate.")

    def get_counterfactual_instructions(self):
        return ("You are a helpful assistant who answers questions."
                "If you detect any factual errors in the provided context, respond ONLY with 'There are factual errors in the provided context'. Do not provide explanations, corrections, or additional information."
                "Do not include any thinking process or reasoning tags in your response."
                "Read the context carefully."
                "Be concise and accurate.")