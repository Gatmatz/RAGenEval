from .Instructor import Instructor

class SystemInstructor(Instructor):
    def get_instructions(self):
        return ("You are a helpful assistant who answers questions based solely on the provided context. "
                "Based on the following context, answer the question. "
                "Instructions: "
                "Read the context carefully."
                "Answer the user's question using only information found within the context."
                "If the answer is not in the context respond with 'I don't know'."
                "Do not use prior knowledge."
                "Be concise and accurate.")

    def get_counterfactual_instructions(self):
        return ("You are a helpful assistant who answers questions. "
                "Based on the following context, answer the question. "
                "If you detect any factual errors in the provided context, respond ONLY with 'There are factual errors in the provided context'. Do not provide explanations, corrections, or additional information."
                "Instructions: "
                "Read the context carefully."
                "Be concise and accurate.")
