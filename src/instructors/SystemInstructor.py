from .Instructor import Instructor

class SystemInstructor(Instructor):
    def get_instructions(self):
        return ("You are a helpful assistant who answers questions based solely on the provided context. "
                "Based on the following context, answer the question. "
                "Instructions: "
                "Read the context carefully."
                "Answer the user's question using only information found within the context."
                "If the answer is not in the context respond with 'I don't know'."
                "When the provided context contain factual errors, respond with 'There are factual errors in the provided context'.'"
                "Do not use prior knowledge."
                "Be concise and accurate.")
