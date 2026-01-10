from .Instructor import Instructor

class SystemInstructor(Instructor):
    def get_instructions(self):
        return ("You are a helpful assistant designed to provide accurate and concise information. "
                "Based on the following context, answer the question. "
                "If the context does not contain the answer, respond with 'I don't know'.")