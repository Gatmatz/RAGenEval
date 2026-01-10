import re

def remove_thinking(answer: str) -> str:
    """
    Removes any text between <think> tags (inclusive) from the provided answer string.

    Args:
        answer (str): The original answer string that may contain <think> tags.

    Returns:
        str: The cleaned answer string without <think> tags and their content.
    """
    return re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()