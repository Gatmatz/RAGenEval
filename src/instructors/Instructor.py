from abc import ABC, abstractmethod

class Instructor(ABC):
    """
    Abstract base class for all instructors.
    Defines the interface that all instructor subclasses must implement.
    """
    @abstractmethod
    def get_instructions(self) -> str:
        pass