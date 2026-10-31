from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, problem, name: str = "BaseAgent"):
        self.problem = problem
        self.name = name

    @abstractmethod
    def select_action(self, state, metrics) -> tuple:
        """
        Determines the next action given the current state.
        :param state: Current environment state (room, completed_steps)
        :param metrics: MetricsTracker instance to log node expansions or costs
        :return: An action tuple, e.g., ('move', 'acad_1') or ('complete', 'acad_1')
        """
        pass

    def reset(self):
        """Resets any internal agent memory or state between simulation runs."""
        pass