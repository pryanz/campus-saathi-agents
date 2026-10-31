import random
from agents.base_agent import BaseAgent

class QLearningAgent(BaseAgent):
    def __init__(self, problem, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        super().__init__(problem, name="QLearningAgent")
        self.q_table = {}  
        self.alpha = alpha        #Learning rate
        self.gamma = gamma        # Discount factor
        self.epsilon = epsilon    #Exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def _get_state_key(self, state):

        current_room, completed_steps = state
        return (current_room, tuple(sorted(completed_steps)))

    def get_q(self, state_key, action):
        return self.q_table.get((state_key, action), 0.0)

    def select_action(self, state, metrics=None):
        #Selects an action using an epsilon-greedy policy
        state_key = self._get_state_key(state)
        actions = self.problem.actions(state)
        
        if not actions:
            return None

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(actions)

        q_values = [self.get_q(state_key, act) for act in actions]
        max_q = max(q_values)
        
        # Handle ties randomly
        best_actions = [act for act, q in zip(actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        #Updates Q-value using the standard Q-learning Bellman equation
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        current_q = self.get_q(state_key, action)
        
        if done:
            max_next_q = 0.0
        else:
            next_actions = self.problem.actions(next_state)
            if next_actions:
                max_next_q = max([self.get_q(next_state_key, act) for act in next_actions])
            else:
                max_next_q = 0.0

        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state_key, action)] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)