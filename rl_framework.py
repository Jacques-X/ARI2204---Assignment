import random
import math
from collections import defaultdict
from base_classes import *

# blackjack agent that uses various RL algorithms to learn the optimal policy.
class BlackjackAgent:
    def __init__(self):
        self.q_values = defaultdict(lambda: {'hit': 0.0, 'stand': 0.0})  # Q(s, a)
        self.q1_values = defaultdict(lambda: {'hit': 0.0, 'stand': 0.0}) # For Double Q-Learning
        self.q2_values = defaultdict(lambda: {'hit': 0.0, 'stand': 0.0}) # For Double Q-Learning
        self.visit_counts = defaultdict(lambda: {'hit': 0, 'stand': 0})  # N(s, a)
        self.episode_count = 0

    # define the state key as a tuple (player_sum, dealer_card, usable_ace)
    def get_state_key(self, state):
        player_sum, dealer_card, usable_ace = state
        # Only learn for player sums between 12 and 20
        if 12 <= player_sum <= 20:
            return state
        else:
            return None # Indicate that this state is not learned on

    # define the epsilon value based on the strategy chosen
    def get_epsilon(self, epsilon_strategy):
        k = self.episode_count
        if epsilon_strategy == 'constant':
            return 0.1
        elif epsilon_strategy == '1/k':
            return 1 / (k + 1)
        elif epsilon_strategy == 'exp1':
            return math.exp(-k / 1000)
        elif epsilon_strategy == 'exp2':
            return math.exp(-k / 10000)
        else:
            return 0.1 # Default to constant epsilon

    # define the action to take based on the epsilon-greedy policy
    def choose_action(self, state, epsilon_strategy, exploring_start=False):
        state_key = self.get_state_key(state)
        if state_key is None: # Always hit below 12, stand at 21
            if state[0] < 12:
                return 'hit'
            else: # state[0] == 21
                return 'stand'

        epsilon = self.get_epsilon(epsilon_strategy)

        # Exploring starts for the first action of an episode in Monte Carlo ES
        if exploring_start and self.visit_counts[state_key]['hit'] == 0 and self.visit_counts[state_key]['stand'] == 0:
             return random.choice(['hit', 'stand'])

        if random.random() < epsilon:
            return random.choice(['hit', 'stand'])
        else:
            # Choose greedily based on current Q-values
            if self.q_values[state_key]['hit'] > self.q_values[state_key]['stand']:
                return 'hit'
            elif self.q_values[state_key]['stand'] > self.q_values[state_key]['hit']:
                return 'stand'
            else:
                return random.choice(['hit', 'stand']) # Break ties randomly

    # Define the action to take for Double Q-Learning based on the average Q-values
    def choose_action_double_q(self, state, epsilon_strategy):
        state_key = self.get_state_key(state)
        if state_key is None: # Always hit below 12, stand at 21
            if state[0] < 12:
                return 'hit'
            else: # state[0] == 21
                return 'stand'

        epsilon = self.get_epsilon(epsilon_strategy)

        if random.random() < epsilon:
            return random.choice(['hit', 'stand'])
        else:
            # Choose greedily based on average Q-values
            avg_q_hit = (self.q1_values[state_key]['hit'] + self.q2_values[state_key]['hit']) / 2
            avg_q_stand = (self.q1_values[state_key]['stand'] + self.q2_values[state_key]['stand']) / 2

            if avg_q_hit > avg_q_stand:
                return 'hit'
            elif avg_q_stand > avg_q_hit:
                return 'stand'
            else:
                return random.choice(['hit', 'stand']) # Break ties randomly


    # 2.3 Monte Carlo On-Policy Control
    # This method is designed to be called to run a single episode and return it for external processing
    # The update logic for Monte Carlo is handled in the evaluation script after the episode
    def generate_episode_monte_carlo(self, env, epsilon_strategy, exploring_start=False):
        episode = []
        state = env.reset()
        done = False

        # Handle exploring starts for the first action if applicable
        if exploring_start and self.get_state_key(state) is not None:
             action = random.choice(['hit', 'stand'])
        else:
            action = self.choose_action(state, epsilon_strategy, exploring_start=exploring_start)

        while not done:
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))

            if not done:
                state = next_state
                # For Monte Carlo ES, subsequent actions in the same episode use the epsilon-greedy policy
                action = self.choose_action(state, epsilon_strategy, exploring_start=False)


        return episode

    # 2.4 SARSA On-Policy Control
    # This method runs a single step and performs updates step by step
    def step_sarsa(self, env, state, action, epsilon_strategy):
        next_state, reward, done = env.step(action)
        next_state_key = self.get_state_key(next_state)

        # Choose next action A' based on the next state S' (epsilon-greedy)
        if next_state_key is None:
            next_action = self.choose_action(next_state, epsilon_strategy)
        else:
            next_action = self.choose_action(next_state, epsilon_strategy)

        # Get state_key for the current state BEFORE it's updated to next_state
        state_key = self.get_state_key(state)

        # Update Q-value for the current state-action pair only if it's a learning state
        if state_key is not None:
            self.visit_counts[state_key][action] += 1
            alpha = 1 / (self.visit_counts[state_key][action] + 1) # Step size as 1/N(s,a)+1

            if done:
                # Terminal state
                self.q_values[state_key][action] += alpha * (reward - self.q_values[state_key][action])
            else:
                # Non-terminal state
                self.q_values[state_key][action] += alpha * (reward + self.q_values[next_state_key][next_action] - self.q_values[state_key][action])

        return next_state, next_action, reward, done

    # 2.5 Q-Learning (SARSAMAX) Off-Policy Control
    # This method runs a single step and performs the Q-learning update
    def step_q_learning(self, env, state, action, epsilon_strategy):
        next_state, reward, done = env.step(action)
        next_state_key = self.get_state_key(next_state)

        # Get state_key for the current state BEFORE it's updated to next_state
        state_key = self.get_state_key(state)


        # Update Q-value for the current state-action pair only if it's a learning state
        if state_key is not None:
            self.visit_counts[state_key][action] += 1
            alpha = 1 / (self.visit_counts[state_key][action] + 1) # Step size as 1/N(s,a)+1

            if done:
                 # Terminal state
                self.q_values[state_key][action] += alpha * (reward - self.q_values[state_key][action])
            else:
                # Non-terminal state: Update using max Q-value of next state
                max_q_next = max(self.q_values[next_state_key].values()) if next_state_key is not None else 0
                self.q_values[state_key][action] += alpha * (reward + max_q_next - self.q_values[state_key][action])

        # Choose the next action from the next state using epsilon-greedy policy (behavior policy)
        next_action = self.choose_action(next_state, epsilon_strategy)

        return next_state, next_action, reward, done


    # 2.6 Double Q-Learning Off-Policy Control
    # This method runs a single step and performs the Double Q-learning update
    def step_double_q_learning(self, env, state, action, epsilon_strategy):
        next_state, reward, done = env.step(action)
        next_state_key = self.get_state_key(next_state)

        # Get state_key for the current state BEFORE it's updated to next_state
        state_key = self.get_state_key(state)

        # Update Q1 or Q2 only if the current state is a learning state
        if state_key is not None:
             # Randomly choose to update Q1 or Q2
            if random.random() < 0.5:
                # Update Q1
                self.visit_counts[state_key][action] += 1 # Still count visits for step size
                alpha = 1 / (self.visit_counts[state_key][action] + 1)
                if next_state_key is not None:
                     max_q2_next_action = max(self.q2_values[next_state_key], key=self.q2_values[next_state_key].get)
                     self.q1_values[state_key][action] += alpha * (reward + self.q2_values[next_state_key][max_q2_next_action] - self.q1_values[state_key][action])
                else:
                     self.q1_values[state_key][action] += alpha * (reward - self.q1_values[state_key][action])

            else:
                # Update Q2
                self.visit_counts[state_key][action] += 1 # Still count visits for step size
                alpha = 1 / (self.visit_counts[state_key][action] + 1)
                if next_state_key is not None:
                    max_q1_next_action = max(self.q1_values[next_state_key], key=self.q1_values[next_state_key].get)
                    self.q2_values[state_key][action] += alpha * (reward + self.q1_values[next_state_key][max_q1_next_action] - self.q2_values[state_key][action])
                else:
                    self.q2_values[state_key][action] += alpha * (reward - self.q2_values[state_key][action])

        # Choose the next action from the next state using epsilon-greedy policy based on average Q-values (behavior policy)
        next_action = self.choose_action_double_q(next_state, epsilon_strategy)


        return next_state, next_action, reward, done


    # Calculate final Q-values as the average of Q1 and Q2 after training
    def calculate_double_q_average(self):
        for state_key in self.q1_values:
            for action in self.q1_values[state_key]:
                 self.q_values[state_key][action] = (self.q1_values[state_key][action] + self.q2_values[state_key][action]) / 2