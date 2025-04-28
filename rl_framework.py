import random
import math
from collections import defaultdict
from base_classes import *

#blackjack agent that uses Q-learning to learn the optimal policy.
class BlackjackAgent:
    def __init__(self, epsilon_strategy='constant', initial_epsilon=0.1):
        self.q_values = defaultdict(lambda: {'hit': 0.0, 'stand': 0.0})  # Q(s, a)
        self.visit_counts = defaultdict(lambda: {'hit': 0, 'stand': 0})  # N(s, a)
        self.epsilon_strategy = epsilon_strategy
        self.initial_epsilon = initial_epsilon
        self.episode_count = 0

    #define the state key as a tuple (player_sum, dealer_card, usable_ace)
    def get_state_key(self, state):
        return state

    #define the epsilon value based on the strategy chosen
    def get_epsilon(self):
        k = self.episode_count
        if self.epsilon_strategy == 'constant':
            return self.initial_epsilon
        elif self.epsilon_strategy == '1/k':
            return 1 / (k + 1)
        elif self.epsilon_strategy == 'exp1':
            return math.exp(-1 * k / 10000)
        elif self.epsilon_strategy == 'exp2':
            return math.exp(-1 * k / 100000)
        else:
            return self.initial_epsilon

    #define the action to take based on the epsilon-greedy policy
    def choose_action(self, state, exploring_start=False):
        state_key = self.get_state_key(state)
        epsilon = self.get_epsilon()

        if exploring_start:
            return random.choice(['hit', 'stand'])

        if random.random() < epsilon:
            return random.choice(['hit', 'stand'])
        else:
            return max(self.q_values[state_key], key=self.q_values[state_key].get)

    #define the function to generate an episode
    def generate_episode(self, env, exploring_start=False):
        self.episode_count += 1
        episode = []

        state = env.reset()
        done = False

        if state[0] < 12:
            #always hit
            action = 'hit'
        elif state[0] == 21:
            #always stand
            action = 'stand'
        else:
            action = self.choose_action(state, exploring_start=exploring_start)

        while not done:
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))

            if done:
                break

            state = next_state

            if state[0] < 12:
                action = 'hit'
            elif state[0] == 21:
                action = 'stand'
            else:
                action = self.choose_action(state)

        return episode

    #update Q-values based on episode generated
    def update_q_values(self, episode):
        visited = set()
        for state, action, reward in episode:
            state_key = self.get_state_key(state)
            if (state_key, action) not in visited:
                self.visit_counts[state_key][action] += 1
                alpha = 1 / self.visit_counts[state_key][action]
                self.q_values[state_key][action] += alpha * (reward - self.q_values[state_key][action])
                visited.add((state_key, action))


#testing
env = BlackJackEnv()
agent = BlackjackAgent(epsilon_strategy='1/k')

#generate and update from 10 episodes
for _ in range(10):
    episode = agent.generate_episode(env)
    agent.update_q_values(episode)

#print some Q-values
for state, actions in list(agent.q_values.items())[:5]:
    print(f"State: {state} | Q-values: {actions}")
