import random
from collections import defaultdict
from base_classes import *

class BlackjackAgent:
    def __init__(self):
        self.q_values = defaultdict(lambda: {'hit': 0.0, 'stand': 0.0})  # Q(s, a)
        self.visit_counts = defaultdict(lambda: {'hit': 0, 'stand': 0})  # N(s, a)
        self.episode_count = 0

    #convert the state into a key for the Q-value dictionary.
    def get_state_key(self, state):
        return state

    #chooses an action based on the current state using an epsilon-greedy policy.
    def choose_action(self, state):
        player_sum = state[0]
        if player_sum < 12:
            return 'hit'
        elif player_sum == 21:
            return 'stand'
        else:
            #epsilon-greedy action selection (epsilon = 0, for now, to meet the 2.2 requirements)
            if random.random() < 0.0: 
                return random.choice(['hit', 'stand'])
            else:
                state_key = self.get_state_key(state)
                return max(self.q_values[state_key], key=self.q_values[state_key].get)

    #generate an episode using the environment.
    def generate_episode(self, env):
        self.episode_count += 1
        episode = []
        state = env.reset()
        done = False

        while not done:
            action = self.choose_action(state)
            next_state, reward, done = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        return episode

    #update Q-values based on the episode
    def update_q_values(self, episode):
        visited = set()
        for state, action, reward in episode:
            state_key = self.get_state_key(state)
            if (state_key, action) not in visited:
                self.visit_counts[state_key][action] += 1
                alpha = 1 / self.visit_counts[state_key][action]  # Learning rate
                self.q_values[state_key][action] += alpha * (reward - self.q_values[state_key][action])
                visited.add((state_key, action))



if __name__ == "__main__":
    env = BlackJackEnv()  #use the BlackJackEnv from base_classes.py
    agent = BlackjackAgent()

    #generate and update from a few episodes.
    for _ in range(1000):
        episode = agent.generate_episode(env)
        agent.update_q_values(episode)

    #print some Q-values to observe learning.
    for state, actions in list(agent.q_values.items())[:5]:
        print(f"State: {state} | Q-values: {actions}")
