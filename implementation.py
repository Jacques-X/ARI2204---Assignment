# implementation.py

import random
import math
from collections import defaultdict

# Note: This file contains the RL Agent implementation (Point 2.2 onwards).
# The Blackjack Environment classes (Card, Deck, Player, Dealer, BlackJackEnv)
# which are part of Point 2.1, are assumed to be in a separate file (e.g., base_classes.py)
# and are not included here based on the user's specific splitting instruction.


# --- Class for Reinforcement Learning Agent (Based on original rl_framework.py and PDF Point 2.2 - 2.6) ---

class BlackjackAgent:
    def __init__(self):
        # Dictionaries to store Q-values and visit counts
        # Using defaultdict with a lambda for easier access to new state-action pairs
        self.q_values = defaultdict(lambda: {'hit': 0.0, 'stand': 0.0})  # Q(s, a) estimates
        self.q1_values = defaultdict(lambda: {'hit': 0.0, 'stand': 0.0}) # For Double Q-Learning
        self.q2_values = defaultdict(lambda: {'hit': 0.0, 'stand': 0.0}) # For Double Q-Learning
        self.visit_counts = defaultdict(lambda: {'hit': 0, 'stand': 0})  # N(s, a) counts
        self.episode_count = 0 # To track episode number for epsilon decay

    def get_state_key(self, state):
        """
        Defines the state representation used by the agent for learning.
        Filters states to the learning range [12..20] as per PDF Point 2.2 and 2.2.1.
        """
        player_sum, dealer_card, usable_ace = state
        # The agent only learns for player sums between 12 and 20
        if 12 <= player_sum <= 20:
            return state # Use the full state tuple as the key for these states
        else:
            return None # Indicate that this state is outside the learning range


    def get_epsilon(self, epsilon_strategy):
        """
        Calculates the epsilon value based on the chosen strategy and episode count.
        Aligns with the epsilon decay strategies mentioned in PDF Points 2.3, 2.4, 2.5, 2.6.
        """
        k = self.episode_count
        if epsilon_strategy == 'constant':
            return 0.1 # Constant epsilon = 0.1
        elif epsilon_strategy == '1/k':
            # Epsilon = 1/k. Handle k=0 to avoid division by zero, although episodes start from 1.
            return 1 / k if k > 0 else 1.0 #
        elif epsilon_strategy == 'exp1':
            return math.exp(-k / 1000) # Epsilon = exp(-k/1000)
        elif epsilon_strategy == 'exp2':
            return math.exp(-k / 10000) # Epsilon = exp(-k/10000)
        else:
            return 0.1 # Default to constant epsilon if strategy is unrecognized

    def choose_action(self, state, epsilon_strategy, exploring_start=False):
        """
        Chooses an action ('hit' or 'stand') based on the current state and epsilon-greedy policy.
        Handles fixed actions for sums < 12 or == 21.
        Supports Exploring Starts for the very first action of an episode (for Monte Carlo ES).
        Aligns with PDF Point 2.2, 2.3, 2.4, 2.5.
        """
        player_sum, _, _ = state
        state_key = self.get_state_key(state)

        # Fixed policy for sums outside the learning range
        if player_sum < 12:
            return 'hit' # Always HIT when sum is less than 12
        elif player_sum >= 21: # Player sum is 21 or busted (though busted is terminal)
            return 'stand' # Always STAND when sum is 21

        # If it's a learning state [12..20], apply epsilon-greedy or exploring starts
        epsilon = self.get_epsilon(epsilon_strategy)

        # Exploring Starts: first action is random if applicable (for Monte Carlo ES)
        if exploring_start and state_key is not None: # Apply only for learning states
             return random.choice(['hit', 'stand'])

        # Epsilon-greedy policy for learning states
        if random.random() < epsilon:
            return random.choice(['hit', 'stand']) # Choose randomly with probability epsilon
        else:
            # Choose greedily based on current Q-values
            # Need to handle cases where a state_key or action hasn't been encountered yet
            # Use 0.0 as a default value for unseen state-action pairs
            q_hit = self.q_values.get(state_key, {}).get('hit', 0.0)
            q_stand = self.q_values.get(state_key, {}).get('stand', 0.0)

            if q_hit > q_stand:
                return 'hit'
            elif q_stand > q_hit:
                return 'stand'
            else:
                return random.choice(['hit', 'stand']) # Break ties randomly

    # Define the action to take for Double Q-Learning based on the average Q-values (behavior policy)
    # This is the behavior policy for Double Q-Learning
    def choose_action_double_q(self, state, epsilon_strategy):
        """
        Chooses an action for Double Q-Learning based on the average of Q1 and Q2.
        This is the behavior policy (epsilon-greedy) for Double Q-Learning.
        Aligns with PDF Point 2.6.
        """
        player_sum, _, _ = state
        state_key = self.get_state_key(state)

        # Fixed policy for sums outside the learning range
        if player_sum < 12:
            return 'hit'
        elif player_sum >= 21:
            return 'stand'

        # If it's a learning state [12..20], apply epsilon-greedy based on average Q
        epsilon = self.get_epsilon(epsilon_strategy)

        if random.random() < epsilon:
            return random.choice(['hit', 'stand']) # Choose randomly with probability epsilon
        else:
            # Choose greedily based on average Q-values (Q1 + Q2) / 2
            q1_hit = self.q1_values.get(state_key, {}).get('hit', 0.0)
            q1_stand = self.q1_values.get(state_key, {}).get('stand', 0.0)
            q2_hit = self.q2_values.get(state_key, {}).get('hit', 0.0)
            q2_stand = self.q2_values.get(state_key, {}).get('stand', 0.0)

            avg_q_hit = (q1_hit + q2_hit) / 2
            avg_q_stand = (q1_stand + q2_stand) / 2

            if avg_q_hit > avg_q_stand:
                return 'hit'
            elif avg_q_stand > avg_q_hit:
                return 'stand'
            else:
                return random.choice(['hit', 'stand']) # Break ties randomly


    # Helper to ensure state_key and action exist in Q dictionaries for updates
    # This prevents KeyError when accessing Q values for states/actions not yet encountered.
    def _ensure_q_entry(self, q_dict, state_key, action):
        if state_key not in q_dict:
            q_dict[state_key] = {'hit': 0.0, 'stand': 0.0}
        if action not in q_dict[state_key]:
             q_dict[state_key][action] = 0.0

    # --- Algorithm Update Methods (Aligning with PDF Point 2.3, 2.4, 2.5, 2.6) ---

    # Monte Carlo On-Policy Control Update (First-Visit) - Performed *after* the episode
    # Expects a list of (state, action, reward) tuples for the entire episode.
    # Updates Q-values for the *first* occurrence of each state-action pair in the episode.
    # Aligns with PDF Point 2.3.
    def mc_update(self, episode_data):
        """
        Performs the First-Visit Monte Carlo update for a single episode.
        Updates Q-values for the first occurrence of each (state, action) pair in the episode.
        Uses alpha = 1 / N(s,a).
        Aligns with PDF Point 2.3.
        """
        # Track visited state-action pairs within this episode for first-visit
        visited_states_actions = set()
        # Calculate the return G from the end of the episode (gamma=1)
        G = 0
        # Iterate backwards through the episode data to calculate returns
        for t in range(len(episode_data) - 1, -1, -1):
            state_t, action_t, reward_t = episode_data[t]
            G += reward_t # Return G_t is the sum of rewards from step t onwards (gamma=1 as per PDF Point 2.2.3)

            state_key_t = self.get_state_key(state_t)

            # Only perform update if it's a learning state [12..20]
            if state_key_t is not None:
                # Check if this is the first visit to (state_t, action_t) in this episode
                if (state_key_t, action_t) not in visited_states_actions:
                    # Ensure the state-action entry exists before updating
                    self._ensure_q_entry(self.q_values, state_key_t, action_t)
                    self._ensure_q_entry(self.visit_counts, state_key_t, action_t)

                    # Increment visit count for (S_t, A_t)
                    self.visit_counts[state_key_t][action_t] += 1
                    # Calculate step size alpha = 1 / N(S_t, A_t) for standard MC.
                    # Note: PDF specifies alpha=1/(N(s,a)+1) for TD methods, but doesn't explicitly state for MC.
                    # Using 1/N(s,a) is standard for MC First-Visit control.
                    alpha = 1 / self.visit_counts[state_key_t][action_t]

                    # Perform the Monte Carlo update: Q(S_t, A_t) <- Q(S_t, A_t) + alpha * (G_t - Q(S_t, A_t))
                    self.q_values[state_key_t][action_t] += alpha * (G - self.q_values[state_key_t][action_t])

                    # Mark this state-action pair as visited for first-visit MC in this episode
                    visited_states_actions.add((state_key_t, action_t))


    # SARSA On-Policy Control Update - Performed *at each step*
    # Expects current state (S), action (A), reward (R), next state (S'), and next action (A').
    # Aligns with PDF Point 2.4.
    def sarsa_update(self, state, action, reward, next_state, next_action):
        """
        Performs the SARSA update using (S, A, R, S', A').
        Uses alpha = 1 / (N(s,a) + 1) and gamma = 1.
        Aligns with PDF Point 2.4.
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        # Only update if the current state is a learning state [12..20]
        if state_key is not None:
            # Ensure the state-action entry exists before updating
            self._ensure_q_entry(self.q_values, state_key, action)
            self._ensure_q_entry(self.visit_counts, state_key, action)

            # Increment visit count for (S_t, A_t)
            self.visit_counts[state_key][action] += 1
            # Calculate step size alpha = 1 / (N(s,a) + 1)
            alpha = 1 / (self.visit_counts[state_key][action] + 1)

            # Get Q value for the next state-action pair (S', A'). If next state is terminal, Q(S', A') is 0.
            q_next = 0.0
            # next_action is None if the next_state was terminal in the environment step
            if next_state_key is not None and next_action is not None: # Not a terminal state in the learning range
                 # Ensure the next state-action entry exists
                 self._ensure_q_entry(self.q_values, next_state_key, next_action)
                 q_next = self.q_values[next_state_key][next_action]

            # SARSA Update Rule: Q(S, A) <- Q(S, A) + alpha * (R + gamma * Q(S', A') - Q(S, A))
            # With gamma = 1
            self.q_values[state_key][action] += alpha * (reward + q_next - self.q_values[state_key][action])


    # Q-Learning (SARSAMAX) Off-Policy Control Update - Performed *at each step*
    # Expects current state (S), action (A), reward (R), and next state (S').
    # Uses the max Q value of the next state for the target (Off-Policy).
    # Aligns with PDF Point 2.5.
    def q_learning_update(self, state, action, reward, next_state):
        """
        Performs the Q-Learning (SARSAMAX) update.
        Uses alpha = 1 / (N(s,a) + 1) and gamma = 1.
        Aligns with PDF Point 2.5.
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        # Only update if the current state is a learning state [12..20]
        if state_key is not None:
            # Ensure the state-action entry exists before updating
            self._ensure_q_entry(self.q_values, state_key, action)
            self._ensure_q_entry(self.visit_counts, state_key, action)

            # Increment visit count for (S_t, A_t)
            self.visit_counts[state_key][action] += 1
            # Calculate step size alpha = 1 / (N(s,a) + 1)
            alpha = 1 / (self.visit_counts[state_key][action] + 1)

            # Get max Q value for the next state S'. If next state is terminal, max Q is 0.
            max_q_next = 0.0
            if next_state_key is not None: # Not a terminal state in the learning range
                # Ensure the next state entries exist for both 'hit' and 'stand' to find the max
                self._ensure_q_entry(self.q_values, next_state_key, 'hit')
                self._ensure_q_entry(self.q_values, next_state_key, 'stand')
                # Find the maximum Q value for the next state S' across all actions 'a'
                max_q_next = max(self.q_values[next_state_key]['hit'], self.q_values[next_state_key]['stand'])

            # Q-Learning Update Rule: Q(S, A) <- Q(S, A) + alpha * (R + gamma * max_a Q(S', a) - Q(S, A))
            # With gamma = 1
            self.q_values[state_key][action] += alpha * (reward + max_q_next - self.q_values[state_key][action])

    # Double Q-Learning Off-Policy Control Update - Performed *at each step*
    # Expects current state (S), action (A), reward (R), and next state (S').
    # Updates either Q1 or Q2 randomly, using the other Q network for the target.
    # Aligns with PDF Point 2.6.
    def double_q_learning_update(self, state, action, reward, next_state):
        """
        Performs the Double Q-Learning update to either Q1 or Q2.
        Uses alpha = 1 / (N(s,a) + 1) and gamma = 1.
        Aligns with PDF Point 2.6.
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        # Only update if the current state is a learning state [12..20]
        if state_key is not None:
             # Randomly choose to update Q1 or Q2 with 0.5 probability
            if random.random() < 0.5:
                # Update Q1
                self._ensure_q_entry(self.q1_values, state_key, action)
                self._ensure_q_entry(self.visit_counts, state_key, action) # Visit counts for step size
                alpha = 1 / (self.visit_counts[state_key][action] + 1) # Step size

                target_q = 0.0
                if next_state_key is not None: # Not a terminal state in the learning range
                     # Find action that maximizes Q2 in the next state
                     self._ensure_q_entry(self.q2_values, next_state_key, 'hit')
                     self._ensure_q_entry(self.q2_values, next_state_key, 'stand')
                     # Choose the action that maximizes Q2
                     if self.q2_values[next_state_key]['hit'] > self.q2_values[next_state_key]['stand']:
                        max_q2_next_action = 'hit'
                     else:
                        max_q2_next_action = 'stand' # Break ties arbitrarily
                     # Target is reward + gamma * Q2(S', argmax_a Q2(S', a))
                     target_q = self.q2_values.get(next_state_key, {}).get(max_q2_next_action, 0.0)


                # Double Q-Learning Update Rule for Q1 (gamma = 1)
                self.q1_values[state_key][action] += alpha * (reward + target_q - self.q1_values[state_key][action])

            else:
                # Update Q2
                self._ensure_q_entry(self.q2_values, state_key, action)
                self._ensure_q_entry(self.visit_counts, state_key, action) # Visit counts for step size
                alpha = 1 / (self.visit_counts[state_key][action] + 1) # Step size

                target_q = 0.0
                if next_state_key is not None: # Not a terminal state in the learning range
                    # Find action that maximizes Q1 in the next state
                    self._ensure_q_entry(self.q1_values, next_state_key, 'hit')
                    self._ensure_q_entry(self.q1_values, next_state_key, 'stand')
                    # Choose the action that maximizes Q1
                    if self.q1_values[next_state_key]['hit'] > self.q1_values[next_state_key]['stand']:
                       max_q1_next_action = 'hit'
                    else:
                       max_q1_next_action = 'stand' # Break ties arbitrarily
                    # Target is reward + gamma * Q1(S', argmax_a Q1(S', a))
                    target_q = self.q1_values.get(next_state_key, {}).get(max_q1_next_action, 0.0)


                # Double Q-Learning Update Rule for Q2 (gamma = 1)
                self.q2_values[state_key][action] += alpha * (reward + target_q - self.q2_values[state_key][action])


    # Calculate final Q-values as the average of Q1 and Q2 after training
    # This is done after the simulation is complete for Double Q-Learning
    def calculate_double_q_average(self):
        """
        Calculates the average of Q1 and Q2 after training and stores it in self.q_values.
        Aligns with PDF Point 2.6.
        """
        # Iterate through all state_keys encountered in either Q1 or Q2
        all_state_keys = set(self.q1_values.keys()).union(set(self.q2_values.keys()))
        for state_key in all_state_keys:
             # Initialize the average Q for this state key if it doesn't exist
             if state_key not in self.q_values:
                 self.q_values[state_key] = {}
             for action in ['hit', 'stand']:
                  # Get Q1 and Q2 values, defaulting to 0.0 if the state-action was not seen in a specific Q table
                  q1 = self.q1_values.get(state_key, {}).get(action, 0.0)
                  q2 = self.q2_values.get(state_key, {}).get(action, 0.0)
                  # Calculate the average Q-value
                  self.q_values[state_key][action] = (q1 + q2) / 2