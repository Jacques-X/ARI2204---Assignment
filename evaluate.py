# evaluation.py

import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
import random

# Import the BlackjackAgent from the implementation file
from implementation import BlackjackAgent

# Import the Blackjack Environment class from base_classes.py (assumed to exist)
# This class contains the Card, Deck, Player, and Dealer definitions.
try:
    from black_jack_env import BlackJackEnv
except ImportError:
    print("Error: Could not import BlackJackEnv from base_classes.py.")
    print("Please ensure base_classes.py is in the same directory and contains the BlackJackEnv class.")
    # Exit or use a dummy environment if necessary for structural checks
    class BlackJackEnv:
         def __init__(self): print("Dummy BlackJackEnv initialized. Simulation will not run.")
         def reset(self): print("Dummy reset."); return (12, 2, False)
         def step(self, action): print(f"Dummy step with {action}."); return (12, 2, False), 0, True


# Helper function to create safe filenames for saving plots/results
def safe_file_name(name):
    """Replaces characters that could cause issues in file paths."""
    # Added more replacements for common problematic characters
    return name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_').replace(' ', '_').replace('=', '_')


# --- Simulation Runner Function (Based on parts of original evaluate.py and PDF Point 3) ---

def run_simulation(agent, env, algorithm, num_episodes, epsilon_strategy, exploring_starts=False):
    """
    Runs a simulation of the specified RL algorithm in the Blackjack environment
    for a given number of episodes and collects evaluation data.

    This function is responsible for the interaction loop between the agent and the environment,
    episode management, action selection during the simulation, and collecting the metrics
    required for evaluation (wins/losses/draws, state-action counts, final Q-values).
    It calls the algorithm-specific update methods implemented in the Agent class.

    Aligns with PDF Point 3, specifically "Run each algorithm configuration..." and
    extracting information (items 1-4).

    Args:
        agent: The BlackjackAgent instance from implementation.py.
        env: The BlackJackEnv instance from base_classes.py.
        algorithm (str): The name of the algorithm ('monte_carlo', 'sarsa', 'q_learning', 'double_q_learning').
        num_episodes (int): The total number of episodes to run.
        epsilon_strategy (str): The epsilon decay strategy ('constant', '1/k', 'exp1', 'exp2').
        exploring_starts (bool): Whether to use exploring starts (primarily for Monte Carlo).

    Returns:
        tuple: A tuple containing:
            - win_counts_1k (list): List of wins per 1000 episodes.
            - loss_counts_1k (list): List of losses per 1000 episodes.
            - draw_counts_1k (list): List of draws per 1000 episodes.
            - unique_state_actions (set): Set of unique (state_key, action) pairs encountered over all episodes.
            - state_action_counts (defaultdict): Counts of each (state_key, action) pair over all episodes.
            - q_values_at_end (dict): Final Q-values for all encountered learning state-action pairs after all episodes.
                                      (For Double Q-Learning, this contains the average Q-values).
    """
    # Lists to store win, loss, and draw counts to be reported every 1000 episodes [cite: 59]
    win_counts_1k = []
    loss_counts_1k = []
    draw_counts_1k = []

    # Sets and dictionaries to track unique state-actions and their counts over the whole simulation [cite: 60, 61]
    unique_state_actions = set()
    state_action_counts = defaultdict(int)

    # Dictionary to store final Q-values - will be populated after the simulation loop [cite: 62]
    q_values_at_end = {}

    # Counters for wins, losses, and draws within the current 1000-episode block
    current_1k_wins = 0
    current_1k_losses = 0
    current_1k_draws = 0

    # Main simulation loop over episodes
    for i in range(1, num_episodes + 1):
        agent.episode_count = i # Update episode count for epsilon decay calculation

        # Reset environment for a new episode (gets a new shuffled deck) [cite: 35]
        state = env.reset()
        done = False
        episode_data_mc = [] # To store (state, action, reward) tuples for Monte Carlo updates (after episode)

        # Determine the first action of the episode based on algorithm and exploring_starts
        # Exploring starts applies only to the very first action of an episode in Monte Carlo ES [cite: 42]
        # and only if the initial state is a learning state (player sum 12-20).
        initial_state_key = agent.get_state_key(state)
        if algorithm == 'monte_carlo' and exploring_starts and initial_state_key is not None:
             action = random.choice(['hit', 'stand']) # Random action for exploring starts [cite: 42]
        else:
            # For all other cases (MC without ES, SARSA, Q-Learning, Double Q-Learning),
            # choose the first action based on the epsilon-greedy policy.
            if algorithm == 'double_q_learning':
                action = agent.choose_action_double_q(state, epsilon_strategy) # Use average Q for Double Q-Learning behavior policy
            else:
                action = agent.choose_action(state, epsilon_strategy, exploring_start=False) # Use standard epsilon-greedy

        # --- Episode Loop ---
        while not done:
            # Get the state key for the current state. This is used for tracking and updates for states 12-20.
            state_key = agent.get_state_key(state)

            # Record the state-action pair encountered during the episode, but only for learning states (12-20) [cite: 60, 61]
            if state_key is not None:
                 unique_state_actions.add((state_key, action)) # Track unique state-action pairs [cite: 60]
                 state_action_counts[(state_key, action)] += 1 # Count occurrences of each state-action pair [cite: 61]

            # Take the action in the environment and observe the next state and reward
            next_state, reward, done = env.step(action)

            # Perform algorithm-specific updates and determine the next action for the *next* step
            if algorithm == 'monte_carlo':
                # For MC, we store episode data (state, action, reward) and perform updates *after* the episode ends [cite: 39]
                episode_data_mc.append((state, action, reward))
                state = next_state # Update state for the next iteration in the loop
                if not done:
                    # The next action is chosen by the epsilon-greedy policy for subsequent steps in MC [cite: 43, 44]
                    action = agent.choose_action(state, epsilon_strategy, exploring_start=False) # Exploring starts only for the first action

            elif algorithm == 'sarsa':
                # For SARSA, we need the next action A' from the next state S' *before* performing the update
                if not done:
                    # Choose the next action A' from the next state S' using the epsilon-greedy policy (On-Policy)
                    next_action = agent.choose_action(next_state, epsilon_strategy)
                else:
                    next_action = None # No action from a terminal state

                # Perform the SARSA update using (S, A, R, S', A')
                agent.sarsa_update(state, action, reward, next_state, next_action)
                state = next_state # Update state for the next iteration
                action = next_action # Update action for the next iteration


            elif algorithm == 'q_learning':
                # For Q-Learning, we perform the update using the max Q of the next state (Off-Policy Target)
                agent.q_learning_update(state, action, reward, next_state)
                state = next_state # Update state for the next iteration
                if not done:
                    # Choose the next action using the epsilon-greedy policy (Behavior Policy)
                    action = agent.choose_action(state, epsilon_strategy)
                else:
                    action = None # No action from a terminal state


            elif algorithm == 'double_q_learning':
                 # For Double Q-Learning, we perform the update to either Q1 or Q2 randomly
                 agent.double_q_learning_update(state, action, reward, next_state)
                 state = next_state # Update state for the next iteration
                 if not done:
                     # Choose the next action using the epsilon-greedy policy based on average Q-values (Behavior Policy)
                     action = agent.choose_action_double_q(state, epsilon_strategy)
                 else:
                     action = None # No action from a terminal state


        # --- End of Episode ---

        # After the episode is done, perform Monte Carlo update if applicable [cite: 39, 40]
        if algorithm == 'monte_carlo':
            agent.mc_update(episode_data_mc)
            # The final reward for MC can be considered the reward from the last step in the episode data [cite: 38]
            final_episode_reward = episode_data_mc[-1][2] if episode_data_mc else 0

        elif algorithm in ['sarsa', 'q_learning', 'double_q_learning']:
            # For TD methods, the reward received when transitioning to the terminal state is the final reward [cite: 38]
             final_episode_reward = reward


        # Update win, loss, and draw counts for the current 1000-episode block [cite: 59]
        if final_episode_reward == 1:
            current_1k_wins += 1
        elif final_episode_reward == -1:
            current_1k_losses += 1
        # Draws have a reward of 0 [cite: 38]
        else: # final_episode_reward == 0
            current_1k_draws += 1


        # Record win/loss/draw counts every 1000 episodes [cite: 59]
        if i % 1000 == 0:
            win_counts_1k.append(current_1k_wins)
            loss_counts_1k.append(current_1k_losses)
            draw_counts_1k.append(current_1k_draws)
            # Reset counters for the next 1000 episodes
            current_1k_wins = 0
            current_1k_losses = 0
            current_1k_draws = 0


    # After all episodes, prepare final Q-values for reporting [cite: 62]
    # For Double Q-Learning, calculate the average Q-values into agent.q_values first [cite: 54]
    if algorithm == 'double_q_learning':
        agent.calculate_double_q_average()

    # Copy the final Q-values from the agent for return
    for state_key in agent.q_values:
         if state_key not in q_values_at_end:
             q_values_at_end[state_key] = {}
         for action in agent.q_values[state_key]:
             q_values_at_end[state_key][action] = agent.q_values[state_key][action]


    # Return the collected data for evaluation and reporting [cite: 59, 60, 61, 62]
    return win_counts_1k, loss_counts_1k, draw_counts_1k, unique_state_actions, state_action_counts, q_values_at_end

# --- Plotting Functions (Aligning with PDF Point 3, items 5, 6, 7) ---

def plot_win_loss_draw(win_counts, loss_counts, draw_counts, algorithm, config_name):
    """
    Plots wins, losses, and draws per 1000 episodes over the course of training.
    Aligns with PDF Point 3, item 5.
    """
    episodes = np.arange(1000, (len(win_counts) + 1) * 1000, 1000)
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, win_counts, label='Wins')
    plt.plot(episodes, loss_counts, label='Losses')
    plt.plot(episodes, draw_counts, label='Draws')
    plt.xlabel('Episodes')
    plt.ylabel('Count per 1000 Episodes')
    plt.title(f'{algorithm} - {config_name}: Wins, Losses, and Draws over Episodes')
    plt.legend()
    plt.grid(True)

    # Create directory if it doesn't exist and save the plot
    save_dir = f'{algorithm}_{safe_file_name(config_name)}'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'win_loss_draw.png'))
    plt.close()

def plot_state_action_counts(state_action_counts, algorithm, config_name):
    """
    Plots the counts of each unique state-action pair encountered during training.
    Aligns with PDF Point 3, item 6.
    """
    # Sort by count in descending order, then by state key and action for ties
    # Converting state_key tuple to string for reliable sorting as secondary key
    sorted_counts = sorted(state_action_counts.items(), key=lambda item: (-item[1], str(item[0]), item[1]))
    states_actions = [f"{sa[0]}-{sa[1]}" for sa, count in sorted_counts] # Format state-action for label
    counts = [count for sa, count in sorted_counts]

    plt.figure(figsize=(15, 7))
    # Limit to top N for readability, as requested implicitly by needing a plottable bar chart
    num_to_plot = min(len(states_actions), 50) # Plot at most 50 for readability
    plt.bar(states_actions[:num_to_plot], counts[:num_to_plot])
    plt.xlabel('State-Action Pair (PlayerSum, DealerCard, UsableAce - Action)')
    plt.ylabel('Count')
    plt.title(f'{algorithm} - {config_name}: Top {num_to_plot} State-Action Pair Counts') # Updated title to reflect top N
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Create directory if it doesn't exist and save the plot
    save_dir = f'{algorithm}_{safe_file_name(config_name)}'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'state_action_counts.png'))
    plt.close()

def plot_unique_state_actions(unique_counts_per_config, algorithm):
    """
    Plots the total number of unique state-action pairs explored across configurations for an algorithm.
    Aligns with PDF Point 3, item 7.
    """
    # Sort configurations for consistent plotting
    sorted_configs = sorted(unique_counts_per_config.keys())
    configs = sorted_configs
    counts = [unique_counts_per_config[config] for config in sorted_configs]

    plt.figure(figsize=(10, 6))
    plt.bar(configs, counts)
    plt.xlabel('Configuration')
    plt.ylabel('Number of Unique State-Action Pairs Explored')
    plt.title(f'{algorithm}: Total Unique State-Action Pairs Explored by Configuration')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Create directory if it doesn't exist and save the plot
    save_dir = f'{algorithm}_unique_counts'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'unique_state_actions.png'))
    plt.close()


# --- Strategy Table Functions (Aligning with PDF Point 3, item 8) ---

def create_strategy_table(q_values, usable_ace):
    """
    Creates a strategy table (player sum vs. dealer card) based on learned Q-values.
    Indicates 'H' for Hit and 'S' for Stand.
    Aligns with PDF Point 3, item 8.
    """
    # Dealer card values: 2-11 (A=11)
    # Player sum values: 12-20 (as per learning range)
    player_sums = list(range(12, 21))
    dealer_cards_map = {i: i for i in range(2, 11)}
    dealer_cards_map[11] = 'A' # Map 11 to 'A' for table header
    dealer_card_values = list(range(2, 11)) + [11] # Dealer card values to consider

    strategy_table = {}

    for player_sum in player_sums:
        strategy_table[player_sum] = {}
        for dealer_card_value in dealer_card_values:
            state = (player_sum, dealer_card_value, usable_ace)
            # The state key for Q-values is the same as the state tuple for player sums 12-20
            state_key = state

            action = '-' # Default: indicates state not encountered or no clear preferred action
            # Determine the best action ('H' or 'S') based on the final Q-values
            if state_key in q_values:
                # Get Q-values for hit and stand. Use a very low number if an action wasn't encountered
                # to ensure the other action is preferred in comparisons.
                q_hit = q_values[state_key].get('hit', -float('inf'))
                q_stand = q_values[state_key].get('stand', -float('inf'))

                if q_hit > q_stand:
                    action = 'H' # Hit has a higher Q-value
                elif q_stand > q_hit:
                    action = 'S' # Stand has a higher Q-value
                else:
                    action = 'S' # Break ties by standing (common practice in Blackjack basic strategy) [cite: 67]

            # Store the determined action in the strategy table
            strategy_table[player_sum][dealer_cards_map[dealer_card_value]] = action

    return strategy_table

def print_strategy_table(strategy_table, usable_ace, algorithm, config_name):
    """Prints the strategy table in a formatted way."""
    print(f"\n--- {algorithm} - {safe_file_name(config_name)} Strategy Table (Usable Ace: {usable_ace}) ---")
    # Create the header row for the table (Player Sum and Dealer Card values)
    dealer_cards_header = ["Player Sum"] + list(range(2, 11)) + ['A']
    print(" | ".join(map(str, dealer_cards_header)))
    # Print a separator line for readability
    print("-" * (sum(len(str(h)) + 3 for h in dealer_cards_header) - 3)) # Dynamically adjust line length

    # Sort player sums in descending order for the rows of the table
    player_sums_rows = sorted(strategy_table.keys(), reverse=True)

    # Print each row of the strategy table
    for player_sum in player_sums_rows:
        row = [player_sum] # Start the row with the player's sum
        for dealer_card_header in list(range(2, 11)) + ['A']:
             # Get the action for the current player sum and dealer card.
             # Use .get with a default '-' in case a state-dealer card combination was not in the Q-values.
             row.append(strategy_table[player_sum].get(dealer_card_header, '-'))
        print(" | ".join(map(str, row)))

# --- Dealer Advantage Calculation (Aligning with PDF Point 3, item 8) ---

def calculate_dealer_advantage(win_counts_1k, loss_counts_1k):
    """
    Calculates the dealer advantage using the formula (l - w) / (l + w)
    based on the mean wins and losses over the last 10,000 episodes.
    Aligns with PDF Point 3, item 8. [cite: 68]
    """
    # win_counts_1k and loss_counts_1k contain counts per 1000 episodes.
    # We need the mean over the last 10,000 episodes, which is the sum over the last 10 intervals of 1000 episodes.

    if len(win_counts_1k) < 10 or len(loss_counts_1k) < 10:
        # If less than 10,000 episodes of data are available (less than 10 intervals),
        # calculate the advantage using all available data.
        print(f"Warning: Less than 10,000 episodes ({len(win_counts_1k)*1000} recorded) of data available for dealer advantage calculation. Using all available data.")
        last_10k_wins = sum(win_counts_1k)
        last_10k_losses = sum(loss_counts_1k)
    else:
        # Sum the counts from the last 10 elements (representing the last 10,000 episodes)
        last_10k_wins = sum(win_counts_1k[-10:])
        last_10k_losses = sum(loss_counts_1k[-10:])

    total_hands = last_10k_wins + last_10k_losses
    if total_hands == 0:
        return 0.0 # Avoid division by zero if no relevant hands occurred in the last 10k episodes
    else:
        # Calculate the dealer advantage using the formula (l - w) / (l + w) [cite: 68]
        return (last_10k_losses - last_10k_wins) / total_hands


# --- Main Evaluation Execution Block (Orchestrates simulations and reporting) ---

if __name__ == "__main__":
    num_episodes = 100000 # Total number of episodes to run for each configuration [cite: 46, 47, 49, 50, 55, 58]

    # Define epsilon strategies for each algorithm as per the brief
    # Monte Carlo: 1/k (ES), 1/k (No ES), exp1, exp2
    # SARSA, Q-Learning, Double Q-Learning: constant (0.1), 1/k, exp1, exp2
    mc_epsilon_strategies = ['1/k', 'exp1', 'exp2']
    td_epsilon_strategies = ['constant', '1/k', 'exp1', 'exp2']

    algorithms = ['monte_carlo', 'sarsa', 'q_learning', 'double_q_learning'] # Algorithms to evaluate [cite: 57]

    # Dictionaries to store results for final comparison plots and reporting
    dealer_advantages = defaultdict(dict) # To store dealer advantage for each config [cite: 68]
    all_unique_state_action_counts = defaultdict(dict) # To store unique state-action counts for each config [cite: 60]

    # Iterate through each algorithm to be evaluated
    for algorithm in algorithms:
        unique_counts_for_algorithm = {} # Temporary storage for unique counts per config within this algorithm (for plotting item 7) [cite: 65]
        configs_to_run = [] # List to hold configuration dictionaries for the current algorithm

        # Define the specific configurations to run for each algorithm based on the brief
        if algorithm == 'monte_carlo':
            # Monte Carlo configurations:
            # 1. Exploring Starts with epsilon=1/k [cite: 42]
            configs_to_run.append({'epsilon': '1/k', 'exploring_starts': True})
            # 2. No Exploring Starts with epsilon=1/k, exp1, exp2 [cite: 44]
            for strat in mc_epsilon_strategies:
                 configs_to_run.append({'epsilon': strat, 'exploring_starts': False})

            # Ensure unique configurations in case of overlap (e.g., '1/k' No ES might be added twice without care)
            seen_configs = set()
            unique_configs_to_run = []
            for config in configs_to_run:
                 config_tuple = (config['epsilon'], config['exploring_starts'])
                 if config_tuple not in seen_configs:
                     unique_configs_to_run.append(config)
                     seen_configs.add(config_tuple)
            configs_to_run = unique_configs_to_run


        else: # SARSA, Q-Learning, Double Q-Learning configurations
            # TD methods are always No Exploring Starts in this project
            for strat in td_epsilon_strategies:
                configs_to_run.append({'epsilon': strat, 'exploring_starts': False})


        # Run simulations for each configuration of the current algorithm
        for config in configs_to_run:
            epsilon_strategy = config['epsilon']
            exploring_starts = config['exploring_starts']

            # Generate a descriptive config name for plotting and saving files
            if algorithm == 'monte_carlo':
                if exploring_starts:
                    config_name = f'{epsilon_strategy}_es' # Example: 1/k_es
                else:
                    # Name for MC No ES cases: 1/k_no_es, exp1, exp2
                    config_name = f'{epsilon_strategy}_no_es' if epsilon_strategy == '1/k' else epsilon_strategy
            else: # TD methods
                config_name = epsilon_strategy # Example: constant, 1/k, exp1, exp2

            # Print the current simulation details to the console for tracking progress
            print(f"Running {algorithm} - {config_name} with {num_episodes} episodes...")

            # Initialize a new agent and environment for each run to ensure a clean start
            # The BlackjackEnv is imported from base_classes.py
            env = BlackJackEnv()
            agent = BlackjackAgent() # The BlackjackAgent is imported from implementation.py

            # Run the simulation and collect the results
            win_counts_1k, loss_counts_1k, draw_counts_1k, unique_state_actions, state_action_counts, q_values = run_simulation(
                 agent, env, algorithm, num_episodes, epsilon_strategy, exploring_starts=exploring_starts
            )

            # --- Perform Evaluation and Reporting for this Configuration (Point 3) ---

            # Plot wins, losses, and draws over episodes [cite: 63]
            plot_win_loss_draw(win_counts_1k, loss_counts_1k, draw_counts_1k, algorithm, config_name)

            # Plot state-action counts [cite: 64]
            plot_state_action_counts(state_action_counts, algorithm, config_name)

            # Record unique state-action counts for later plotting across configurations [cite: 65]
            unique_counts_for_algorithm[config_name] = len(unique_state_actions)
            all_unique_state_action_counts[algorithm][config_name] = len(unique_state_actions)

            # Calculate and record dealer advantage for later plotting [cite: 68]
            dealer_advantage = calculate_dealer_advantage(win_counts_1k, loss_counts_1k)
            dealer_advantages[algorithm][config_name] = dealer_advantage

            # Create and print strategy tables based on final Q-values [cite: 67]
            strategy_table_no_ace = create_strategy_table(q_values, usable_ace=False)
            strategy_table_with_ace = create_strategy_table(q_values, usable_ace=True)
            print_strategy_table(strategy_table_no_ace, usable_ace=False, algorithm=algorithm, config_name=config_name)
            print_strategy_table(strategy_table_with_ace, usable_ace=True, algorithm=algorithm, config_name=config_name)


        # After running all configurations for an algorithm, plot unique state-action pairs explored [cite: 65]
        plot_unique_state_actions(unique_counts_for_algorithm, algorithm)


    # --- Final Comparison Plots and Reporting (Aggregating results across algorithms/configs) ---

    # Plot dealer advantage for all algorithm configurations on one chart [cite: 68]
    plt.figure(figsize=(15, 7))
    # Collect all config names and advantages for the plot in a consistent order
    all_configs = []
    all_advantages = []
    # Ensure a consistent order for plotting - sort by algorithm then config name
    sorted_algorithms = sorted(dealer_advantages.keys())
    for algorithm in sorted_algorithms:
        # Sort configurations within each algorithm for consistent plotting
        sorted_configs = sorted(dealer_advantages[algorithm].keys())
        for config_name in sorted_configs:
            all_configs.append(f"{algorithm}-{config_name}")
            all_advantages.append(dealer_advantages[algorithm][config_name])

    # Create the bar chart
    plt.bar(all_configs, all_advantages)

    # Add labels and title
    plt.xlabel('Algorithm and Configuration')
    plt.ylabel('Dealer Advantage (Mean over last 10,000 episodes)')
    plt.title('Dealer Advantage Comparison Across Algorithms and Configurations')
    plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for readability
    plt.grid(axis='y', linestyle='--') # Add a dashed grid on the y-axis
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.savefig('dealer_advantage_comparison.png') # Save the plot
    plt.close() # Close the plot figure

    # Print final summary results to the console
    print("\n\nEvaluation complete. Plots and strategy tables generated in respective directories.")
    print("\nSummary of Results:")

    print("\nDealer Advantages (Mean over last 10,000 episodes):")
    # Print dealer advantages sorted by algorithm and then config for readability
    for algorithm in sorted_algorithms:
        print(f"- {algorithm}:")
        sorted_configs = sorted(dealer_advantages[algorithm].keys())
        for config_name in sorted_configs:
            # Format the dealer advantage to a few decimal places for readability
            print(f"  - {config_name}: {dealer_advantages[algorithm][config_name]:.4f}")


    print("\nTotal Unique State-Action Pairs Explored:")
    # Print unique counts sorted by algorithm and then config for readability
    for algorithm in sorted(all_unique_state_action_counts.keys()):
         print(f"- {algorithm}:")
         sorted_configs = sorted(all_unique_state_action_counts[algorithm].keys())
         for config_name in sorted_configs:
             print(f"  - {config_name}: {all_unique_state_action_counts[algorithm][config_name]}")