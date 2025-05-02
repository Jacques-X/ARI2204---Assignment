from implementation import BlackjackAgent
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import random
import os

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

# --- Simulation Runner Function ---
def run_simulation(agent, env, algorithm, num_episodes, epsilon_strategy, exploring_starts=False):
    # Lists to store win, loss, and draw counts to be reported every 1000 episodes
    win_counts_1k = []
    loss_counts_1k = []
    draw_counts_1k = []

    # Sets and dictionaries to track unique state-actions and their counts over the whole simulation
    unique_state_actions = set()
    state_action_counts = defaultdict(int)

    # Dictionary to store final Q-values - will be populated after the simulation loop
    q_values_at_end = {}

    # Counters for wins, losses, and draws within the current 1000-episode block
    current_1k_wins = 0
    current_1k_losses = 0
    current_1k_draws = 0

    # Main simulation loop over episodes
    for i in range(1, num_episodes + 1):
        agent.episode_count = i # Update episode count for epsilon decay calculation

        # Reset environment for a new episode (gets a new shuffled deck)
        state = env.reset()
        done = False
        episode_data_mc = [] # To store (state, action, reward) tuples for Monte Carlo updates (after episode)

        initial_state_key = agent.get_state_key(state)
        if algorithm == 'monte_carlo' and exploring_starts and initial_state_key is not None:
             action = random.choice(['hit', 'stand']) # Random action for exploring starts
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

            # Record the state-action pair encountered during the episode, but only for learning states
            if state_key is not None:
                 unique_state_actions.add((state_key, action)) # Track unique state-action pairs
                 state_action_counts[(state_key, action)] += 1 # Count occurrences of each state-action pair

            # Take the action in the environment and observe the next state and reward
            next_state, reward, done = env.step(action)

            # Perform algorithm-specific updates and determine the next action for the *next* step
            if algorithm == 'monte_carlo':
                # For MC, we store episode data (state, action, reward) and perform updates *after* the episode ends
                episode_data_mc.append((state, action, reward))
                state = next_state # Update state for the next iteration in the loop
                if not done:
                    # The next action is chosen by the epsilon-greedy policy for subsequent steps in MC
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
                 agent.double_q_learning_update(state, action, reward, next_state)
                 state = next_state # Update state for the next iteration
                 if not done:
                     action = agent.choose_action_double_q(state, epsilon_strategy)
                 else:
                     action = None # No action from a terminal state


        # --- End of Episode ---

        # After the episode is done, perform Monte Carlo update if applicable
        if algorithm == 'monte_carlo':
            agent.mc_update(episode_data_mc)
            # The final reward for MC can be considered the reward from the last step in the episode data
            final_episode_reward = episode_data_mc[-1][2] if episode_data_mc else 0

        elif algorithm in ['sarsa', 'q_learning', 'double_q_learning']:
            # For TD methods, the reward received when transitioning to the terminal state is the final reward
             final_episode_reward = reward


        # Update win, loss, and draw counts for the current 1000-episode block
        if final_episode_reward == 1:
            current_1k_wins += 1
        elif final_episode_reward == -1:
            current_1k_losses += 1
        # Draws have a reward of 0
        else: # final_episode_reward == 0
            current_1k_draws += 1


        # Record win/loss/draw counts every 1000 episodes
        if i % 1000 == 0:
            win_counts_1k.append(current_1k_wins)
            loss_counts_1k.append(current_1k_losses)
            draw_counts_1k.append(current_1k_draws)
            # Reset counters for the next 1000 episodes
            current_1k_wins = 0
            current_1k_losses = 0
            current_1k_draws = 0


    # After all episodes, prepare final Q-values for reporting
    # For Double Q-Learning, calculate the average Q-values into agent.q_values first
    if algorithm == 'double_q_learning':
        agent.calculate_double_q_average()

    # Copy the final Q-values from the agent for return
    for state_key in agent.q_values:
         if state_key not in q_values_at_end:
             q_values_at_end[state_key] = {}
         for action in agent.q_values[state_key]:
             q_values_at_end[state_key][action] = agent.q_values[state_key][action]


    # Return the collected data for evaluation and reporting
    return win_counts_1k, loss_counts_1k, draw_counts_1k, unique_state_actions, state_action_counts, q_values_at_end

# --- Plotting Functions ---

def plot_win_loss_draw(win_counts, loss_counts, draw_counts, algorithm, config_name, main_folder):
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
    save_dir = os.path.join(main_folder, 'individual_plots', f'{algorithm}_{safe_file_name(config_name)}')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'win_loss_draw.png'))
    plt.close()

def plot_state_action_counts(state_action_counts, algorithm, config_name, main_folder):
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
    save_dir = os.path.join(main_folder, 'individual_plots', f'{algorithm}_{safe_file_name(config_name)}')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'state_action_counts.png'))
    plt.close()

def plot_unique_state_actions(unique_counts_per_config, algorithm, main_folder):
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
    save_dir = os.path.join(main_folder, 'comparison_plots')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{algorithm}_unique_state_actions.png'))
    plt.close()


# --- Strategy Table Functions---

def create_strategy_table(q_values, usable_ace):
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
                    action = 'S' # Break ties by standing (common practice in Blackjack basic strategy)

            # Store the determined action in the strategy table
            strategy_table[player_sum][dealer_cards_map[dealer_card_value]] = action

    return strategy_table

def print_strategy_table(strategy_table, usable_ace, algorithm, config_name):
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

# --- Dealer Advantage Calculation  ---

def calculate_dealer_advantage(win_counts_1k, loss_counts_1k):
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
        # Calculate the dealer advantage using the formula (l - w) / (l + w)
        return (last_10k_losses - last_10k_wins) / total_hands


# --- Main Evaluation Execution Block  ---

if __name__ == "__main__":
    num_episodes = 100000 # Total number of episodes to run for each configuration
    main_plots_folder = "blackjack_plots" # Define the main folder name

    # Create the main plots folder if it doesn't exist
    os.makedirs(main_plots_folder, exist_ok=True)

    # Define epsilon strategies for each algorithm as per the brief
    # Monte Carlo: 1/k (ES), 1/k (No ES), exp1, exp2
    # SARSA, Q-Learning, Double Q-Learning: constant (0.1), 1/k, exp1, exp2
    mc_epsilon_strategies = ['1/k', 'exp1', 'exp2']
    td_epsilon_strategies = ['constant', '1/k', 'exp1', 'exp2']

    algorithms = ['monte_carlo', 'sarsa', 'q_learning', 'double_q_learning'] # Algorithms to evaluate

    # Dictionaries to store results for final comparison plots and reporting
    dealer_advantages = defaultdict(dict) # To store dealer advantage for each config
    all_unique_state_action_counts = defaultdict(dict) # To store unique state-action counts for each config

    # Iterate through each algorithm to be evaluated
    for algorithm in algorithms:
        unique_counts_for_algorithm = {} # Temporary storage for unique counts per config within this algorithm (for plotting item 7)
        configs_to_run = [] # List to hold configuration dictionaries for the current algorithm

        # Define the specific configurations to run for each algorithm based on the brief
        if algorithm == 'monte_carlo':
            # Monte Carlo configurations:
            # 1. Exploring Starts with epsilon=1/k
            configs_to_run.append({'epsilon': '1/k', 'exploring_starts': True})
            # 2. No Exploring Starts with epsilon=1/k, exp1, exp2
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

            # Plot wins, losses, and draws over episodes
            plot_win_loss_draw(win_counts_1k, loss_counts_1k, draw_counts_1k, algorithm, config_name, main_plots_folder)

            # Plot state-action counts
            plot_state_action_counts(state_action_counts, algorithm, config_name, main_plots_folder)

            # Record unique state-action counts for later plotting across configurations
            unique_counts_for_algorithm[config_name] = len(unique_state_actions)
            all_unique_state_action_counts[algorithm][config_name] = len(unique_state_actions)

            # Calculate and record dealer advantage for later plotting
            dealer_advantage = calculate_dealer_advantage(win_counts_1k, loss_counts_1k)
            dealer_advantages[algorithm][config_name] = dealer_advantage

            # Create and print strategy tables based on final Q-values
            strategy_table_no_ace = create_strategy_table(q_values, usable_ace=False)
            strategy_table_with_ace = create_strategy_table(q_values, usable_ace=True)
            print_strategy_table(strategy_table_no_ace, usable_ace=False, algorithm=algorithm, config_name=config_name)
            print_strategy_table(strategy_table_with_ace, usable_ace=True, algorithm=algorithm, config_name=config_name)


        # After running all configurations for an algorithm, plot unique state-action pairs explored
        plot_unique_state_actions(unique_counts_for_algorithm, algorithm, main_plots_folder)


    # --- Final Comparison Plots and Reporting ---

    # Plot dealer advantage for all algorithm configurations on one chart
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

    # Save the comparison plot in the comparison_plots subfolder within the main folder
    comparison_plot_path = os.path.join(main_plots_folder, 'comparison_plots', 'dealer_advantage_comparison.png')
    os.makedirs(os.path.dirname(comparison_plot_path), exist_ok=True)
    plt.savefig(comparison_plot_path)
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