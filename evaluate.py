import matplotlib.pyplot as plt
import numpy as np
import random
import os # Import the os module
from collections import defaultdict
from rl_framework import BlackjackAgent
from base_classes import BlackJackEnv

def safe_file_name(name):
    """Replaces characters that could cause issues in file paths."""
    return name.replace('/', '_').replace('\\', '_')

def run_simulation(agent, env, algorithm, num_episodes, epsilon_strategy, exploring_starts=False):
    win_counts = []
    loss_counts = []
    draw_counts = []
    unique_state_actions = set()
    state_action_counts = defaultdict(int)
    q_values_at_end = {}

    wins = 0
    losses = 0
    draws = 0

    for i in range(1, num_episodes + 1):
        agent.episode_count = i # Update episode count for epsilon decay

        if algorithm == 'monte_carlo':
            episode = agent.generate_episode_monte_carlo(env, epsilon_strategy, exploring_start=exploring_starts)

            # Monte Carlo Update (First-Visit) and data collection per episode
            visited_states_actions = set()
            for state, action, reward in episode:
                 state_key = agent.get_state_key(state)
                 if state_key is not None: # Only process learning states
                     if (state_key, action) not in visited_states_actions:
                         agent.visit_counts[state_key][action] += 1
                         alpha = 1 / agent.visit_counts[state_key][action] # Step size for MC
                         agent.q_values[state_key][action] += alpha * (reward - agent.q_values[state_key][action])
                         visited_states_actions.add((state_key, action))

                     # Collect all state-action pairs encountered in the episode
                     unique_state_actions.add((state_key, action))
                     state_action_counts[(state_key, action)] += 1

            # Determine win/loss/draw from the final reward of the episode
            final_reward = episode[-1][2]
            if final_reward == 1:
                wins += 1
            elif final_reward == -1:
                losses += 1
            else:
                draws += 0 # Rewards for draw is 0, so it doesn't affect wins/losses count


        elif algorithm in ['sarsa', 'q_learning', 'double_q_learning']:
            state = env.reset()
            done = False

            # For TD methods, we iterate step by step
            while not done:
                # Choose action based on the current state and policy
                if algorithm == 'double_q_learning':
                    action = agent.choose_action_double_q(state, epsilon_strategy)
                else:
                    action = agent.choose_action(state, epsilon_strategy)

                # Get state_key and collect data *before* taking the step
                state_key = agent.get_state_key(state)
                if state_key is not None:
                    unique_state_actions.add((state_key, action))
                    state_action_counts[(state_key, action)] += 1


                # Take the step and perform the update
                if algorithm == 'sarsa':
                    state, action, reward, done = agent.step_sarsa(env, state, action, epsilon_strategy)
                elif algorithm == 'q_learning':
                    state, action, reward, done = agent.step_q_learning(env, state, action, epsilon_strategy)
                elif algorithm == 'double_q_learning':
                    state, action, reward, done = agent.step_double_q_learning(env, state, action, epsilon_strategy)

            # Determine win/loss/draw at the end of the episode based on the final reward
            if reward == 1:
                wins += 1
            elif reward == -1:
                losses += 1
            else:
                draws += 1 # Count draws for TD methods


        # Record win/loss/draw counts every 1000 episodes
        if i % 1000 == 0:
            win_counts.append(wins)
            loss_counts.append(losses)
            draw_counts.append(draws)
            wins = 0 # Reset counts for the next 1000 episodes
            losses = 0
            draws = 0


    # After all episodes, store final Q-values and calculate average for Double Q-Learning
    if algorithm == 'double_q_learning':
        agent.calculate_double_q_average() # Calculate the average Q-values into agent.q_values

    # Store the final Q-values from agent.q_values for all algorithms
    for state_key in agent.q_values:
         for action in agent.q_values[state_key]:
             q_values_at_end[(state_key, action)] = agent.q_values[state_key][action]


    return win_counts, loss_counts, draw_counts, unique_state_actions, state_action_counts, q_values_at_end


def plot_win_loss_draw(win_counts, loss_counts, draw_counts, algorithm, config_name):
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

    # Create directory if it doesn't exist
    save_dir = f'{algorithm}_{safe_file_name(config_name)}'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'win_loss_draw.png'))
    plt.close()

def plot_state_action_counts(state_action_counts, algorithm, config_name):
    # Sort by state key then action for consistent plotting
    sorted_counts = sorted(state_action_counts.items())
    states_actions = [f"{sa[0]}-{sa[1]}" for sa, count in sorted_counts] # Format state-action for label
    counts = [count for sa, count in sorted_counts]

    plt.figure(figsize=(15, 7))
    plt.bar(states_actions, counts)
    plt.xlabel('State-Action Pair (PlayerSum, DealerCard, UsableAce - Action)')
    plt.ylabel('Count')
    plt.title(f'{algorithm} - {config_name}: State-Action Pair Counts')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Create directory if it doesn't exist
    save_dir = f'{algorithm}_{safe_file_name(config_name)}'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'state_action_counts.png'))
    plt.close()

def plot_unique_state_actions(unique_counts_per_config, algorithm):
    configs = list(unique_counts_per_config.keys())
    counts = list(unique_counts_per_config.values())

    plt.figure(figsize=(10, 6))
    plt.bar(configs, counts)
    plt.xlabel('Configuration')
    plt.ylabel('Number of Unique State-Action Pairs Explored')
    plt.title(f'{algorithm}: Total Unique State-Action Pairs Explored by Configuration')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Create directory if it doesn't exist
    save_dir = f'{algorithm}_unique_counts'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'unique_state_actions.png'))
    plt.close()


def create_strategy_table(q_values, usable_ace):
    # Dealer card values: 2-11 (A=11)
    # Player sum values: 12-20
    player_sums = list(range(12, 21))
    dealer_cards_map = {i: i for i in range(2, 11)}
    dealer_cards_map[11] = 'A' # Map 11 to 'A' for table header
    dealer_card_values = list(range(2, 11)) + [11]


    strategy_table = {}

    for player_sum in player_sums:
        strategy_table[player_sum] = {}
        for dealer_card_value in dealer_card_values:
            state = (player_sum, dealer_card_value, usable_ace)
            state_key = state # Assuming get_state_key returns the state itself for learning states

            action = '-' # Default if state not encountered or learned on
            if state_key in q_values:
                # Ensure both 'hit' and 'stand' actions are present for the state_key
                if 'hit' in q_values[state_key] and 'stand' in q_values[state_key]:
                    if q_values[state_key]['hit'] > q_values[state_key]['stand']:
                        action = 'H'
                    elif q_values[state_key]['stand'] > q_values[state_key]['hit']:
                        action = 'S'
                    else:
                        action = 'S' # Break ties by standing (common practice)
                elif 'hit' in q_values[state_key]:
                     action = 'H' # Only hit was an option/encountered
                elif 'stand' in q_values[state_key]:
                    action = 'S' # Only stand was an option/encountered


            strategy_table[player_sum][dealer_cards_map[dealer_card_value]] = action


    return strategy_table

def print_strategy_table(strategy_table, usable_ace):
    print(f"\nStrategy Table (Usable Ace: {usable_ace})")
    dealer_cards_header = ["Player Sum"] + list(range(2, 11)) + ['A']
    print(" | ".join(map(str, dealer_cards_header)))
    print("-" * (len(" | ".join(map(str, dealer_cards_header)))))

    player_sums_rows = sorted(strategy_table.keys(), reverse=True)

    for player_sum in player_sums_rows:
        row = [player_sum]
        for dealer_card_header in list(range(2, 11)) + ['A']:
             row.append(strategy_table[player_sum].get(dealer_card_header, '-'))
        print(" | ".join(map(str, row)))

def calculate_dealer_advantage(win_counts, loss_counts):
    # Calculate mean over the last 10 intervals of 1000 episodes (last 10,000 episodes)
    if len(win_counts) < 10:
        print("Warning: Less than 10,000 episodes of data available for dealer advantage calculation.")
        last_10k_wins = sum(win_counts)
        last_10k_losses = sum(loss_counts)
    else:
        last_10k_wins = sum(win_counts[-10:])
        last_10k_losses = sum(loss_counts[-10:])


    if (last_10k_losses + last_10k_wins) == 0:
        return 0 # Avoid division by zero
    else:
        return (last_10k_losses - last_10k_wins) / (last_10k_losses + last_10k_wins)


# Main evaluation execution
if __name__ == "__main__":
    num_episodes = 100000
    # Adjust epsilon strategies based on the specific runs required per algorithm
    # Monte Carlo: 1/k (ES), 1/k (No ES), exp1, exp2
    # SARSA, Q-Learning, Double Q-Learning: constant (0.1), 1/k, exp1, exp2
    mc_epsilon_strategies = ['1/k', 'exp1', 'exp2'] # '1/k' will be run with and without ES
    td_epsilon_strategies = ['constant', '1/k', 'exp1', 'exp2']

    algorithms = ['monte_carlo', 'sarsa', 'q_learning', 'double_q_learning']

    dealer_advantages = defaultdict(dict)
    all_unique_state_action_counts = defaultdict(dict)

    for algorithm in algorithms:
        unique_counts_for_algorithm = {}

        # Handle Monte Carlo with Exploring Starts separately (only with 1/k epsilon)
        if algorithm == 'monte_carlo':
            epsilon_strategy = '1/k'
            config_name = f'{epsilon_strategy}_es'
            env = BlackJackEnv()
            agent = BlackjackAgent()
            print(f"Running Monte Carlo (Exploring Starts) with epsilon={epsilon_strategy}...")
            win_counts, loss_counts, draw_counts, unique_state_actions, state_action_counts, q_values = run_simulation(
                agent, env, algorithm, num_episodes, epsilon_strategy, exploring_starts=True
            )
            plot_win_loss_draw(win_counts, loss_counts, draw_counts, algorithm, config_name)
            plot_state_action_counts(state_action_counts, algorithm, config_name)
            unique_counts_for_algorithm[config_name] = len(unique_state_actions)
            all_unique_state_action_counts[algorithm][config_name] = len(unique_state_actions)
            dealer_advantage = calculate_dealer_advantage(win_counts, loss_counts)
            dealer_advantages[algorithm][config_name] = dealer_advantage
            strategy_table_no_ace = create_strategy_table(q_values, usable_ace=False)
            strategy_table_with_ace = create_strategy_table(q_values, usable_ace=True)
            print(f"\n--- {algorithm} - {config_name} Strategy Tables ---")
            print_strategy_table(strategy_table_no_ace, usable_ace=False)
            print_strategy_table(strategy_table_with_ace, usable_ace=True)

        # Handle the remaining configurations for each algorithm (all are No Exploring Starts)
        configs_to_run_no_es = mc_epsilon_strategies if algorithm == 'monte_carlo' else td_epsilon_strategies
        if algorithm == 'monte_carlo' and '1/k' in configs_to_run_no_es:
             # For MC, '1/k' in this loop means the No ES case.
             # We need to make sure to run exp1 and exp2 for MC No ES as well.
             pass # The loop below handles these

        for epsilon_strategy in configs_to_run_no_es:
             # Determine config name and exploring_starts flag
             if algorithm == 'monte_carlo':
                  # MC No ES cases
                  if epsilon_strategy == '1/k':
                     config_name = f'{epsilon_strategy}_no_es'
                  else: # exp1, exp2
                     config_name = epsilon_strategy
                  exploring_starts = False
             else:
                 # TD methods (SARSA, Q-Learning, Double Q-Learning) - always No ES
                 config_name = epsilon_strategy
                 exploring_starts = False

             # Skip the MC 1/k ES run here as it was handled before the loop
             if algorithm == 'monte_carlo' and epsilon_strategy == '1/k' and not exploring_starts:
                 # This is the MC 1/k No ES case - proceed
                 pass
             elif algorithm == 'monte_carlo' and epsilon_strategy == '1/k' and exploring_starts:
                 # This case was handled above the loop, skip it here
                 continue


             env = BlackJackEnv()
             agent = BlackjackAgent()
             print(f"Running {algorithm} - {config_name}...")
             win_counts, loss_counts, draw_counts, unique_state_actions, state_action_counts, q_values = run_simulation(
                  agent, env, algorithm, num_episodes, epsilon_strategy, exploring_starts=exploring_starts
             )
             plot_win_loss_draw(win_counts, loss_counts, draw_counts, algorithm, config_name)
             plot_state_action_counts(state_action_counts, algorithm, config_name)
             unique_counts_for_algorithm[config_name] = len(unique_state_actions)
             all_unique_state_action_counts[algorithm][config_name] = len(unique_state_actions)
             dealer_advantage = calculate_dealer_advantage(win_counts, loss_counts)
             dealer_advantages[algorithm][config_name] = dealer_advantage

             # Create and print strategy tables
             strategy_table_no_ace = create_strategy_table(q_values, usable_ace=False)
             strategy_table_with_ace = create_strategy_table(q_values, usable_ace=True)
             print(f"\n--- {algorithm} - {config_name} Strategy Tables ---")
             print_strategy_table(strategy_table_no_ace, usable_ace=False)
             print_strategy_table(strategy_table_with_ace, usable_ace=True)


        # Plot unique state-action pairs for the current algorithm across its configurations
        plot_unique_state_actions(unique_counts_for_algorithm, algorithm)


    # Plot dealer advantage for all algorithm configurations on one chart
    plt.figure(figsize=(15, 7))
    # Collect all config names and advantages
    all_configs = []
    all_advantages = []
    # Ensure a consistent order for plotting
    sorted_algorithms = sorted(dealer_advantages.keys())
    for algorithm in sorted_algorithms:
        # Sort configurations for consistent plotting
        sorted_configs = sorted(dealer_advantages[algorithm].keys())
        for config_name in sorted_configs:
            all_configs.append(f"{algorithm}-{config_name}")
            all_advantages.append(dealer_advantages[algorithm][config_name])


    plt.bar(all_configs, all_advantages)


    plt.xlabel('Algorithm and Configuration')
    plt.ylabel('Dealer Advantage (Mean over last 10,000 episodes)')
    plt.title('Dealer Advantage Comparison Across Algorithms and Configurations')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('dealer_advantage_comparison.png')
    plt.close()

    print("\nEvaluation complete. Plots and strategy tables generated.")
    print("Dealer Advantages:")
    for algorithm in dealer_advantages:
        print(f"- {algorithm}: {dealer_advantages[algorithm]}")

    print("\nTotal Unique State-Action Pairs Explored:")
    for algorithm in all_unique_state_action_counts:
        print(f"- {algorithm}: {all_unique_state_action_counts[algorithm]}")