import seaborn as sns
from matplotlib import pyplot as plt

from gymconnectx.envs import ConnectGameEnv
from scenario.scenario_3x3 import Scenario_3x3


def env_3x3():
    env = ConnectGameEnv(
        connect=3,
        width=3,
        height=3,
        reward_winner=3,
        reward_loser=-3,
        reward_draw=1,
        reward_hell=-0.1,
        reward_hell_prob=-1.5,
        reward_win_prob=1.5,
        reward_living=-0.1,
        obs_number=True)
    return env


import numpy as np
import os
import random


class AgentQLearning:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0, role='player_1', q_table=None):
        self.q_tables = {'player_1': {}, 'player_2': {}}  # Separate Q-tables for each player role
        self.role = role  # Player role: 'player_1' or 'player_2'
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        if q_table is not None:
            self.q_tables[self.role] = q_table

    def get_state_key(self, state):
        # Convert numpy array to a tuple which is hashable and can be used as a dictionary key
        return tuple(state)

    def choose_action(self, state, possible_actions, debug=True):
        state_key = self.get_state_key(state)
        if random.random() < self.epsilon:
            chosen_action = random.choice(possible_actions)
            q_values = None if not debug else {action: self.q_tables[self.role].get(state_key, {}).get(action, 0) for
                                               action in possible_actions}
            return chosen_action, q_values
        else:
            q_table = self.q_tables[self.role]
            if state_key not in q_table:
                q_table[state_key] = {action: 0 for action in possible_actions}
            possible_q_values = {action: q_table[state_key].get(action, 0) for action in possible_actions}
            chosen_action = max(possible_q_values, key=possible_q_values.get)
            return chosen_action, possible_q_values if debug else chosen_action

    def update_q_value(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        q_table = self.q_tables[self.role]

        # Ensure state and action entries exist in q_table
        if state_key not in q_table:
            q_table[state_key] = {}
        if action not in q_table[state_key]:
            q_table[state_key][action] = 0

        # Check if next_state_key exists in q_table, if not initialize it
        if next_state_key not in q_table:
            q_table[next_state_key] = {a: 0 for a in range(
                len(next_state))}  # Assuming all possible actions from next_state are valid and initialized to 0

        # Calculate the maximum Q-value for the next state
        next_max = max(q_table[next_state_key].values(), default=0)

        # Update the Q-value for the current state and action
        q_table[state_key][action] += self.alpha * (reward + self.gamma * next_max - q_table[state_key][action])

    def save_q_table_to_npy(self, file_name):
        # Need to handle serialization for tuple keys and numpy array conversion
        data = {str(k): v for k, v in self.q_tables[self.role].items()}
        np.save(file_name, data)  # Save as .npy format

    def load_q_table_from_npy(self, file_name):
        if os.path.exists(file_name):
            data = np.load(file_name, allow_pickle=True).item()
            self.q_tables[self.role] = {eval(k): v for k, v in data.items()}  # Convert string keys back to tuples


# Example usage:
def train_agent_env(alpha=0.1, gamma=0.9, epsilon=0.1, no_episodes=1000):
    env = env_3x3()
    scenario = Scenario_3x3()
    valid_scenario_train = scenario.generate_permutations()

    agent1 = AgentQLearning(alpha, gamma, epsilon, role='player_1')
    agent2 = AgentQLearning(alpha, gamma, epsilon, role='player_2')

    for n in range(0, len(valid_scenario_train)):
        for game in range(no_episodes):
            env.reset()
            scenario_player_1_and_2 = scenario.generate_permutations()[n]
            last_state = ""
            last_action = -1
            last_player = ""
            while not env.is_done:
                try:
                    current_player = 'player_1' if env.get_current_player() == 1 else 'player_2'
                    agent = agent1 if current_player == 'player_1' else agent2
                    if env.current_step < 4:
                        state = env.get_obs()
                        action = scenario_player_1_and_2[env.current_step]  # based on scenario
                    else:
                        if current_player == 'player_1':
                            state = env.get_obs()
                            possible_action = env.get_moves()
                            action, possible_q_values = agent.choose_action(state, possible_action, debug=True)
                        else:
                            action = env.get_action_random()

                    next_state, rewards, done, _, info = env.step(action)
                    print(f'game: {game}, action: {action}, reward: {rewards}')

                    env.render(mode='terminal_display')

                    if current_player == 'player_1':
                        agent.update_q_value(state, action, rewards[current_player], next_state)

                    if done:
                        # update agent for last action if lost
                        if current_player == 'player_2':
                            agent = agent1
                            agent.update_q_value(last_state, last_action, rewards[last_player], next_state)
                        print("------------------------")
                        print(env.get_game_status())
                        print(f'game: {game}, action: {action}, reward: {rewards}')
                        break
                    else:
                        last_player = current_player
                        last_state = state
                        last_action = action
                        env.current_step += 1

                except Exception as e:
                    print(f"An error occurred: {str(e)}")
                    break

        # Save the Q-tables in .npy format
        agent1.save_q_table_to_npy(f"case_{n + 1}_player_1_connect3_3x3.npy")
        agent2.save_q_table_to_npy(f"case_{n + 1}_player_2_connect3_3x3.npy")


def play_with_q_table(scenario=None, file_name=""):
    env = env_3x3()
    agent = AgentQLearning(role='player_1')
    agent.load_q_table_from_npy(file_name)
    env.reset()

    while not env.is_done:
        current_player = env.get_current_player()
        if scenario is None:
            if current_player == 1:
                state = tuple(env.get_obs())
                possible_actions = env.get_moves()
                action, possible_q_values = agent.choose_action(state, possible_actions, debug=True)
                plot_possible_q_values(possible_q_values)
            else:
                action = env.set_players(player_2_mode='human_gui')
        else:
            if env.current_step < 4:
                action = scenario[env.current_step]
            else:
                if current_player == 1:
                    state = tuple(env.get_obs())
                    possible_actions = env.get_moves()
                    action, possible_q_values = agent.choose_action(state, possible_actions, debug=True)
                    plot_possible_q_values(possible_q_values)
                else:
                    action = env.set_players(player_2_mode='human_gui')

        next_state, rewards, done, _, info = env.step(action)
        env.render('terminal_display')

        if done:
            print(env.get_game_status())
            break
        else:
            env.current_step += 1

def plot_possible_q_values(possible_q_values, title="Q-values for Possible Actions"):
    if possible_q_values:
        actions = list(possible_q_values.keys())
        values = list(possible_q_values.values())

        plt.figure(figsize=(8, 4))
        plt.bar(range(len(actions)), values, tick_label=[str(action) for action in actions])
        plt.xlabel('Actions')
        plt.ylabel('Q-value')
        plt.title(title)
        plt.show()
