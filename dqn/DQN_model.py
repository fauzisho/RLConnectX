import torch
from matplotlib import pyplot as plt
from torch import optim
from tqdm import tqdm

from Util import QConnectNet, ReplayBufferConnectGame, trainDQN
from gymconnectx.envs import ConnectGameEnv
from scenario.scenario_connected4_6x7 import Scenario_Connected4_6x7


def env_connected4_6x7():
    env = ConnectGameEnv(
        connect=4,
        width=7,
        height=6,
        reward_winner=10,
        reward_loser=-10,
        reward_draw=1,
        reward_hell=-5,
        reward_hell_prob=-1.5,
        reward_win_prob=1.5,
        reward_living=-1,
        obs_number=True)
    return env


def train_agent_dqn(batch_size, gamma, buffer_limit, learning_rate, num_episodes, num_games):
    env = env_connected4_6x7()
    no_actions = env.width
    no_states = env.width * env.height

    q_net = QConnectNet(no_actions=no_actions, no_states=no_states)
    q_target = QConnectNet(no_actions=no_actions, no_states=no_states)
    q_target.load_state_dict(q_net.state_dict())

    memory = ReplayBufferConnectGame(buffer_limit=buffer_limit)

    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)

    rewards_plot_player1 = []
    rewards_plot_player2 = []
    episode_reward_player_1 = 0.0
    episode_reward_player_2 = 0.0

    scenario = Scenario_Connected4_6x7()
    valid_scenario_train = [scenario.generate_permutations()[0]]

    for i in tqdm(range(len(valid_scenario_train)), desc="Scenario Episodes"):
        scenario_player_1_and_2 = valid_scenario_train[i]
        for n_episode in tqdm(range(num_episodes), desc="Training Episodes"):
            epsilon = max(0.01, 0.1 - 0.005 * (n_episode / 100))
            for _ in tqdm(range(num_games), desc="Games per Episode", leave=False):
                last_state = env.reset()
                last_action = -1
                last_player = ""
                while not env.is_done:
                    try:
                        current_player = 'player_1' if env.get_current_player() == 1 else 'player_2'
                        state = env.get_obs()
                        # action = q_net.sample_action(torch.from_numpy(state).float(), epsilon, env.get_moves())
                        if env.current_step < 6:
                            action = scenario_player_1_and_2[env.current_step]  # based on scenario
                        else:
                            if current_player == 'player_1':
                                action = q_net.sample_action(torch.from_numpy(state).float(), epsilon, env.get_moves())
                            else:
                                action = env.get_action_random()

                        next_state, rewards, done, _, info = env.step(action)
                        reward = rewards[current_player]

                        done_mask = 0.0 if done else 1.0
                        memory.put((state, action, reward, next_state, done_mask))

                        if current_player == 'player_1':
                            episode_reward_player_1 += reward
                        else:
                            episode_reward_player_2 += reward

                        if done:
                            reward = rewards[last_player]
                            memory.put((last_state, last_action, reward, state, done_mask))

                            trainDQN(q_net, q_target, memory, optimizer, batch_size, gamma)
                            break
                        else:
                            last_player = current_player
                            last_state = state
                            last_action = action
                            env.current_step += 1

                    except Exception as e:
                        print(f"An error occurred: {str(e)}")
                        break
            q_target.load_state_dict(q_net.state_dict())
            rewards_plot_player1.append(episode_reward_player_1)
            rewards_plot_player2.append(episode_reward_player_2)
            episode_reward_player_1 = 0.0
            episode_reward_player_2 = 0.0

    # Save the trained Q-net
    torch.save(q_net.state_dict(), f"dqn_{1}.pth")

    # Plot the training curve
    plt.plot(rewards_plot_player1, label='Reward per Episode (Player 1)')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.legend()
    plt.savefig(f"training_curve_player_1_{1}.png")
    plt.show()

    plt.plot(rewards_plot_player2, label='Reward per Episode (Player 2)')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.legend()
    plt.savefig(f"training_curve_player_2_{1}.png")
    plt.show()


def play_with_dqn(scenario=None, file_name=""):
    env = env_connected4_6x7()

    no_actions = env.width
    no_states = env.width * env.height

    env.reset()

    dqn = QConnectNet(no_actions=no_actions, no_states=no_states)
    dqn.load_state_dict(torch.load(file_name))

    while not env.is_done and env.current_step < env.max_steps:
        try:
            if scenario is None:
                if env.get_current_player() == 1:
                    action = dqn(torch.from_numpy(env.get_obs()).float())
                    mask = torch.full(action.size(), float('-inf'))
                    mask[env.get_moves()] = 0
                    action = action + mask
                    action = action.argmax().item()
                else:
                    action = env.set_players(player_2_mode='human_gui')
            else:
                if env.current_step < 6:
                    action = scenario[env.current_step]
                else:
                    if env.get_current_player() == 1:
                        action = dqn(torch.from_numpy(env.get_obs()).float())
                        mask = torch.full(action.size(), float('-inf'))
                        mask[env.get_moves()] = 0
                        action = action + mask
                        action = action.argmax().item()
                    else:
                        action = env.set_players(player_2_mode='human_gui')

            observations, rewards, done, _, info = env.step(action)
            env.render(mode='terminal_display')
            env.render(mode='gui_update_display')

            print(env.get_game_status())
            if done:
                break
            else:
                env.current_step += 1

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            break
