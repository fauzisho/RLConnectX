from DQN_model import train_agent_dqn, play_with_dqn
from scenario.scenario_connected4_6x7 import Scenario_Connected4_6x7

if __name__ == "__main__":

    batch_size = 32
    gamma = 0.98
    buffer_limit = 50_000
    learning_rate = 0.005
    num_episodes = 100
    num_games = 100

    training = False
    play_with_agent = False
    play_with_agent_with_scenario = True

    if training:
        train_agent_dqn(batch_size, gamma, buffer_limit, learning_rate, num_episodes, num_games)

    if play_with_agent:
        play_with_dqn(file_name="dqn_1.pth")

    if play_with_agent_with_scenario:
        valid_scenario = Scenario_Connected4_6x7().generate_permutations()[0]  # 100% agent win
        play_with_dqn(scenario=valid_scenario,
                      file_name="dqn_1.pth")
