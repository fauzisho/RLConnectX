from Q_learning import train_agent_env, play_with_q_table
from scenario.scenario_3x3 import Scenario_3x3

if __name__ == "__main__":
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    no_episodes = 10

    training = False
    play_with_agent = True
    play_with_agent_with_scenario = False

    if training:
        train_agent_env(alpha,gamma,epsilon,no_episodes)

    if play_with_agent:
        play_with_q_table(file_name="case_78_player_1_connect3_3x3.npy")

    if play_with_agent_with_scenario:
        valid_scenario = Scenario_3x3().generate_permutations()[2]  # 100% agent win
        play_with_q_table(scenario=valid_scenario,
                          file_name="case_78_player_1_connect3_3x3.npy")
