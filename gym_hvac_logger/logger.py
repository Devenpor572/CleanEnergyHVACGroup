import gym
import gym_hvac
import csv
import json
import numpy as np


def main():
    with open('output/schedule.json', 'r') as infile:
        data = json.load(infile)
        data['action_schedule'] = data['action_schedule'].replace('-', '0').replace('.', '1').replace('+', '2')
    hvac_env = gym.make("HVAC-v0")

    # Override parameters
    hvac_env.desired_temperature_low = data['desired_temperature_low']
    hvac_env.desired_temperature_mean = data['desired_temperature_mean']
    hvac_env.desired_temperature_high = data['desired_temperature_high']
    hvac_env.lower_temperature_threshold = data['lower_temperature_threshold']
    hvac_env.upper_temperature_threshold = data['upper_temperature_threshold']
    hvac_env.ground_temperature = data['ground_temperature']
    hvac_env.air_temperature = data['air_temperature']
    hvac_env.hvac_temperature = data['hvac_temperature']
    hvac_env.tau = data['tau']

    # Set the initial state
    hvac_env.state = np.array([hvac_env.get_air_temperature(0),
                               hvac_env.get_ground_temperature(0),
                               0,
                               data['basement_temperature'],
                               data['main_temperature'],
                               data['attic_temperature']])

    # Write initial state
    with open('output/results.csv', 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(['time',
                             'air_temperature',
                             'ground_temperature',
                             'hvac_temperature',
                             'basement_temperature',
                             'main_temperature',
                             'attic_temperature',
                             'head_added',
                             'action',
                             'reward',
                             'terminal'])

        csv_writer.writerow([0] + 
                            hvac_env.state.tolist() +
                            [0, 1, 0, False])

    for action in data['action_schedule']:
        state_next, reward, terminal, info = hvac_env.step(int(action))
        with open('output/results.csv', 'a', newline='') as outfile:
            csv_writer = csv.writer(outfile)
            csv_writer.writerow([hvac_env.time] +
                                state_next.tolist() +
                                [hvac_env.total_heat_added, int(action), reward, terminal])
        if terminal:
            break


if __name__ == '__main__':
    main()
