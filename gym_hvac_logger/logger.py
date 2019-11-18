import gym
import gym_hvac
import csv
import json
import numpy as np


def main():
    with open('output/schedule.json', 'r') as infile:
        data = json.load(infile)
    hvac_env = gym.make("HVAC-v0")

    # Override parameters
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
                             'ground_temperature',
                             'air_temperature',
                             'hvac_temperature',
                             'basement_temperature',
                             'main_temperature',
                             'attic_temperature',
                             'action',
                             'hvac_temp',
                             'reward',
                             'terminal'])

        csv_writer.writerow([0] + 
                            hvac_env.state.tolist() +
                            [1, 0, 0, False])

    for action in data['action_schedule']:
        state_next, reward, terminal, info = hvac_env.step(action)
        with open('output/results.csv', 'a', newline='') as outfile:
            csv_writer = csv.writer(outfile)
            csv_writer.writerow([hvac_env.time] +
                                state_next.tolist() +
                                [action, hvac_env.get_hvac(action), reward, terminal])
        if terminal:
            break


if __name__ == '__main__':
    main()
