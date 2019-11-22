import argparse
import csv
from datetime import datetime
import gym
import gym_hvac
import json
import numpy as np
import os
import sys
import re

import plotter


# HVAC Environment simulator
# Runs the environment and logs the results to a csv file
def simulator(input_filename, results_filename):
    with open(input_filename, 'r') as infile:
        data = json.load(infile)
        data['action_schedule'] = data['action_schedule'].replace('-', '0').replace('.', '1').replace('+', '2')
    hvac_env = gym.make("HVAC-v0")

    # Override parameters
    hvac_env.desired_temperature_low = data['desired_temperature_low']
    hvac_env.desired_temperature_mean = data['desired_temperature_mean']
    hvac_env.desired_temperature_high = data['desired_temperature_high']
    hvac_env.lower_temperature_threshold = data['lower_temperature_threshold']
    hvac_env.upper_temperature_threshold = data['upper_temperature_threshold']
    hvac_env.step_limit = data['step_limit']
    hvac_env.ground_temperature = data['ground_temperature']
    hvac_env.air_temperature = data['air_temperature']
    hvac_env.hvac_temperature = data['hvac_temperature']
    hvac_env.tau = data['tau']
    hvac_env.termination = data['termination']

    # Set the initial state
    hvac_env.state = np.array([hvac_env.get_air_temperature(0),
                               hvac_env.get_ground_temperature(0),
                               0,
                               data['basement_temperature'],
                               data['main_temperature'],
                               data['attic_temperature']])

    # Write initial state
    with open(results_filename, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(['episode',
                             'step',
                             'time',
                             'air_temperature',
                             'ground_temperature',
                             'hvac_temperature',
                             'basement_temperature',
                             'main_temperature',
                             'attic_temperature',
                             'heat_added',
                             'action',
                             'reward',
                             'total_reward',
                             'terminal'])

        csv_writer.writerow([0, hvac_env.step, hvac_env.time] +
                            hvac_env.state.tolist() +
                            [0, 1, 0, 0, False])

    for action in data['action_schedule']:
        state_next, reward, terminal, info = hvac_env.step(int(action))
        with open(results_filename, 'a', newline='') as outfile:
            csv_writer = csv.writer(outfile)
            csv_writer.writerow([0, hvac_env.step, hvac_env.time] +
                                state_next.tolist() +
                                [hvac_env.total_heat_added, int(action), reward, hvac_env.total_reward, terminal])
        if terminal:
            break


# Run-Length Encoding Decoder
# This encoder handles nested groups using parenthesis allowing for repeated patterns
def decode(rle_str):
    mutable_rle_str = rle_str
    decoded_str = ''
    # States
    # 0 - Initial no-context state
    # 1 - Accepting multiplier string
    state = 0
    multiplier = 0
    group = ''
    group_stack = []
    while mutable_rle_str:
        result1 = re.match(r'^(\d*)(.*)', mutable_rle_str)
        result2 = re.match(r'^([^\d\(\)])(.*)', mutable_rle_str)
        result3 = re.match(r'^(\()(.*)', mutable_rle_str)
        result4 = re.match(r'^(\))(.*)', mutable_rle_str)
        if state == 0:
            if result1 and result1.group(1):
                multiplier = int(result1.group(1))
                mutable_rle_str = result1.group(2)
                state = 1
            elif result2 and result2.group(1):
                decoded_str += result2.group(1)
                group += result2.group(1)
                mutable_rle_str = result2.group(2)
                # state = 0
            elif result3 and result3.group(1):
                if group_stack:
                    group_stack[-1] = (group_stack[-1][0], group_stack[-1][1] + group)
                    group = ''
                group_stack.append((1, group))
                mutable_rle_str = result3.group(2)
                state = 0
            elif result4 and result4.group(1):
                decoded_str = group_stack[-1][0] * (group_stack[-1][1] + group)
                group = ''
                mutable_rle_str = result4.group(2)
            else:
                raise argparse.ArgumentError('action_schedule invalid format')
        elif state == 1:
            if result2 and result2.group(1):
                decoded_str += result2.group(1) * multiplier
                group += result2.group(1) * multiplier
                multiplier = 0
                mutable_rle_str = result2.group(2)
                state = 0
            elif result3 and result3.group(1):
                if group_stack:
                    group_stack[-1] = (group_stack[-1][0], group_stack[-1][1] + group)
                    group = ''
                group_stack.append((multiplier, ''))
                multiplier = 0
                mutable_rle_str = result3.group(2)
                state = 0
            else:
                raise argparse.ArgumentError('action_schedule invalid format')
    return decoded_str


def interface(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('action_schedule')
    parser.add_argument('--file', default=datetime.today().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--desired_temperature_low', type=float, default=20)
    parser.add_argument('--desired_temperature_mean', type=float, default=21.5)
    parser.add_argument('--desired_temperature_high', type=float, default=23)
    parser.add_argument('--lower_temperature_threshold', type=float, default=10)
    parser.add_argument('--upper_temperature_threshold', type=float, default=33)
    parser.add_argument('--step_limit', type=int, default=672)
    parser.add_argument('--ground_temperature', type=float, default=10)
    parser.add_argument('--air_temperature', type=float, default=0)
    # Roughly 1 degree every five minutes
    parser.add_argument('--hvac_temperature', type=float, default=0.00333)
    parser.add_argument('--basement_temperature', type=float, default=15)
    parser.add_argument('--main_temperature', type=float, default=20)
    parser.add_argument('--attic_temperature', type=float, default=25)
    parser.add_argument('--tau', type=float, default=900)
    parser.add_argument('--ylim_lower', type=float, default=-5)
    parser.add_argument('--ylim_upper', type=float, default=40)
    parser.add_argument('--termination', action='store_true')

    args = parser.parse_args(argv)
    vargs = vars(args)
    vargs['action_schedule'] = decode(vargs['action_schedule'])
    vargs['file'] = os.path.join('output', vargs['file'])
    os.makedirs(vargs['file'])
    vargs['schedule_file'] = os.path.join(vargs['file'], 'schedule.json')
    vargs['results_file'] = os.path.join(vargs['file'], 'results.csv')
    vargs['image_file'] = os.path.join(vargs['file'], 'plot.png')
    with open(vargs['schedule_file'], 'w') as outfile:
        json.dump(vargs, outfile)
    return vargs


def main(argv):
    vargs = interface(argv)
    simulator(vargs['schedule_file'], vargs['results_file'])
    plotter.plotter(0, vargs['results_file'], vargs['file'], (vargs['ylim_lower'], vargs['ylim_upper']))


if __name__ == '__main__':
    main(sys.argv[1:])
