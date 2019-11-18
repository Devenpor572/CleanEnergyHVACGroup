import json
from itertools import groupby
import argparse
import sys
import re
# from re import sub

#
# def decode(text):
#     '''
#     Doctest:
#         >>> decode('12W1B12W3B24W1B14W')
#         'WWWWWWWWWWWWBWWWWWWWWWWWWBBBWWWWWWWWWWWWWWWWWWWWWWWWBWWWWWWWWWWWWWW'
#     '''
#
#     return sub(r'(\d+)(\D)', lambda m: m.group(2) * int(m.group(1)), text)


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


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('action_schedule')
    parser.add_argument('--desired_temperature_low', type=float, default=20)
    parser.add_argument('--desired_temperature_mean', type=float, default=21.5)
    parser.add_argument('--desired_temperature_high', type=float, default=23)
    parser.add_argument('--lower_temperature_threshold', type=float, default=-1000)
    parser.add_argument('--upper_temperature_threshold', type=float, default=1000)
    parser.add_argument('--ground_temperature', type=float, default=10)
    parser.add_argument('--air_temperature', type=float, default=0)
    # Roughly 1 degree every five minutes
    parser.add_argument('--hvac_temperature', type=float, default=0.00333)
    parser.add_argument('--basement_temperature', type=float, default=15)
    parser.add_argument('--main_temperature', type=float, default=20)
    parser.add_argument('--attic_temperature', type=float, default=25)
    parser.add_argument('--tau', type=float, default=300)

    args = parser.parse_args(argv)
    data = vars(args)
    data['action_schedule'] = decode(data['action_schedule'])
    with open('output/schedule.json', 'w') as outfile:
        json.dump(data, outfile)


if __name__ == '__main__':
    main(sys.argv[1:])
