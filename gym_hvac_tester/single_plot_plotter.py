import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import sys
from matplotlib.image import imread
from tempfile import NamedTemporaryFile


def get_size(fig, dpi=100):
    fig.savefig('temp/file.png', bbox_inches='tight', dpi=dpi)
    height, width, _channels = imread('temp/file.png').shape
    return width / dpi, height / dpi


def set_size(fig, size, dpi=100, eps=1e-2, give_up=2, min_size_px=10):
    target_width, target_height = size
    set_width, set_height = target_width, target_height # reasonable starting point
    deltas = [] # how far we have
    while True:
        fig.set_size_inches([set_width, set_height])
        actual_width, actual_height = get_size(fig, dpi=dpi)
        set_width *= target_width / actual_width
        set_height *= target_height / actual_height
        deltas.append(abs(actual_width - target_width) + abs(actual_height - target_height))
        if deltas[-1] < eps:
            return True
        if len(deltas) > give_up and sorted(deltas[-give_up:]) == deltas[-give_up:]:
            return False
        if set_width * dpi < min_size_px or set_height * dpi < min_size_px:
            return False


def plotter(episode, results_filename, output_dir, argv=None):
    sns.set(style="darkgrid")
    df = pd.read_csv(results_filename)
    df = df[df['episode'] == episode]
    # df['time'] = pd.to_timedelta(df['time'], unit='seconds')
    x = ['time']
    y_temp = ['ground_temperature',
              'air_temperature',
              'basement_temperature',
              'main_temperature',
              'attic_temperature']
    y_action = ['hvac_temperature']
    y_action_rolling = ['rolling_mean_hvac_temperature']
    y_reward = ['total_reward']
    temperature_df = pd.melt(df[x + y_temp], id_vars=x, value_vars=y_temp)
    action_df_unmelted = df[x + y_action]
    action_df = pd.melt(action_df_unmelted, id_vars=x, value_vars=y_action)
    action_df_rolling_1hr = action_df_unmelted
    action_df_rolling_1hr[y_action_rolling] = action_df_rolling_1hr.loc[:, y_action].rolling(window=4 * 6, center=True, min_periods=1).mean()
    action_df_rolling_1hr = pd.melt(action_df_rolling_1hr[x + y_action_rolling], id_vars=x, value_vars=y_action_rolling)
    reward_df = pd.melt(df[x + y_reward], id_vars=x, value_vars=y_reward)
    # Create two subplots sharing y axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, facecolor='w', edgecolor='k', figsize=(15, 7.5))
    fig.suptitle('Mediocre Learner Controlling HVAC System Over a Week in the Winter')
    sns.lineplot(x='time', y='value', hue='variable', data=temperature_df, ax=ax1)
    ax1.axhspan(20, 23, color='#36D7B7', alpha=0.5)
    # ax2.step(x='time', y='value', data=action_df)
    sns.lineplot(x='time', y='value', hue='variable', data=action_df_rolling_1hr, palette=sns.color_palette("hls", 1), ax=ax2)
    sns.lineplot(x='time', y='value', hue='variable', data=reward_df, ax=ax3)
    for item in ax1.get_xticklabels():
        item.set_rotation(60)
    for item in ax2.get_xticklabels():
        item.set_rotation(60)
    for item in ax3.get_xticklabels():
        item.set_rotation(60)
    if argv is not None:
        ax1.set_xlim(argv['xlim_left'], argv['xlim_right'])
        #ax1.set_ylim(argv['temp_ylim_lower'], argv['temp_ylim_upper'])
        ax2.set_xlim(argv['xlim_left'], argv['xlim_right'])
        ax2.set_ylim(-1, 2)
        ax3.set_xlim(argv['xlim_left'], argv['xlim_right'])
        #ax3.set_ylim(argv['reward_ylim_lower'], argv['reward_ylim_upper'])
    # ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax2.legend().set_visible(False)
    ax3.legend().set_visible(False)
    ax1.set_ylabel('Model Temperatures (C)')
    ax2.set_ylabel('HVAC Temperature (C)\n6 Hour Rolling Mean\n')
    ax3.set_ylabel('Reward')
    plt.xlabel('Time (Seconds)')
    # set_size(fig, (15, 7.5), dpi=300)
    plt.savefig(os.path.join(output_dir, '{:0>3}.png'.format(episode)), bbox_inches='tight')
    # plt.show()
    plt.close()


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--temp_ylim_lower', type=float, default=-5)
    parser.add_argument('--temp_ylim_upper', type=float, default=40)
    parser.add_argument('--reward_ylim_lower', type=float, default=-10)
    parser.add_argument('--reward_ylim_upper', type=float, default=100)
    parser.add_argument('--xlim_left', type=float, default=0)
    parser.add_argument('--xlim_right', type=float, default=672*900)
    args = parser.parse_args(argv)
    return vars(args)


def parse_config_file(config_file_name):
    with open(config_file_name, 'r') as config_file:
        argv = config_file.read().replace('\n', '').split(' ')
    return parse_args(argv)


def __main__(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('output_dir')
    parser.add_argument('episode_upper', type=int)
    parser.add_argument('--episode_lower', type=int, default=0)
    args = parser.parse_args(argv)
    vargs = vars(args)
    arg_file_vargs = parse_config_file(vargs['config_file'])
    for episode in range(vargs['episode_lower'], vargs['episode_upper'] + 1):
        # TODO DELETE THIS
        episode = 247
        print('Plotting episode {}'.format(episode))
        plotter(episode,
                os.path.join(vargs['output_dir'], 'results.csv'),
                vargs['output_dir'],
                arg_file_vargs)
        #TODO DELETE THIS
        break


if __name__ == '__main__':
    __main__(sys.argv[1:])
