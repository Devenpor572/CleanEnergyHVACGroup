import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import sys


def plotter(episode, results_filename, output_dir, ylim):
    df = pd.read_csv(results_filename)
    df = df[df['episode'] == episode]
    x = ['time']
    y = ['ground_temperature',
         'air_temperature',
         'hvac_temperature',
         'heat_added',
         'basement_temperature',
         'main_temperature',
         'attic_temperature',
         'total_reward',
         'reward']
    selected_df = df[x + y]
    melted_df = pd.melt(selected_df, id_vars=x, value_vars=y)
    sns.set(style="darkgrid")
    plt.figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = sns.lineplot(x='time', y='value', hue='variable', data=melted_df)
    ax.set(ylim=ylim)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(os.path.join(output_dir, '{:0>3}.png'.format(episode)), bbox_inches='tight')
    plt.show()


def __main__(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('output_dir')
    parser.add_argument('episode_upper', type=int)
    parser.add_argument('--episode_lower', type=int, default=0)
    parser.add_argument('--ylim_lower', type=float, default=-5)
    parser.add_argument('--ylim_upper', type=float, default=40)
    args = parser.parse_args(argv)
    vargs = vars(args)
    for episode in range(vargs['episode_lower'], vargs['episode_upper'] + 1):
        plotter(episode,
                os.path.join(vargs['output_dir'], 'results.csv'),
                vargs['output_dir'],
                (vargs['ylim_lower'], vargs['ylim_upper']))


if __name__ == '__main__':
    __main__(sys.argv[1:])
