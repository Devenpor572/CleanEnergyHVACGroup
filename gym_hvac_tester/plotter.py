import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import sys


def plotter(episode, results_filename, image_filename, ylim):
    df = pd.read_csv(results_filename)
    df = df[df['episode'] == episode]
    x = ['time']
    y = ['ground_temperature',
         'air_temperature',
         'hvac_temperature',
         'head_added',
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
    plt.savefig(image_filename, bbox_inches='tight')
    plt.show()


def __main__(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('episode', type=int)
    parser.add_argument('results_filename')
    parser.add_argument('image_filename')
    parser.add_argument('--ylim_lower', type=float, default=-5)
    parser.add_argument('--ylim_upper', type=float, default=40)
    args = parser.parse_args(argv)
    vargs = vars(args)
    plotter(vargs['episode'],
            vargs['results_filename'],
            vargs['image_filename'],
            (vargs['ylim_lower'], vargs['ylim_upper']))


if __name__ == '__main__':
    __main__(sys.argv[1:])
