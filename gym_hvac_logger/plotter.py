import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime


def main():
    df = pd.read_csv('output/results.csv')
    x = ['time']
    y = ['ground_temperature',
         'air_temperature',
         'hvac_temperature',
         'head_added',
         'basement_temperature',
         'main_temperature',
         'attic_temperature',
         'reward']
    selected_df = df[x + y]
    melted_df = pd.melt(selected_df, id_vars=x, value_vars=y)
    ax = sns.lineplot(x='time', y='value', hue='variable', data=melted_df)
    ax.set(ylim=(-5, 40))
    plt.savefig('plots/' + datetime.today().strftime('%Y%m%d_%H%M%S.png'))
    plt.show()


if __name__ == '__main__':
    main()
