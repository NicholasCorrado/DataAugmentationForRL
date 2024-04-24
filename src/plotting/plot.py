import os

import numpy as np

import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn

from src.plotting.utils import get_paths,get_data

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils
from rliable.plot_utils import _decorate_axis, _annotate_and_decorate_axis, plot_sample_efficiency_curve

if __name__ == "__main__":


    timestep_dict = {}
    results_dict = {}

    nrows = 2
    ncols = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*3, nrows*3))
    axs = axs.flatten()

    env_ids = ['PointMaze_UMaze-v3']
    algos = ['ddpg']

    for env_id, ax in zip(env_ids, axs):
        # add all results you want to plot on a single subplot
        results_dict = {}
        for algo in algos:
            results_dir = f"../results/train/results/{env_id}/{algo}/No_DA"
            timesteps, results = get_data(results_dir=results_dir)

            # A warning will be raised when we fail to load from `results_dir`. Skip these failures.
            if len(results) > 0:
                key = algo
                results_dict[key] = results

        results_dict = {algorithm: score for algorithm, score in results_dict.items()}
        iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., frame])
                                       for frame in range(scores.shape[-1])])
        iqm_scores, iqm_cis = rly.get_interval_estimates(results_dict, iqm, reps=500)

        ax.set_title(f'{env_id}')

        plot_sample_efficiency_curve(
            timesteps, # assumes `timesteps` is the same for all curves
            iqm_scores,
            iqm_cis,
            ax=ax,
            algorithms=None,
            xlabel='Timesteps',
            ylabel='IQM Return',
            labelsize=12,
            ticklabelsize=12,
            # legend=True,
        )

    # plt.suptitle('Training Curves')
    plt.tight_layout()

    # Push plots down to make room for the the legend
    fig.subplots_adjust(left=0.1, top=0.87)
    # Fetch and plot the legend from one of the subplots.
    ax = fig.axes[1]
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize='large')

    save_dir = f'figures'
    save_name = f'return_iqm.png'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{save_name}', dpi=300)
    plt.show()
