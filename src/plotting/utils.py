import os
import warnings
import numpy as np


def load_data(path, name, success_threshold=None):
    with np.load(path) as data:
        # for key in data:
        #     print(key)

        try:
            t = data['t']
        except:
            t = data['timesteps']

        if name == 'se_normalized':
            r = np.clip(data['diff_kl_mle_target']/np.abs(data['ref_kl_mle_target']), -10000, 100000)
        elif name == 'diff_kl_mle_target':
            r = np.clip(data['diff_kl_mle_target'], -10000,10000)
        else:
            r = data[name]
        if success_threshold is not None:
            r = r > success_threshold

        if len(r.shape) > 1:
            avg = np.average(r, axis=1)
        else:
            avg = r

    return t, t, avg


def plot(save_dict, name, m=100000, success_threshold=None, return_cutoff=-np.inf):
    i = 0

    # palette = seaborn.color_palette()
    print(os.getcwd())

    for agent, info in save_dict.items():
        paths = info['paths']
        x_scale = info['x_scale']
        max_t = info['max_t']
        avgs = []
        for path in paths:
            u, t, avg = load_data(path, name=name, success_threshold=success_threshold)
            if avg is not None:
                if max_t:
                    cutoff = np.where(t <= max_t/x_scale)[0]
                    avg = avg[cutoff]
                    t = t[cutoff]

                elif m:
                    avg = avg[:m]
                avgs.append(avg)
                t_good = t

        if len(avgs) == 0:
            continue
        elif len(avgs) == 1:
            avg_of_avgs = avg
            q05 = np.zeros_like(avg)
            q95 = np.zeros_like(avg)

        else:

            min_l = np.inf
            for a in avgs:
                l = len(a)
                if l < min_l:
                    min_l = l

            if min_l < np.inf:
                for i in range(len(avgs)):
                    avgs[i] = avgs[i][:min_l]

            avg_of_avgs = np.mean(avgs, axis=0)

            # if avg_of_avgs.mean() > 0: continue
            # print(np.median(avg_of_avgs))
            # if np.median(avg_of_avgs) > 0: continue

            std = np.std(avgs, axis=0)
            N = len(avgs)
            ci = 1.96 * std / np.sqrt(N) * 1.96
            q05 = avg_of_avgs - ci
            q95 = avg_of_avgs + ci


            # if avg_of_avgs[-10:].mean() < 4900 or N < 10: continue

        style_kwargs = get_line_styles(agent)
        style_kwargs['linewidth'] = 2

        # style_kwargs['linewidth'] = 1.5

        style_kwargs['color'] = None
        # if 'PROPS' in agent:
        #     style_kwargs['linestyle'] = '-'
        #     style_kwargs['linewidth'] = 3
        #     # style_kwargs['color'] = 'k'
        #
        #
        # elif 'ppo_buffer' in agent or 'PPO-Buffer' in agent or 'b=' in agent or 'Buffer' in agent:
        #     style_kwargs['linestyle'] = '--'
        # elif 'ppo,' in agent or 'PPO,' in agent or 'PPO with' in agent or 'PPO' == agent:
        #     style_kwargs['linestyle'] = ':'
        # elif 'Priv' in agent:
        #     style_kwargs['linestyle'] = '-.'
        #
        # elif '0.0001' in agent:
        #     style_kwargs['linestyle'] = '--'

        # print(agent, N, avg_of_avgs[-1], q05[-1], q95[-1])

        try:
            times = info['times']
            x = times
        except:
            x = t_good * x_scale
            if t is None:
                x = np.arange(len(avg_of_avgs))
            if m:
                x = x[:m]
                avg_of_avgs = avg_of_avgs[:m]
                q05 = q05[:m]
                q95 = q95[:m]
        plt.plot(x, avg_of_avgs, label=agent, **style_kwargs)
        if style_kwargs['linestyle'] == 'None':
            plt.fill_between(x, q05, q95, alpha=0)
        else:
            plt.fill_between(x, q05, q95, alpha=0.2)
        # plt.fill_between(x, q05, q95, alpha=0.2, color=style_kwargs['color'])

        i += 1
    # return fig


def get_paths(results_dir, filename='evaluations.npz'):
    paths = []
    for subdir in os.listdir(results_dir):
        if 'run_' in subdir:
            paths.append(f'{results_dir}/{subdir}/{filename}')
    return paths


def get_data(results_dir, field_name='returns', filename='evaluations.npz'):

    try:
        paths = get_paths(results_dir, filename)
    except:
        warnings.warn(f'Data not found at path {results_dir}')
        paths = []

    timesteps = None
    results = []
    for path in paths:

        with np.load(path) as data:

            vals = data[field_name]
            if len(vals.shape) > 1:
                avg_vals = np.average(vals, axis=1)
            else:
                avg_vals = vals

            results.append(avg_vals)
            timesteps = data['timesteps']

    return timesteps, np.array(results)
