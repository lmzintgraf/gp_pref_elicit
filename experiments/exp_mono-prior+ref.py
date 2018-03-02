"""
@author: Luisa M Zintgraf (2017, Vrije Universiteit Brussel)

Experiments for figure 4 in the paper.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
from gp_utilities import utils_experiment, utils_parameters

start_seed = 13
num_queries = 25
num_iter = 100
num_obj = 5

plt.figure(figsize=(10.3, 5))
plt_idx = 1

for noise_level in [0.01]:

    plt.title('{} obj., {} noise, avg over {}'.format(num_obj, noise_level, num_iter))

    for s in [
        [
            ['linear-zero', [False, False]],
            ['linear', [False, False]],
            ['zero', [False, False]],
        ],
        [
            ['zero', ['beginning', 'beginning']],
            ['zero', ['full', 'full']],
            ['zero', [False, False]],
        ],
        [
            ['linear-zero', ['full', 'full']],
            ['linear-zero', [False, False]],
            ['zero', ['full', 'full']],
        ],
    ]:
        for [prior_type, reference_points] in s:

            plt.subplot(1, 3, plt_idx)

            # 5 is the number of diff query types;
            # 50 is the number of queries we ask
            all_query_types = ['pairwise', 'clustering', 'ranking', 'top_rank']
            utl_vals = np.zeros((len(all_query_types) * num_iter, num_queries))
            iter_idx = 0

            for query_type in all_query_types:

                params = utils_parameters.get_parameter_dict(query_type=query_type, num_objectives=num_obj, utility_noise=noise_level)
                params['reference min'] = reference_points[0]
                params['reference max'] = reference_points[1]
                params['gp prior mean'] = prior_type
                params['num queries'] = num_queries

                params['seed'] = start_seed

                for _ in range(num_iter):

                    experiment = utils_experiment.Experiment(params)
                    result = experiment.run(recalculate=False)

                    utl_vals[iter_idx] = result[0]
                    params['seed'] += 1
                    iter_idx += 1

            style = '-'
            color = 'limegreen'
            if params['gp prior mean'] == 'zero' and (params['reference min'] == False and params['reference max'] == False):
                color = 'black'
                style = '-'
            elif params['gp prior mean'] == 'linear' and (params['reference min'] == False and params['reference max'] == False):
                color = 'maroon'
                style = ':'
            elif params['gp prior mean'] == 'linear-zero' and (params['reference min'] == False and params['reference max'] == False):
                color = 'darkorange'
                style ='--'
            elif params['gp prior mean'] == 'zero' and (params['reference min'] == 'beginning' or params['reference max'] == 'beginning'):
                color = 'turquoise'
            elif params['gp prior mean'] == 'zero' and (params['reference min'] == 'full' or params['reference max'] == 'full'):
                color = 'royalblue'
                style = ':'
            elif params['gp prior mean'] == 'linear-zero' and (params['reference min'] == 'beginning' or params['reference max'] == 'beginning'):
                color = 'limegreen'
                style = '--'
            else:
                print("you forgot something....")
                print(params['gp prior mean'])
                print(params['reference min'])

            if params['gp prior mean'] == 'linear-zero':
                params['gp prior mean'] = 'lin. prior (start)'
            elif params['gp prior mean'] == 'linear':
                params['gp prior mean'] = 'lin. prior (full)'
            else:
                params['gp prior mean'] = 'zero prior'

            if params['reference min'] == 'beginning':
                params['reference min'] = 'start'
            if params['reference max'] == 'beginning':
                params['reference max'] = 'start'

            if plt_idx == 1:
                label = '{}'.format(params['gp prior mean'], fontsize=15)
            elif plt_idx == 2:
                if params['reference min'] or params['reference max']:
                    label = 'ref. points ({})'.format(params['reference min']) if params['reference max']==False else 'ref. points ({})'.format(params['reference max'])
                else:
                    label = 'no ref. points'
            else:
                if params['reference min'] != False or params['reference max'] != False:
                    label = '{}, \nref. points ({})'.format(params['gp prior mean'], params['reference min'])
                else:
                    label = '{}, \nno ref. points'.format(params['gp prior mean'], params['reference min'])

            plt.plot(range(1, num_queries+1), np.mean(utl_vals, axis=0), style, color=color, label=label, linewidth=3)

        if plt_idx > 1:
            plt.yticks([])
        else:
            plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=15)
            plt.ylabel('utility', fontsize=20)

        plt.ylim([0.4, 0.95])
        plt.xlim([1, num_queries])
        plt.xticks([1, 5, 10, 15, 20, 25], fontsize=15)
        plt.xlabel('query', fontsize=20)
        plt.legend(fontsize=13, loc=4)
        plt.gca().set_ylim(top=1.0)
        plt_idx += 1

plt.tight_layout(rect=(-0.015, -0.02, 1.015, 1.02))
dir_plots = './result_plots'
if not os.path.exists(dir_plots):
    os.mkdir(dir_plots)
plt.savefig(os.path.join(dir_plots, 'mono_prior+refpoints_{}'.format(num_iter)))
plt.show()
