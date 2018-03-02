"""
@author: Luisa M Zintgraf (2017, Vrije Universiteit Brussel)

Experiments for figure 5 in the paper.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(0, '.')
from gp_utilities import utils_experiment, utils_parameters

start_seed = 13
num_queries = 25
num_iter = 100
num_obj = 5

plt.figure(figsize=(6.5, 7))
plt_idx = 1

for prior_type, ref_min, ref_max in [['zero', False, False], ['linear-zero', 'full', 'full']]:

    for utl_noise in [0.01, 0.1]:

        plt.subplot(2, 2, plt_idx)

        # loop through query types
        for query_type in ['pairwise', 'clustering', 'ranking', 'clustering4', 'top_rank']:

            # get the  parameters
            params = utils_parameters.get_parameter_dict(num_objectives=num_obj, query_type=query_type, utility_noise=utl_noise)

            params['num queries'] = num_queries

            params['gp prior mean'] = prior_type
            params['reference min'] = ref_min
            params['reference max'] = ref_max

            if params['query type'] == 'clustering4':
                params['query type'] = 'clustering'
                params['num clusters'] = 4

            params['seed'] = start_seed

            utilities = np.zeros((num_iter, params['num queries']))

            for iter_idx in range(num_iter):

                experiment = utils_experiment.Experiment(params)
                result = experiment.run(recalculate=False)

                utilities[iter_idx] = result[0]
                params['seed'] += 1

            if params['query type'] == 'clustering':
                params['query type'] = 'clustering ({})'.format(params['num clusters'])
            plt.plot(range(1, params['num queries']+1), np.mean(utilities, axis=0), label=params['query type'], linewidth=2)

            if plt_idx == 1:
                plt.legend(bbox_to_anchor=(-0.1, 1, 2.2, 1.1), loc=3, fontsize=15, ncol=3, mode='expand')
            if plt_idx > 2:
                plt.xlabel('query', fontsize=15)
                plt.xticks([1, 5, 10, 15, 20, 25])
            else:
                plt.xticks([])
            if plt_idx == 1 or plt_idx == 3:
                plt.ylabel('utility', fontsize=15)
            else:
                plt.yticks([])
            plt.xlim([0.5, 25.5])
            plt.ylim([0.4, 1])

        plt_idx += 1

plt.gcf().tight_layout(rect=(-0.01, -0.01, 1.02, 0.89))
dir_plots = './result_plots'
if not os.path.exists(dir_plots):
    os.mkdir(dir_plots)
plt.savefig(os.path.join(dir_plots, 'queries_{}'.format(num_iter)))
plt.show()
