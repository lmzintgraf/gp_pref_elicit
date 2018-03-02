"""
@author: Luisa M Zintgraf (2017, Vrije Universiteit Brussel)

Experiments for figure 6 in the paper.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
from gp_utilities import utils_experiment, utils_parameters

plt.figure(figsize=(10, 7))

num_obj = 2
utl_noise = 0.01

plt_idx = 1

for seed in [13, 666]:

    for query_type in ['pairwise', 'ranking']:

        plt.subplot(2, 2, plt_idx)

        params = utils_parameters.get_parameter_dict(query_type=query_type, num_objectives=num_obj,  utility_noise=utl_noise)
        params['num queries'] = 10
        params['seed'] = seed

        # run the experiment
        experiment = utils_experiment.Experiment(params)
        result = experiment.run(recalculate=False)

        # -- true utility --

        # get the input domain
        input_domain = result[1]
        idx_order = np.argsort(-input_domain[:, 0])

        # get the true utility function
        true_utility = result[2]

        # plot the true utility function
        plt.plot(input_domain[:, 0][idx_order], true_utility[idx_order], 'k', linewidth=2, label='true utility')

        # -- queried datapoints --

        # get the datapoints
        datapoints = result[5]
        utility_datapoints = result[6]

        # plot the datapoints
        plt.plot(datapoints[:, 0], utility_datapoints, 'o', color='b', linewidth=5, label='datapoints')

        # -- gaussian process --

        # get mean and variance of gaussian process
        gp_mean = result[3]
        # normalise
        gp_mean = (gp_mean-np.min(gp_mean)) / (np.max(gp_mean) - np.min(gp_mean))
        gp_var = result[4]

        # plot gp mean
        plt.plot(input_domain[:, 0][idx_order], gp_mean[idx_order], '--', color='red', linewidth=3, label='GP mean')

        # plot gp variance
        plt.fill_between(input_domain[:, 0][idx_order], gp_mean[idx_order] - gp_var[idx_order],
                         gp_mean[idx_order] + gp_var[idx_order],
                         alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848', label='GP variance')

        # -- go to next plot --

        if plt_idx == 1:
            plt.title(query_type, fontsize=20)
        if plt_idx == 2:
            plt.title(query_type, fontsize=20)
        if plt_idx == 4:
            plt.legend(fontsize=17, bbox_to_anchor=(-1.015, -0.25, 2, 1), loc=3, ncol=4, mode='expand')

        plt.yticks([])
        plt.xticks([])
        plt.xlim([np.min(input_domain[:, 0])-0.02, np.max(input_domain[:, 0])+0.02])
        plt.ylim([np.min(gp_mean[idx_order] - gp_var[idx_order])-0.02, np.max(gp_mean[idx_order] + gp_var[idx_order])+0.02])
        plt_idx += 1

# -- show plot --
plt.tight_layout(rect=(0, 0.08, 1, 1))
dir_plots = './result_plots'
if not os.path.exists(dir_plots):
    os.mkdir(dir_plots)
plt.savefig(os.path.join(dir_plots, 'gp_shape'))
plt.show()
