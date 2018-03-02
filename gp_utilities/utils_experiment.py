"""
@author: Luisa M Zintgraf (2017, Vrije Universiteit Brussel)
"""
import os
import numpy as np
import sys
sys.path.insert(0, '..')
from gp_utilities import utils_parameters, utils_ccs
from gp_utilities.utils_user import UserPreference
import acquisition_function
from dataset import DatasetPairwise
from gaussian_process import GPPairwise

# get the folder where we store the results
PATH_RESULTS_DIR = os.path.join('../experiments/results')
if not os.path.isdir(PATH_RESULTS_DIR):
    os.makedirs(PATH_RESULTS_DIR)


class Experiment:
    """
    This class is one experiment.
    If you want to make any alterations, use this class as a parent and override its methods.
    """
    def __init__(self, parameters):

        self.params = parameters

        # make sure nobody accidentally added new parameters (by e.g. spelling mistakes)
        assert len(parameters) == len(utils_parameters.get_parameter_dict())

        # get the input domain on which the utility is defined
        self.input_domain = self.initialise_input_domain()

        # get a user with preferences over the input domain
        self.user = self.initialise_user()

        # get an acquirer which will pick the queries
        self.acquirer = self.initialise_acquirer()

        # initialise the Gaussian process
        self.gp = self.initialise_gaussian_process()

        # initialise dataset
        self.dataset = self.initialise_dataset()

        self.curr_x_max = None
        self.curr_x_new = None

    def run(self, recalculate=False):

        # get the path where we store the results and return this if they exist and we don't want to recalculate
        path_result_file = os.path.join(PATH_RESULTS_DIR, utils_parameters.get_filename(self.params) + '.npy')
        if (not recalculate) and os.path.exists(path_result_file):
            print('loading...', path_result_file)
            return np.load(path_result_file)

        # keep track of the maximum utility found
        max_utility_per_query = np.empty(self.params['num queries'])

        # loop: ask queries and update the gaussian process
        for q in range(self.params['num queries']):

            print("... query ", q + 1)

            # get the datapoint(s) for the next (first) query
            if q == 0:
                self.curr_x_max, self.curr_x_new = self.acquirer.get_start_points(self.gp)
            else:
                self.curr_x_new = self.acquirer.get_next_point(self.gp, self.dataset)

            if self.params['reference min'] == 'full':
                self.dataset.add_single_comparison(self.curr_x_new, np.zeros(self.params['num objectives']))
            elif self.params['reference min'] == 'beginning' and q < 5:
                self.dataset.add_single_comparison(self.curr_x_new, np.zeros(self.params['num objectives']))

            if self.params['reference max'] == 'full':
                self.dataset.add_single_comparison(np.ones(self.params['num objectives']), self.curr_x_new)
            elif self.params['reference max'] == 'beginning' and q < 5:
                self.dataset.add_single_comparison(np.ones(self.params['num objectives']), self.curr_x_new)

            new_x_max = self.make_query(x_max=self.curr_x_max, x_new=self.curr_x_new)

            if self.params['gp prior mean'] == 'linear-zero' and q > 4:
                self.gp.prior_mean_type = 'zero'

            print(self.dataset.comparisons)
            self.gp.update(self.dataset)

            y_max = self.user.get_preference(new_x_max, add_noise=False)[0]
            max_utility_per_query[q] = y_max

            self.curr_x_max = new_x_max

        results = self.gather_results(max_utility_per_query)

        np.save(path_result_file, results)

        return results

    def gather_results(self, max_utility_per_query):
        # ccs_grid_true_utility = user_preference.get_preference(input_domain, add_noise=False)
        true_utility = self.user.get_preference(self.input_domain, add_noise=False)
        # ccs_grid_approx_utility = gp.get_predictive_params(input_domain, pointwise=True)
        gp_pred_mean = np.zeros(self.input_domain.shape[0])
        gp_pred_var = np.zeros(self.input_domain.shape[0])
        batch_size = 64
        for curr_idx in range(0, self.input_domain.shape[0]+batch_size, batch_size):
            gp_pred = self.gp.get_predictive_params(self.input_domain[curr_idx:curr_idx+batch_size], True)
            gp_pred_mean[curr_idx:curr_idx+batch_size] = gp_pred[0]
            gp_pred_var[curr_idx:curr_idx+batch_size] = gp_pred[1]

        results = [max_utility_per_query, self.input_domain, true_utility, gp_pred_mean, gp_pred_var,
                   self.acquirer.history, self.user.get_preference(self.acquirer.history, add_noise=False)]
        return results

    def make_query(self, x_max, x_new):
        # add new data to dataset
        # -> if preference information is given as pairwise comparisons
        if self.params["query type"] == 'pairwise':
            comp_result = self.user.pairwise_comparison(x_max, x_new, add_noise=True)
            if comp_result:
                self.dataset.add_single_comparison(x_max, x_new)
            else:
                self.dataset.add_single_comparison(x_new, x_max)
                x_max = x_new
        elif self.params["query type"] == 'clustering':
            # include_winner = parameters.winner_from <= q + 1
            clusters = self.user.clustering(self.acquirer.history, num_clusters=self.params["num clusters"], add_noise=True, include_winner=True)
            self.dataset.add_clustered_preferences(clusters, keep_prev_info=self.params["keep previous info"])
            x_max = clusters[0][0]
        elif self.params['query type'] == 'ranking':
            ranking = self.user.ranking(self.acquirer.history, add_noise=True)
            self.dataset.add_ranked_preferences(ranking)
            x_max = ranking[0]
        elif self.params['query type'] == 'top_rank':
            top_rank = self.user.top_rank(self.acquirer.history, n_rank=3, add_noise=True)
            self.dataset.add_top_rank_preferences(top_rank, keep_prev_info=True)
            x_max = top_rank[0][0]
        elif self.params['query type'] == 'best_worst':
            best_worst = self.user.best_worst(self.acquirer.history, add_noise=True)
            self.dataset.add_best_worst_preferences(best_worst, keep_prev_info=True)
            x_max = best_worst[0]
        else:
            raise TypeError("query type {} unknown".format(self.params["query type"]))

        if self.params["transitive closure"]:
            self.dataset.make_transitive_closure()
        if self.params["remove inconsistencies"]:
            self.dataset.remove_inconsistencies()

        return x_max

    def initialise_input_domain(self):
        ccs_size = self.params['ccs size']
        num_obj = self.params['num objectives']
        pcs_point_dist = self.params['pcs point dist']
        min_size = self.params['pcs min size']
        input_domain = utils_ccs.get_pcs_grid(ccs_size=ccs_size, num_objectives=num_obj, eucledian_dist=pcs_point_dist, min_size=min_size, seed=self.params['seed'])
        return input_domain

    def initialise_user(self):
        # create a user
        user = UserPreference(self.params['num objectives'], self.params['utility noise'], seed=self.params['seed'])
        # rescale on input domain
        user.rescale_on_input_domain(self.input_domain)
        return user

    def initialise_acquirer(self):
        return acquisition_function.DiscreteAcquirer(self.input_domain, query_type=self.params['query type'], seed=self.params['seed'], acquisition_type=self.params['acquisition function'])

    def initialise_gaussian_process(self):
        if self.params['gp prior mean'] == 'linear-zero':
            prior_mean_type = 'linear'
        else:
            prior_mean_type = self.params['gp prior mean']
        return GPPairwise(num_objectives=self.params['num objectives'], std_noise=self.params["gp noise hyperparameter"], kernel_width=self.params["gp kernel hyperparameter"], prior_mean_type=prior_mean_type, seed=self.params['seed'])

    def initialise_dataset(self):
        return DatasetPairwise(self.params['num objectives'])
