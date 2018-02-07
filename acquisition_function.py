"""
@author: Luisa M Zintgraf (2017, Vrije Universiteit Brussel)
"""
import copy
import numpy as np
from scipy.stats import norm
import sys
sys.path.insert(0, '..')
from gp_utilities.utils_data import array_in_matrix


class DiscreteAcquirer:
    def __init__(self, input_domain, query_type, seed, acquisition_type='expected improvement'):
        """
        An acquirer for a discrete set of points, using the expected improvement.
        :param input_domain:     (np.array) the datapoints on which the discrete acquirer is defined.
        :param query_type:       (str) the query type of the current experiment (pairwise/ranking/clustering/top_rank)
        :param seed:             (int) random seed
        :param acquisition_type: (str) type of acquisition function, can be "expected improvement" or "thompson sampling"
        """
        self.input_domain = copy.deepcopy(input_domain)
        self.query_type = query_type
        self.acq_type = acquisition_type
        self.random_state = np.random.RandomState(seed)
        self.history = np.empty((0, self.input_domain.shape[1]))

    def get_start_points(self, gaussian_process):
        """
        Get datapoints which should be queried first
        :param gaussian_process:
        :return:
        """
        # get the expected improvement of the whole input domain (in batches, might be slow otherwise)
        exp_impr = np.zeros(self.input_domain.shape[0])
        batch_size = 64
        for curr_idx in range(0, self.input_domain.shape[0]+batch_size, batch_size):
            exp_impr[curr_idx:curr_idx+batch_size] = get_expected_improvement(self.input_domain[curr_idx:curr_idx+batch_size], gaussian_process, self.history)

        # get the best expected improvement things
        best_points = self.input_domain[exp_impr == np.max(exp_impr)]
        if len(best_points) > 1:
            indices = self.random_state.choice(range(len(best_points)), 2, replace=False)
            datapoint1 = best_points[indices[0]]
            datapoint2 = best_points[indices[1]]
        else:
            datapoint1 = best_points[0]
            second_best_points = self.input_domain[exp_impr == np.sort(exp_impr)[-2]]
            idx = self.random_state.choice(range(len(second_best_points)), replace=False)
            datapoint2 = second_best_points[idx]

        self.history = np.vstack((self.history, datapoint1, datapoint2))
        return datapoint1, datapoint2

    def get_next_point(self, gaussian_process, dataset):
        """
        Get the next datapoint to query
        :param gaussian_process:
        :param dataset:
        :return:
        """
        # points which can't be queried next
        exclude = exclude_points(self.query_type, dataset)

        if self.acq_type == 'expected improvement':
            next_point = self.get_next_point_EI(gaussian_process, exclude)
        elif self.acq_type == 'thompson sampling':
            next_point = self.get_next_point_thompson(gaussian_process, exclude)
        else:
            raise RuntimeError()

        # append to history
        self.history = np.vstack((self.history, next_point))

        return next_point

    def get_next_point_thompson(self, gaussian_process, exclude):
        """
        Returns the next datapoint to query, using thompson sampling
        :param gaussian_process:
        :param exclude:
        :return:
        """
        # get a sample
        sample_mean = gaussian_process.sample(self.input_domain)

        next_point = self.input_domain[np.argmax(sample_mean)]
        next_point_idx = 1
        while array_in_matrix(next_point, exclude):
            if next_point_idx >= self.input_domain.shape[0]:
                "Run out of points to display next. You should end the experiment."
                break
            next_point = self.input_domain[np.argsort(-sample_mean)[next_point_idx]]
            next_point_idx += 1

        return next_point

    def get_next_point_EI(self, gaussian_process, exclude):
        """
        Returns the next datapoint to query, using expected improvement
        :param gaussian_process:
        :param exclude:
        :return:
        """
        # get the expected improvement of the whole input domain (in batches, might be slow otherwise)
        expected_improvement = np.zeros(self.input_domain.shape[0])
        batch_size = 64
        for curr_idx in range(0, self.input_domain.shape[0]+batch_size, batch_size):
            expected_improvement[curr_idx:curr_idx+batch_size] = get_expected_improvement(self.input_domain[curr_idx:curr_idx+batch_size], gaussian_process, self.history)

        # find the point with the highest EI, and which can be queried next
        next_point = self.input_domain[np.argmax(expected_improvement)]
        next_point_idx = 1
        while array_in_matrix(next_point, exclude):
            if next_point_idx >= self.input_domain.shape[0]:
                "Run out of points to display next. You should end the experiment."
                break
            next_point = self.input_domain[np.argsort(-expected_improvement)[next_point_idx]]
            next_point_idx += 1

        return next_point


def get_expected_improvement(datapoints, gaussian_process, datapoints_hist, xi=0.01):
    """
    Calculate the expected improvement of datapoints given a GP
    :param datapoints:          the datapoints for which we want to get the EI
    :param gaussian_process:    the gaussian process model
    :param datapoints_hist:     datapoints we have already queried
    :param xi:                  hyperparameter
    :return:    expected improvement for given datapoints
    """

    # initialise the expected improvement vector
    exp_impr = np.zeros(datapoints.shape[0])

    # predicted mean and variance at datapoints
    pred_mean, pred_var = gaussian_process.get_predictive_params(datapoints, pointwise=True)

    # get the value of the point (from our history) with the highest predicted mean
    max_f = 0
    if datapoints_hist.shape[0] > 0:
        max_f = np.max(gaussian_process.get_predictive_params(datapoints_hist, pointwise=True)[0])

    mfxi = pred_mean[pred_var != 0] - max_f - xi
    z = mfxi / pred_var[pred_var != 0]
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)

    # if the variance is zero, the EI is defined as zero
    exp_impr[pred_var != 0] = mfxi * cdf_z + pred_var[pred_var != 0] * pdf_z

    return exp_impr


def get_probability_of_improvement(x, gaussian_process, x_previous, xi=0.01):
    """
    Get the probability of improvement for a single point
    :param x:                   the point we are evaluating
    :param gaussian_process:    the current gaussian process
    :param x_previous:          the points we have evaulated thus far
    :param xi:                  noise parameter
    :return:
    """

    # the predicted mean and variance of the point we are evaluating
    pred_mean, pred_var = gaussian_process.get_predictive_params(x, pointwise=True)

    # get the value of the point (from our history) with the highest predicted mean
    max_f = 0
    if x_previous.shape[0] > 0:
        max_f = np.max(gaussian_process.get_predictive_params(x_previous, pointwise=True)[0])

    z = (pred_mean - max_f - xi) / pred_var

    prob_improv = norm.cdf(z)

    return prob_improv


def exclude_points(query_type, dataset):
    if query_type == 'pairwise':
        return exclude_points_pairwise(dataset)
    elif query_type == 'ranking' or query_type == 'clustering' or query_type == 'top_rank' or query_type == 'best_worst':
        return exclude_points_ranking(dataset)
    else:
        raise NotImplementedError("Query type {} unknown.".format(query_type))


def exclude_points_pairwise(dataset):
    """
    When we do pairwise comparisons, we exclude all jobs the current max
    was already compared to
    :param dataset:
    :return:
    """
    job_max = dataset.datapoints[dataset.comparisons[-1, 0]]
    exclude = job_max[np.newaxis, :]
    job_max_idx = dataset.get_index(job_max)
    for comp in dataset.comparisons:
        if job_max_idx in list(comp):
            new_idx = comp[1 - list(comp).index(job_max_idx)]
            exclude = np.vstack((exclude, dataset.datapoints[new_idx]))
    return exclude


def exclude_points_ranking(dataset):
    """
    When we do ranking, none of the existing jobs should be added again
    :param dataset:
    :return:
    """
    return dataset.datapoints


class RandomAcquirer:
    def __init__(self, input_domain, seed=None):
        self.input_domain = input_domain
        self.random_state = np.random.RandomState(seed)
        self.history = np.empty((0, self.input_domain.shape[1]))

    def get_start_points(self):
        return self.get_next_point(), self.get_next_point()

    def get_next_point(self, *args):
        idx = self.random_state.randint(0, self.input_domain.shape[0], 1)[0]
        next_point = self.input_domain[idx]
        self.history = np.vstack((self.history, next_point))
        return next_point
