"""
@author: Luisa M Zintgraf (2017, Vrije Universiteit Brussel)

This is a virtual user, for which we "know" (well, define) the utility function.
We use this in our computer-based experiments.
"""

import numpy as np
from numpy.random import RandomState
from scipy import special
from sklearn.cluster import KMeans
import sys
sys.path.insert(0, '..')
from gp_utilities import utils_data, utils_ccs


class UserPreference:
    def __init__(self, num_objectives, std_noise, seed=None):
        """
        Class to simulate a user with specific preferences;
        can rank/clustering data according to the intrinsic preference function.
        Functions are defined on [0,1] -> [0,1].
        :param num_objectives:
        :param std_noise:
        :param random_state:
        """

        self.num_objectives = num_objectives
        self.std_noise = std_noise
        self.random_state = RandomState(seed)

        # default case: x in [0,1]^d, y in [0, 1]
        self.min_x, self.min_y = (np.zeros(self.num_objectives), 0)
        self.max_x, self.max_y = (np.ones(self.num_objectives), 1)

        self.utility_func = self._initialise_utility_function()

    def _initialise_utility_function(self):

        if self.num_objectives == 1:

            self.num_preference_funcs = self.random_state.choice(range(1, 3), 1)[0]

            # the intrinsic preference function
            num_zeros = self.random_state.choice(range(2, 16), 1)[0]
            z = np.linspace(-0.05 * num_zeros, 1 + 0.05 * num_zeros, num_zeros)
            # shift the zero's of the polynomial a bit
            noise_max = self.random_state.uniform(0.01, 0.05, 1)[0]
            z += self.random_state.normal(0, noise_max, num_zeros)
            self.poly = np.poly(z)

            # the min/max values of the intrinsic preference function (needed for scaling)
            x_linsp = np.linspace(0, 1, 500)[:, np.newaxis]
            y_linsp = np.polyval(self.poly, x_linsp)
            self.min_y = np.min(y_linsp)
            self.max_y = np.max(y_linsp)

            utl_func = lambda x: np.polyval(self.poly, x)

        else:

            # for each objective, randomly select a monotonic function
            funcs_1d = _mon_utilities_1d(self.random_state)
            funcs_1d = self.random_state.choice(funcs_1d, self.num_objectives)

            # weights for the individual preference functions
            # (if num_obj > 2 this reflects how much the user cares about the objectives)
            weight_preference_funcs = self.random_state.uniform(0.2, 0.8, len(funcs_1d))
            weight_preference_funcs /= np.sum(weight_preference_funcs)

            def utl_func(x):

                y = 0.
                for d in range(len(funcs_1d)):
                    y += weight_preference_funcs[d] * funcs_1d[d](x[:, d % self.num_objectives])

                return y

        return utl_func

    def rescale_on_input_domain(self, input_domain):

        max_at_border = True
        while max_at_border:

            # get the true utility on the input domain
            true_utility = self.get_preference(input_domain, add_noise=False)

            # test if the maximum is at one of the extremas
            argmax_input = input_domain[np.argmax(true_utility)]
            if np.min(argmax_input) < 0.01 or np.max(argmax_input) > 0.99:
                self.utility_func = self._initialise_utility_function()
            else:
                max_at_border = False

        # reset min/max values in user preferences; will be used to scale the utility values between 0 and 1
        self.min_y = np.min(true_utility)
        self.max_y = np.max(true_utility)

    def get_preference(self, x, add_noise):

        # bring data into right format first
        x = utils_data.format_data(x, self.num_objectives)

        # EVALUATE user preferences
        if self.num_objectives == 1:
            y = np.polyval(self.poly, x)
        else:
            y = self.utility_func(x)
        y = y.flatten()
        assert len(y) == x.shape[0]
        y = (y - self.min_y) / (self.max_y - self.min_y)

        if add_noise:
            # the noise is different every time the scalarisation function is called,
            # but we always draw from the same distribution
            noise = self.random_state.normal(0, self.std_noise, x.shape[0])
            y += noise

        return y

    def pairwise_comparison(self, x1, x2, add_noise):
        """
        Returns boolean indicating whether x1 wins against x2.
        :param x1:  first datapoint
        :param x2:  second datapoint
        :return:    boolean indicating if f(x1)>f(x2)
        """
        x = np.vstack((x1, x2))
        y = self.get_preference(x, add_noise=add_noise)
        return int(y[0] > y[1])

    def ranking(self, x, add_noise):

        # get the true utility
        utility = self.get_preference(x, add_noise)

        # rank
        x = x[np.argsort(-utility)]

        return x

    def clustering(self, x, num_clusters, add_noise, include_winner):
        # scalarise with noise and sort x and y according to those values
        y = self.get_preference(x, add_noise=add_noise).flatten()
        sorted_indices = np.argsort(-y)
        y_sorted = y[sorted_indices]
        x_sorted = x[sorted_indices]

        # initialise a list for the clusters
        clusters_x = []
        clusters_y = []

        if include_winner:
            clusters_x.append(x_sorted[0][np.newaxis, :])
            clusters_y.append(y_sorted[0])
            x_sorted = x_sorted[1:, :]
            y_sorted = y_sorted[1:]

        if num_clusters is None or len(y_sorted) < num_clusters:
            num_clusters = len(y_sorted)

        # clustering the rest using k-means
        kmeans = KMeans(n_clusters=num_clusters).fit(y_sorted[:, np.newaxis])

        for i in range(num_clusters):
            clusters_x.append(x_sorted[kmeans.labels_ == i])
            clusters_y.append(np.mean(y_sorted[kmeans.labels_ == i]))
            # note: if y ever takes the max value, this will be the winner
            # so we will not exclude anything in the groups by keeping
            # the "<" here (assuming that there can only be one value in
            # y that is the maximum)
        # sort list so that the best points come first
        clusters_x = [clusters_x[j] for j in np.argsort(-np.array(clusters_y))]

        # make sure we didn't forget any of the data points
        assert np.sum([c.shape[0] for c in clusters_x]) == x.shape[0]

        return clusters_x

    def top_rank(self, x, n_rank, add_noise):

        # get the true utility of the input
        utility = self.get_preference(x, add_noise)

        # get the top N
        top_n = x[np.argsort(-utility)[:n_rank]]

        # the rest
        rest = x[np.argsort(-utility)[n_rank:]]

        return [top_n, rest]

    def best_worst(self, x, add_noise):

        # get the true utility of the input
        utility = self.get_preference(x, add_noise)

        # get the top N
        ranking = x[np.argsort(-utility)]

        return [ranking[0], ranking[1:-1], ranking[-1]]


def _mon_utilities_1d(random_state):
    """
    All the monotonic increasing scalarisations in 1 dimension we use
    :return:
    """

    sigmoids = []
    for _ in range(10):
        a = random_state.uniform(10, 50, 1)[0]
        b = random_state.uniform(1, 20, 1)[0]
        n = random_state.randint(1, 10, 1)[0]
        sigmoids.append(lambda x: _stacked_sigmoids(x, a, b, n))

    polys = []
    for _ in range(1):
        a = random_state.uniform(1, 5, 1)[0]
        polys.append(lambda x: _poly(x, a))

    # return np.concatenate((sigmoids, polys))

    return np.concatenate(([_sigmoid_left_1d, _sigmoid_middle_1d, _sigmoid_right_1d], sigmoids))


def _sigmoid_left_1d(x):
    min_y = 1. / (1 + np.exp(-0 * 25 + 5))
    max_y = 1. / (1 + np.exp(-1 * 25 + 5))
    y = 1. / (1 + np.exp(-x * 25 + 5))
    return (y-min_y)/(max_y-min_y)


def _sigmoid_middle_1d(x):
    min_y = 1. / (1 + np.exp(-0 * 20 + 10))
    max_y = 1. / (1 + np.exp(-1 * 20 + 10))
    y = 1. / (1 + np.exp(-x * 20 + 10))
    return (y-min_y)/(max_y-min_y)


def _sigmoid_right_1d(x):
    min_y = 1. / (1 + np.exp(-0 * 17 + 13))
    max_y = 1. / (1 + np.exp(-1 * 17 + 13))
    y = 1. / (1 + np.exp(-x * 17 + 13))
    return (y-min_y)/(max_y-min_y)


def _stacked_sigmoids(x, a, b, n):
    y = 0
    y_min = 0
    y_max = 0
    for i in range(0, 5*n, 5):
        y += 1. / (1 + np.exp(- x * (a-i) + (b+i)))
        y_min += 1. / (1 + np.exp(- 0 * (a-i) + (b+i)))
        y_max += 1. / (1 + np.exp(- 1 * (a-i) + (b+i)))
    return (y - y_min) / (y_max - y_min)


def _poly(x, a):
    return ((x*a-1)**3 + 1) / ((1*a-1)**3 + 1)


def _non_mon_utilities_1d():
    """
    All the monotonic increasing scalarisations in 1 dimension we use
    :return:
    """
    return [
        _squared_1d,
        _squared_flipped_1d,
        _beta_left_1d,
        _beta_right_1d,
        _double_beta_1d,
        _wiggly_beta_1d,
        _poly8_1d
    ]


def _beta_left_1d(x):
    a = 2
    b = 10
    y = (x**(a-1)*(1-x)**(b-1))
    y /= special.beta(a, b)
    y /= 4.2
    return y


def _beta_right_1d(x):
    a = 10
    b = 3
    y = (x**(a-1)*(1-x)**(b-1))
    y /= special.beta(a, b)
    y /= 3.5
    return y


def _double_beta_1d(x):
    return 0.5*_beta_left_1d(x) + _beta_right_1d(x)


def _wiggly_beta_1d(x):
    return 0.5*_beta_left_1d(x) + _beta_right_1d(x+0.1)


def _poly8_1d(x):
    x = np.copy(x)
    x *= 7.7
    x -= 4.35
    y = 1/8*x**8 + 4/7*x**7 - 7/3*x**6 - 56/5*x**5 + 49/4*x**4 + 196/3*x**3 - 18*x**2 - 144*x
    y += 200
    y /= 400
    return 1-y


def _squared_1d(x):
    return (0.25 - (x - 0.5) ** 2) * 4


def _squared_flipped_1d(x):
    return 0.5*(1 - (0.25 - (x - 0.7) ** 2) * 4)
