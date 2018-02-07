"""
@author: Luisa M Zintgraf (2017, Vrije Universiteit Brussel)
"""
import numpy as np
import sys
sys.path.insert(0, '..')
from gp_utilities import utils_data as utl_data


class DatasetPairwise:
    def __init__(self, num_objectives):
        """
        Initialise an empty dataset for data with the given number of objectives.
        :param num_objectives:   dimension of the data
        """
        self.num_objectives = num_objectives
        self.datapoints = np.empty((0, self.num_objectives))
        self.comparisons = np.empty((0, 2), dtype=np.int)

    def add_single_comparison(self, winner, loser):
        """
        Add a single comparison to the dataset.
        :param winner:  datapoint that wins the comparison
        :param loser:   datapoint that loses the comparison
        :return:
        """

        # only add the comparison if we don't compare the same datapoint
        if np.array_equal(winner, loser):
            print("Watch out - trying to add comparison between same datapoints!")
            return False

        # add winner and loser to our datapoints and get the indices in dataset
        winner_idx = self._add_single_datapoint(winner)
        loser_idx = self._add_single_datapoint(loser)

        # add comparison
        if not utl_data.array_in_matrix([winner_idx, loser_idx], self.comparisons):
            self.comparisons = np.vstack((self.comparisons, [winner_idx, loser_idx]))

        return True

    def add_ranked_preferences(self, ranking):
        for job_idx in range(len(ranking)-1):
            winner = ranking[job_idx]
            loser = ranking[job_idx+1]
            self.add_single_comparison(winner, loser)

    def add_clustered_preferences(self, clusters, keep_prev_info=True):
        """
        Add clustered preferences.
        :param clusters:        a list of clusters
                                each cluster is a list of arrays
        :param keep_prev_info:  whether to re-set the dataset
        :return:
        """

        # remove empty clusters
        clusters = list(filter((lambda x: len(x) > 0), clusters))

        if not keep_prev_info:
            self.datapoints = np.empty((0, self.num_objectives))
            self.comparisons = np.empty((0, 2), dtype=np.int)

        for cluster_idx in range(len(clusters)-1):
            clust1 = clusters[cluster_idx]
            clust2 = clusters[cluster_idx+1]
            for winner in clust1:
                for loser in clust2:
                    self.add_single_comparison(winner, loser)

    def add_top_rank_preferences(self, top_rank, keep_prev_info=True):

        clusters = []
        for t in top_rank[0]:
            clusters.append([t])
        clusters.append(top_rank[1])

        self.add_clustered_preferences(clusters, keep_prev_info)

    def add_best_worst_preferences(self, best_worst, keep_prev_info=True):

        if len(best_worst[1]) == 0:
            clusters = [[best_worst[0]], [best_worst[2]]]
        else:
            clusters = [[best_worst[0]], best_worst[1], [best_worst[2]]]

        self.add_clustered_preferences(clusters, keep_prev_info)

    def _add_single_datapoint(self, new_datapoint):
        """
        Add single datapoint to the existing dataset if it doesn't exist yet and return index
        :param new_datapoint:       new datapoint to add
        :return x_new_idx:  the index of the new datapoint in the existing dataset
        """
        new_datapoint = utl_data.format_data(new_datapoint, self.num_objectives)

        # if the datapoint is not in our dataset yet, add it
        if not utl_data.array_in_matrix(new_datapoint, self.datapoints):
            self.datapoints = np.vstack((self.datapoints, new_datapoint))
            new_datapoint_index = self.datapoints.shape[0] - 1

        # if the datapoint is already in our dataset, find its index
        else:
            new_datapoint_index = self.get_index(new_datapoint)

        return new_datapoint_index

    def get_index(self, datapoint):
        """
        Gets the index of a datapoint in the dataset,
        returns None if that doesn't exist
        :param datapoint:   single datapoint
        :return:            None if datapoint is not in the dataset,
                            else the index of datapoint in the dataset
        """
        if not utl_data.array_in_matrix(datapoint, self.datapoints):
            return None
        else:
            return np.argmax(np.sum(datapoint != self.datapoints, axis=1) == 0)

    def make_transitive_closure(self):
        """
        Given the existing comparisons, make a transitive closure, i.e., if
        a > b and b > c is in the dataset, we add a > c

        Note that this method can be quite slow!
        """
        for winner_idx, loser_idx in self.comparisons:
            self._recursive_transitive_closure(winner_idx, loser_idx)

    def _recursive_transitive_closure(self, winner_idx, loser_idx):
        next_layer_losers_indices = self.comparisons[self.comparisons[:, 0] == loser_idx, 1]
        for nll_idx in next_layer_losers_indices:
            # RECURSION WHOOHOOO
            winner = self.datapoints[winner_idx]
            loser = self.datapoints[nll_idx]
            self.add_single_comparison(winner, loser)
            self._recursive_transitive_closure(winner_idx, nll_idx)

    def remove_inconsistencies(self):
        """
        If there are inconsistencies in the comparisons, remove them.
        :return:
        """
        comparisons_flipped = self.comparisons[:, [1, 0]]
        duplicate = utl_data.array_in_matrix(comparisons_flipped, self.comparisons)
        self.comparisons = self.comparisons[duplicate == 0, :]

    def add_mon_info_grid(self, n_per_dim, distance=None):
        """
        Add virtual comparisons on a grid on the entire input space
        :param n_per_dim:   number of points on the grid we want per dimension
        :param distance:    distance between points on the grid;
                            if None it will be the distance to the closest point
        :return:
        """
        min_vals = np.zeros(self.num_objectives)
        max_vals = np.ones(self.num_objectives)

        # put meshgrid on the entire space
        grid = np.meshgrid(*[np.linspace(min_vals[i], max_vals[i], num=n_per_dim) for i in range(self.num_objectives)])
        grid = np.vstack([grid[i].flatten() for i in range(self.num_objectives)]).T

        if distance is None:
            distance = (max_vals - min_vals) / (n_per_dim - 1.)
        else:
            distance = np.ones(self.num_objectives) * distance

        # for each point in the mesh...
        for dp in grid:
            # ...in each dimension...
            for d in range(self.num_objectives):
                dp_comp = np.copy(dp)

                # ...add one dominating point...
                dp_comp[d] += distance[d]
                if dp_comp[d] <= max_vals[d]:
                    self.add_single_comparison(dp_comp, dp)

                # ...and one dominated point...
                dp_comp[d] -= 2 * distance[d]
                if dp_comp[d] >= min_vals[d]:
                    self.add_single_comparison(dp, dp_comp)

    def add_mon_info_pcs(self, ccs_grid, distance=0.01):
        """
        Add virtual comparisons between points on the PCS and points around it
        :param ccs:         a convex coverage set, see class CCS from utils_ccs.py
        :param num_p:       number of points for which to add comparisons
                            per simplex that exists on the ccs
        :param distance:    distance between the points we compare
        :param min_vals:       min values that datapoints can take in each dimension;
                            if scalar is given we use that for every dimension
        :param max_vals:       maximal values that datapoints can take
        :return:
        """
        for x in ccs_grid:

            # the distance to other datapoints may depend on the current simplex
            if distance is not None:
                distance_pos = distance * np.ones(self.num_objectives)
                distance_neg = distance * np.ones(self.num_objectives)
            else:
                # find the distance to the next point in the positive direction
                distance_pos = ccs_grid - x
                distance_pos[distance_pos <= 0] = np.inf
                distance_pos = np.min(distance_pos, axis=0)
                # find the distance to the next point in the negative direction
                distance_neg = x - ccs_grid
                distance_neg[distance_neg <= 0] = np.inf
                distance_neg = np.min(distance_neg, axis=0)

            for d in range(self.num_objectives):
                x_comp = np.copy(x)
                x_comp[d] += distance_pos[d]
                if x_comp[d] <= 1:
                    self.add_single_comparison(x_comp, x)
                x_comp = np.copy(x)
                x_comp[d] -= distance_neg[d]
                if x_comp[d] >= 0:
                    self.add_single_comparison(x, x_comp)
