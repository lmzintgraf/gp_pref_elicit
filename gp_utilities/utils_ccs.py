"""
@author: Luisa M Zintgraf (2017, Vrije Universiteit Brussel)
"""
import os
import numpy as np
from scipy.spatial import ConvexHull
import scipy
from pymodem import value, pruners

ROOT = os.path.dirname(__file__)

FOLDER_CCS_CACHE = os.path.join(ROOT, '../experiments/ccs_cache')
if not os.path.isdir(FOLDER_CCS_CACHE):
    os.makedirs(FOLDER_CCS_CACHE)

FOLDER_PCS_GRID_CACHE = os.path.join(ROOT, '../experiments/pcs_grid_cache')
if not os.path.isdir(FOLDER_PCS_GRID_CACHE):
    os.makedirs(FOLDER_PCS_GRID_CACHE)


def get_pcs_grid(ccs_size, num_objectives, eucledian_dist=0.05, min_size=100, seed=None):

    pcs_grid = np.empty((0, num_objectives))
    random_state = np.random.RandomState(seed)
    while pcs_grid.shape[0] < min_size:

        # get a CCS
        ccs = get_ccs(num_objectives, ccs_size, seed)

        # get a pcs grid from that ccs
        pcs_grid = get_pcs_grid_from_ccs(ccs, eucledian_dist, seed)

        seed += random_state.randint(0, 10000, 1)[0]

        if pcs_grid is None:
            pcs_grid = np.empty((0, num_objectives))

    return pcs_grid


def get_pcs_grid_from_ccs(ccs, max_dist=0.01, seed=None):

    file_path = os.path.join(FOLDER_PCS_GRID_CACHE, 'ccs-size{}_num-obj{}_seed{}'.format(ccs.shape[0], ccs.shape[1], seed))

    if os.path.exists(file_path+'.npy'):
        return np.load(file_path+'.npy')

    else:

        simplices = compute_ccs_simplices(ccs)
        distances_ok = [False for _ in range(len(simplices))]

        pcs_grid = ccs

        # loop through the simplices
        for s_idx in range(len(simplices)):

            # we slowly increase the number of points so the grid is just as dense as it has to be
            for num_points in range(ccs.shape[1], ccs.shape[1]*100):

                # check if the distances for the current simplex were ok already
                if distances_ok[s_idx]:
                    break

                else:
                    simplex = simplices[s_idx]

                    # get a grid on the simplex, including the vertices
                    simplex_grid_full = get_grid_on_simplex(num_points, simplex, include_vertices=True)
                    # get the pairwise distances between the points
                    distances = scipy.spatial.distance.cdist(simplex_grid_full, simplex_grid_full, 'chebyshev')
                    # get distance to closest point
                    np.fill_diagonal(distances, np.inf)
                    distances = np.min(distances, axis=1)

                    # if they're not too far apart, add the points to the overall pcs grid
                    if np.max(distances) <= max_dist:
                        simplex_grid = get_grid_on_simplex(num_points, simplex, include_vertices=False)
                        pcs_grid = np.vstack((pcs_grid, simplex_grid))
                        distances_ok[s_idx] = True

        # if the above loop succeeded (i.e., we didn't break out of it), we save the result
        np.save(file_path, pcs_grid)
        return pcs_grid


def get_ccs(num_objectives, ccs_size, seed=None):
    """
    Returns a random CCS with the given number of objectives.
    Each objective can have values between 0 and 1.
    :param num_objectives:
    :param ccs_size:
    :param seed:
    :return:
    """

    file_path = os.path.join(FOLDER_CCS_CACHE, 'ccs-size{}_num-obj{}_seed{}'.format(ccs_size, num_objectives, seed))

    if os.path.exists(file_path + '.npy'):
        return np.load(file_path + '.npy')

    else:

        # initialise random state with given seed
        random_state = np.random.RandomState(seed)

        # initialise empty ccs
        ccs = np.empty((0, num_objectives))

        # create random CCS's until we have the desired size
        while ccs.shape[0] < ccs_size:

            # generate vectors on the unit sphere in (num_objectives) dimensions
            ccs = np.meshgrid(*[np.linspace(0, 1, num_objectives * ccs_size)[:, np.newaxis] for _ in range(num_objectives - 1)])
            ccs = np.array(ccs).reshape((num_objectives - 1, -1)).T
            ccs = np.hstack((ccs, 1 - np.sum(ccs, axis=1, keepdims=True)))
            ccs = ccs[np.sum(np.abs(ccs), axis=1) <= 1]
            # make unit vector
            ccs /= np.sqrt(np.sum(ccs ** 2, axis=1))[:, np.newaxis]

            # add a little bit of noise
            noise = random_state.normal(0, 0.05, ccs.shape) * (0.5 - np.std(ccs, axis=1)[:, np.newaxis])
            ccs += noise

            # remove vectors where we added too much noise
            ccs = ccs[np.sum(ccs > 1, axis=1) == 0]
            ccs = ccs[np.sum(ccs < 0, axis=1) == 0]

            if ccs.shape[0] == 0:
                continue

            # take a random subset
            ccs = ccs[random_state.choice(range(ccs.shape[0]), np.min([ccs.shape[0], ccs_size * num_objectives * 2]), replace=False)]

            # prune dataset with new datapoint
            vv_ccs = value.ValueVectorSet()
            vv_ccs.addAll(ccs)
            ccs = np.array(pruners.c_prune(vv_ccs).set)

        # randomly select datapoints from the constructed ccs
        ccs = ccs[random_state.choice(range(ccs.shape[0]), ccs_size, replace=False)]

        np.save(file_path, ccs)

        return ccs


def compute_ccs_simplices(ccs):
    """
    Given datapoints (in a convex coverage set), compute the simplices that span this ccs
    :param datapoints:  datapoints in the ccs
    :return simplices:  list of simplices; one simplex is a matrix with the vertices of the simplex
    """

    num_datapoints = ccs.shape[0]
    num_objectives = ccs.shape[1]

    # if the number of objectives is 2, we can just use one of the
    # objectives to sort the vectors and then always take two successive vectors
    if num_objectives == 2:
        sorted_indices = np.argsort(ccs[:, 0])
        simplices = [ccs[[sorted_indices[i], sorted_indices[i + 1]]] for i in range(num_datapoints - 1)]

    else:
        # add the zero vector (otherwise it closes the CH on the bottom)
        datapoints = np.vstack((np.zeros(num_objectives), ccs))

        simplices = ConvexHull(points=ccs).simplices

        # remove the facets that have the zero-vector in them
        simplices = [datapoints[s, :] for s in simplices if 0 not in s]

    return simplices


def get_grid_on_simplex(num_grid_points, simplex, include_vertices=True):
    """
    Get an evenly distributed grid of mixture policies on the simplex
    defined by the vectors in simplex
    :param num_grid_points:     number of points we (maximally) want on the grid
    :param simplex:             matrix of size (num_datapoints)x(num_objectives)
    :param include_vertices:    whether or not to include the vertices
                                (i.e., points that define the simplex)
    :return:
    """
    num_simplex_vertices = simplex.shape[0]

    # create weights for the mixtures of datapoints in the simplex
    w = np.meshgrid(*[np.linspace(0, 1, num_grid_points)[:, np.newaxis] for _ in range(num_simplex_vertices-1)])
    w = np.array(w).reshape((num_simplex_vertices-1, -1)).T
    # add the last weight so that all weights sum up to one
    w = np.hstack((1 - np.sum(w, axis=1, keepdims=True), w))
    # only keep the weights that add up to one
    w = w[np.sum(w, axis=1) == 1]
    # only keep the weights that are positive everywhere
    w = w[np.sum(w < 0, axis=1) == 0]

    # if we don't want to include the vertices we remove rows with weights 0 or 1
    if not include_vertices:
        w = w[np.sum(w == 0, axis=1) == 0]
        w = w[np.sum(w == 1, axis=1) == 0]

    assert np.sum(np.sum(w, axis=1) != 1) == 0

    # make the mixture vectors with the weights we just created
    grid = np.dot(w, simplex)

    return grid
