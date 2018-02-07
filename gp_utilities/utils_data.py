"""
@author: Luisa M Zintgraf (2017, Vrije Universiteit Brussel)

Utilities for handling the data with our GPs.
The class DatasetPairwise can keep track of real and virtual observations.
"""
import numpy as np


def format_data(data, num_objectives):
    """
    Bring data into right format to be consistent
    :param data:
    :param num_objectives:
    :return data:           reformatted data matrix
    """
    data = np.array(data)
    if data.ndim == 0:
        data = np.zeros((1, num_objectives)) + data
    # if we get only one datapoint, make columns the objectives
    elif data.ndim == 1:
        if len(data) == num_objectives:
            data = data[np.newaxis, :]
        else:
            data = data[:, np.newaxis]
    # make sure the columns are the objectives
    elif data.shape[1] != num_objectives and data.shape[0] == num_objectives:
        data = data.T
    elif data.shape[1] == num_objectives:
        data = data
    else:
        raise RuntimeError('Data does not seem to have the right number of objectives.')

    return data


def array_in_matrix(arrays, matrix, rounding_accuracy=5):
    """
    Checks if the array a is in the (rows of) the matrix m
    :param arrays:
    :param matrix:
    :param rounding_accuracy:   we will round the values of the array
                                and the matrix until this many digits after
                                the comma; default is 10
    :return:
    """
    if len(matrix) == 0:
        return False
    arrays = np.array(arrays)
    if matrix.ndim == 1:
        matrix = matrix[np.newaxis, :]
    if arrays.ndim == 1:
        if matrix.shape[0] == 0:
            return 0
        array = np.round(arrays, rounding_accuracy)
        matrix = np.round(matrix, rounding_accuracy)
        return int(np.sum(np.sum(np.abs(matrix - array), axis=1) == 0))
    else:
        if matrix.shape[0] == 0:
            return np.zeros(arrays.shape[0])
        arrays = np.round(arrays, rounding_accuracy)
        matrix = np.round(matrix, rounding_accuracy)
        diff = np.abs(matrix[:, np.newaxis] - arrays)
        diff = np.sum(diff, axis=2)
        a_in_m = np.sum(diff == 0, axis=0) > 0
        return np.array(a_in_m, dtype=int)


def scale_to_unit_interval(x, y=None):
    """
    Scale the values in x to lie between 0 and 1;
    use min and max values from x if y is none;
    else use min and max values from y
    :param x:           values to transform to lie between 0 and 1
    :param y:           optional; if not None we use min(y) and max(y)
                        to rescale_on_ccs values in x
    :return x_scaled:   the values of x, scaled to the unit interval
    """
    min_val = np.min(x) if y is None else np.min(y)
    max_val = np.max(x) if y is None else np.max(y)

    if min_val != max_val:
        x_scaled = (x - min_val) / (max_val - min_val)
    # if all values in x have the same value and this is between 0 and 1, return x
    elif 1 > max_val > 0:
        x_scaled = x
    # if all values in x have the same value which is not between 0 and 1, return zeros
    else:
        x_scaled = np.zeros(x.shape)

    return x_scaled
