"""
@author: Luisa M Zintgraf (2017, Vrije Universiteit Brussel)
"""
import numpy as np
from collections import OrderedDict
import sys
sys.path.insert(0, '../')
from python_backend.user_jobs import TheoreticalUser
from python_backend import specs_jobs, utils_jobs


def get_random_job_matrix(min_num_jobs, seed, normalise=True):
    """ Returns a matrix (num_jobs x num_objectives) of random job offers """
    random_state = np.random.RandomState(seed)

    for j in range(10):

        num_rand_jobs = min_num_jobs*100

        # initialise job matrix
        job_matrix = np.zeros((num_rand_jobs, specs_jobs.NUM_OBJECTIVES))
        # for each objective, choose num_jobs many random (possible) values
        for obj_idx in range(specs_jobs.NUM_OBJECTIVES):
            key = specs_jobs.OBJECTIVES[obj_idx]
            possible_obj_vals = np.arange(specs_jobs.OBJ_MIN[key], specs_jobs.OBJ_MAX[key] + 0.001, specs_jobs.OBJ_STEPSIZE[key])
            job_matrix[:, obj_idx] = random_state.choice(possible_obj_vals, size=num_rand_jobs, replace=True)

        # remove the jobs that are too close together in terms of utility
        user = TheoreticalUser()
        job_utilities = user.utility(job_matrix)
        # keep the first job in the matrix
        job_matrix_reduced = job_matrix[0][np.newaxis, :]
        # keep adding jobs that are not too close to already existing jobs
        for job_idx in range(1, len(job_utilities)):
            job_dist = np.abs(job_utilities[job_idx] - user.utility(job_matrix_reduced))
            if np.sum(job_dist < specs_jobs.JOB_DIST_THRESH) == 0:
                job_matrix_reduced = np.vstack((job_matrix_reduced, job_matrix[job_idx]))
        job_matrix = job_matrix_reduced

        if job_matrix.shape[0] >= min_num_jobs:
            break

    if normalise:
        job_matrix = normalise_job_offer(job_matrix)

    print('Created {} jobs; rerun with different seed if more are  needed.'.format(job_matrix.shape[0]))
    return job_matrix


def get_min_job_offer(normalise=True):
    """ Worst possible job offer """
    job_offer = OrderedDict({key: specs_jobs.OBJ_MIN[key] for key in specs_jobs.OBJECTIVES})
    if normalise:
        for key in specs_jobs.OBJECTIVES:
            job_offer[key] = (job_offer[key] - specs_jobs.OBJ_MIN[key]) / (specs_jobs.OBJ_MAX[key] - specs_jobs.OBJ_MIN[key])
    return job_offer


def get_max_job_offer(normalise=True):
    """ Worst possible job offer """
    job_offer = OrderedDict({key: specs_jobs.OBJ_MAX[key] for key in specs_jobs.OBJECTIVES})
    if normalise:
        for key in specs_jobs.OBJECTIVES:
            job_offer[key] = (job_offer[key] - specs_jobs.OBJ_MIN[key]) / (specs_jobs.OBJ_MAX[key] - specs_jobs.OBJ_MIN[key])
    return job_offer


def normalise_job_offer(job_matrix):

    # if input is array, bring in right format
    if job_matrix.ndim == 1:
        job_matrix = job_matrix[np.newaxis, :]

    for obj_idx in range(specs_jobs.NUM_OBJECTIVES):
        key = specs_jobs.OBJECTIVES[obj_idx]
        job_matrix[:, obj_idx] = (job_matrix[:, obj_idx] - specs_jobs.OBJ_MIN[key]) / (specs_jobs.OBJ_MAX[key] - specs_jobs.OBJ_MIN[key])

    return job_matrix


def denormalise_job_offer(job_matrix):

    # if input is array, bring in right format
    if job_matrix.ndim == 1:
        job_matrix = job_matrix[np.newaxis, :]

    for obj_idx in range(specs_jobs.NUM_OBJECTIVES):
        key = specs_jobs.OBJECTIVES[obj_idx]
        job_matrix[:, obj_idx] = job_matrix[:, obj_idx] * (specs_jobs.OBJ_MAX[key] - specs_jobs.OBJ_MIN[key]) + specs_jobs.OBJ_MIN[key]

    return job_matrix


if __name__ == '__main__':

    jobs = get_random_job_matrix(min_num_jobs=50, seed=73, normalise=True)

    import utility_func_jobs
    job_utils = utility_func_jobs.utility(jobs, min_util=0, max_util=1)

    np.save(utils_jobs.PATH_JOBS, jobs)
