"""
@author: Luisa M Zintgraf (2017, Vrije Universiteit Brussel)
"""
import os
from collections import OrderedDict
import numpy as np
import sys
sys.path.insert(0, '../')
from python_backend import specs_jobs, data_jobs, utils_users
sys.path.insert(0, '../../')


# -- data paths --

ROOT = os.path.dirname(__file__)

DIR_STORAGE = os.path.join(ROOT, 'storage')

PATH_JOBS = os.path.join(DIR_STORAGE, 'job_offers')


def create_paths():

    if not os.path.isdir(DIR_STORAGE):
        os.mkdir(DIR_STORAGE)


# -- handling the jobs that are offered --

def get_jobs():
    if not os.path.exists(PATH_JOBS + '.npy'):
        job_offers = data_jobs.get_random_job_matrix(min_num_jobs=100, seed=specs_jobs.SEED, normalise=True)
        np.save(PATH_JOBS, job_offers)
    else:
        job_offers = np.load(PATH_JOBS + '.npy')
    return job_offers


def get_next_start_jobs(username):
    # get the user's stats
    user_status = utils_users.get_user_status(username)
    experiment_index = np.sum([user_status[k] for k in ['experiment pairwise',
                                                        'experiment clustering',
                                                        'experiment ranking',
                                                        ]])
    if experiment_index < 3:
        job_idx = user_status['order start jobs'][experiment_index]
        start_job_indices = specs_jobs.START_JOB_INDICES[job_idx]
        start_jobs = get_jobs()[start_job_indices]
    else:
        start_jobs = None
    return start_jobs


# --- handle jobs and how they are displayed---

def job_array_to_job_dict(job_array):
    """
    Converts a job array to a dictionary with unnormalised values
    which can be displayed to human readersg
    :param job_array:
    :return:
    """
    # denormalise the job values
    job_mat = data_jobs.denormalise_job_offer(job_array)[0]

    # put job into dictionary with the objectives as keys
    job_dict = OrderedDict({})
    for i in range(specs_jobs.NUM_OBJECTIVES):
        key = specs_jobs.OBJECTIVES[i]
        val = job_mat[i]
        job_dict[key] = val

    # job_dict = OrderedDict({specs_webapp.OBJECTIVES[key_idx]: job_mat[key_idx]
    # for key_idx in range(specs_webapp.NUM_OBJECTIVES})

    return job_dict
