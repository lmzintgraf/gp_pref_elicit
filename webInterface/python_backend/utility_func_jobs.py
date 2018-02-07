"""
@author: Luisa M Zintgraf (2017, Vrije Universiteit Brussel)
"""
import numpy as np
import sys
sys.path.insert(0, '../')
from python_backend import specs_jobs


# TODO: I hardcoded the min/max values; this only works with the current jobs
def utility(job_mat, min_util=0.09222382, max_util=0.95987308):

    # if input is array, bring in right format
    if job_mat.ndim == 1:
        job_mat = job_mat[np.newaxis, :]

    # initialise utility vector (for each job)
    utility_vect = np.zeros(job_mat.shape[0])

    # go through the objectives and sum weighted utilities
    for obj_idx in range(specs_jobs.NUM_OBJECTIVES):
        obj = specs_jobs.OBJECTIVES[obj_idx]
        utility_vect += specs_jobs.OBJ_WEIGHTS[obj] * utility_functions[obj](job_mat[:, obj_idx])

    # normalise
    utility_vect = (utility_vect - min_util) / (max_util - min_util)

    return utility_vect


def utility_wfh(wfh):
    """
    Utility over work-from-home: the person needs to work at least two days a week from home;
    the rest of the time she would prefer to work in the office
    :param wfh:
    :return:
    """
    wfh = np.array(wfh)
    utility = np.zeros(wfh.shape)
    cutoff = 2./5
    utility[wfh < cutoff] = wfh[wfh < cutoff]
    utility[wfh >= cutoff] = 1
    return utility


# utility over salary
def utility_salary(salary):
    return 2 * (1./(1+np.exp(-4 * salary)) - 0.5)


# utility over probation_time
def utility_probation_time(probation_time):
    return (1 - probation_time) ** 2


utility_functions = {
        specs_jobs.obj_salary: utility_salary,
        specs_jobs.obj_wfh: utility_wfh,
        specs_jobs.obj_prob: utility_probation_time,
    }


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    x = np.linspace(0, 1, 100)
    plt.figure()

    plt.subplot(1, 3, 1)
    x_wfh = x * (specs_jobs.OBJ_MAX[specs_jobs.obj_wfh] - specs_jobs.OBJ_MIN[specs_jobs.obj_wfh]) + specs_jobs.OBJ_MIN[specs_jobs.obj_wfh]
    plt.plot(x_wfh, utility_wfh(x))

    plt.subplot(1, 3, 2)
    x_salary = x * (specs_jobs.OBJ_MAX[specs_jobs.obj_salary] - specs_jobs.OBJ_MIN[specs_jobs.obj_salary]) + specs_jobs.OBJ_MIN[specs_jobs.obj_salary]
    plt.plot(x_salary, utility_salary(x))

    plt.subplot(1, 3, 3)
    x_prob = x * (specs_jobs.OBJ_MAX[specs_jobs.obj_prob] - specs_jobs.OBJ_MIN[specs_jobs.obj_prob]) + specs_jobs.OBJ_MIN[specs_jobs.obj_prob]
    plt.plot(x_prob, utility_probation_time(x))
    plt.show()
