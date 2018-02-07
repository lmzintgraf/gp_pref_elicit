"""
@author: Luisa M Zintgraf (2017, Vrije Universiteit Brussel)
"""
import numpy as np
import sys
sys.path.insert(0, '../')
from python_backend import specs_jobs, utility_func_jobs
sys.path.insert(0, '../../')
from gp_utilities.utils_user import UserPreference


class TheoreticalUser(UserPreference):
    def __init__(self):
        self.std_noise = 0.1
        self.utility_functions = self.initialise_utility_functions()
        super().__init__(len(self.utility_functions), self.std_noise)

    # overall utility function
    def utility(self, job_matrix):
        if job_matrix.ndim == 1:
            job_matrix = job_matrix[np.newaxis, :]
        objective_weights = specs_jobs.OBJ_WEIGHTS
        utility = np.zeros(job_matrix.shape[0])
        for objective in specs_jobs.OBJECTIVES:
            obj_vals = job_matrix[:, specs_jobs.OBJECTIVES.index(objective)]
            utility += objective_weights[objective] * self.utility_functions[objective](obj_vals)
        return utility

    @staticmethod
    def initialise_utility_functions():
        return {
            specs_jobs.obj_salary: utility_func_jobs.utility_salary,
            specs_jobs.obj_wfh: utility_func_jobs.utility_wfh,
            specs_jobs.obj_prob: utility_func_jobs.utility_probation_time,
        }
