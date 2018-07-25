"""
@author: Luisa M Zintgraf (2017, Vrije Universiteit Brussel)

All the specifications we need for the experiment.
"""
from collections import OrderedDict

SEED = 666

# the minimum distance (in utility) between jobs
JOB_DIST_THRESH = 0.01

# the attributes a job can have, and the objectives we care about
obj_salary = 'Salary'
obj_wfh = 'Days home'
obj_prob = 'Probation'
OBJECTIVES = [obj_wfh, obj_salary, obj_prob]

NUM_OBJECTIVES = len(OBJECTIVES)

abbrev_wfh = 'WFH'
abbrev_salary = '$$$'
abbrev_prob = 'PRB'
OBJECTIVES_ABBREVIATIONS = [abbrev_wfh, abbrev_salary, abbrev_prob]

# the weights per objective
OBJ_WEIGHTS = {}
OBJ_WEIGHTS[obj_wfh] = 0.5
OBJ_WEIGHTS[obj_salary] = 0.35
OBJ_WEIGHTS[obj_prob] = 0.15
OBJ_WEIGHTS = OrderedDict({key: OBJ_WEIGHTS[key] for key in OBJECTIVES})

# min value that the objectives can take
OBJ_MIN = {}
OBJ_MIN[obj_salary] = 2000.
OBJ_MIN[obj_wfh] = 0.
OBJ_MIN[obj_prob] = 0.
OBJ_MIN = OrderedDict({key: OBJ_MIN[key] for key in OBJECTIVES})

# max values that the objectives can have
OBJ_MAX = {}
OBJ_MAX[obj_salary] = 5000.
OBJ_MAX[obj_wfh] = 5.
OBJ_MAX[obj_prob] = 12.
OBJ_MAX = OrderedDict({key: OBJ_MAX[key] for key in OBJECTIVES})

# max values that the objectives can have
OBJ_STEPSIZE = {}
OBJ_STEPSIZE[obj_salary] = 100.
OBJ_STEPSIZE[obj_wfh] = 0.5
OBJ_STEPSIZE[obj_prob] = 1.
OBJ_STEPSIZE = OrderedDict({key: OBJ_STEPSIZE[key] for key in OBJECTIVES})

# how much time a user has to make an experiment
# -> half a minute!
TIME_EXPERIMENT_SEC = 60

# with which jobs we start
START_JOB_INDICES = [[0, 1], [2, 3], [4, 5]]
