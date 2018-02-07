"""
@author: Luisa M Zintgraf (2017, Vrije Universiteit Brussel)
"""
import os
import pickle as pkl
import numpy as np
import sys
import time
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
from dataset import DatasetPairwise


# -- data paths --

ROOT = os.path.dirname(__file__)

DIR_STORAGE = os.path.join(ROOT, 'storage')

PATH_USERS = os.path.join(DIR_STORAGE, 'users.p')

QUERY_TYPES = ['pairwise', 'clustering', 'ranking']


def create_paths():

    if not os.path.isdir(DIR_STORAGE):
        os.mkdir(DIR_STORAGE)

    if not os.path.exists(PATH_USERS):
        users = {}
        pkl.dump(users, open(PATH_USERS, 'wb'))
create_paths()


# --- users ---

def register_user(username):

    # the values say whether the user did the tutorials/experiments
    user_status = {'name': username,  # same as key; but makes some things easier
                   'tutorial pairwise': 0,  # which tutorials the user already completed
                   'tutorial clustering': 0,
                   'tutorial ranking': 0,
                   'order experiment': np.random.permutation(['pairwise', 'ranking', 'clustering']),
                   'order start jobs': np.random.permutation([0, 1, 2]),
                   'experiment pairwise': 0,  # which experiments the user already completed
                   'experiment clustering': 0,
                   'experiment ranking': 0,
                   'start time pairwise': None,  # start time of the experiment
                   'start time clustering': None,
                   'start time ranking': None,
                   'dataset pairwise': None,  # this is the current dataset which is passed to the GP
                   'dataset clustering': None,
                   'dataset ranking': None,
                   'logs pairwise': [],  # here we log what the user answers at each query
                   'logs clustering': [],
                   'logs ranking': [],
                   'survey results': []  # here we store the answers of the user to the survey
                   }
    save_user_status(user_status)


def get_all_users():
    return pkl.load(open(PATH_USERS, 'rb'))


def get_user_status(username):
    all_users = get_all_users()
    if username not in all_users.keys():
        register_user(username)
    all_users = get_all_users()
    user_status = all_users[username]
    return user_status


def save_user_status(user_status):
    # get the user database
    all_users = get_all_users()
    username = user_status['name']
    all_users[username] = user_status
    pkl.dump(all_users, open(PATH_USERS, 'wb'))


# -- experiment status of users --

def get_next_experiment_type(username):
    # get the user's stats
    user_status = get_user_status(username)
    experiment_index = np.sum([user_status[k] for k in ['experiment pairwise',
                                                        'experiment clustering',
                                                        'experiment ranking',
                                                        ]])
    if experiment_index < len(user_status['order experiment']):
        experiment_type = user_status['order experiment'][experiment_index]
    else:
        experiment_type = None
    return experiment_type


def get_experiment_start_time(username, query_type):
    start_time = get_user_status(username)['start time {}'.format(query_type)]
    if start_time is None:
        start_time = time.time()
        user_status = get_user_status(username)
        user_status['start time {}'.format(query_type)] = start_time
        save_user_status(user_status)

    return start_time


def save_experiment_end_time(username, query_type):
    end_time = time.time()
    user_status = get_user_status(username)
    user_status['end time {}'.format(query_type)] = end_time
    save_user_status(user_status)


def update_experiment_status(query_type, username):
    # get the stats
    user_status = get_user_status(username)
    # update
    user_status['experiment {}'.format(query_type)] = 1
    # save
    save_user_status(user_status)


# -- tutorial status of users --

def get_tutorial_status(username):
    user_status = get_user_status(username)
    tutorial_status = [user_status['tutorial {}'.format(q)] for q in QUERY_TYPES]
    return tutorial_status


def update_tutorial_status(query_type, username):
    # get the user status
    user_status = get_user_status(username)
    # update status of user
    user_status['tutorial {}'.format(query_type)] = 1
    # save
    save_user_status(user_status)


# --- handling the collected data ---

def get_gp_dataset(username, query_type, num_objectives):
    user_status = get_user_status(username)
    gp_dataset = user_status['dataset {}'.format(query_type)]
    if gp_dataset is None:
        gp_dataset = DatasetPairwise(num_objectives)
    return gp_dataset


def update_gp_dataset(username, dataset, query_type):
    user_status = get_user_status(username)
    user_status['dataset {}'.format(query_type)] = dataset
    save_user_status(user_status)


# --- ranking ---
# log the current ranking that the user gave in the last query

def save_ranking(username, ranking):
    user_status = get_user_status(username)
    user_status['logs ranking'].append(ranking)
    save_user_status(user_status)


def get_ranking(username):
    user_status = get_user_status(username)
    curr_ranking_status = user_status['logs ranking'][-1]
    return curr_ranking_status


# --- clustering ---
# log the current clustering that the user gave in the last query

def save_clustering(username, clustering):
    user_status = get_user_status(username)
    user_status['logs clustering'].append(clustering)
    save_user_status(user_status)


def get_clustering(username):
    user_status = get_user_status(username)
    curr_cluster_status = user_status['logs clustering'][-1]
    return curr_cluster_status


# -- survey results --

def save_survey_result(username, survey_result):
    user_status = get_user_status(username)
    user_status['survey results'] = survey_result
    save_user_status(user_status)
