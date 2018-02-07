"""
@author: Luisa M Zintgraf (2017, Vrije Universiteit Brussel)
"""
import numpy as np
import time
from flask import Blueprint
from flask import render_template, request, redirect, url_for
import sys
sys.path.insert(0, '../')
from python_backend import specs_jobs, utils_jobs, data_jobs, utils_users
sys.path.insert(0, '../../')
from gaussian_process import GPPairwise
from acquisition_function import DiscreteAcquirer

bp_pairwise_jobs = Blueprint('bp_pairwise_jobs', __name__)


# --- pairwise experiment --

@bp_pairwise_jobs.route('/start_pairwise_jobs/<username>', methods=['POST', 'GET'])
def start(username):

    # register start time for this user
    utils_users.get_experiment_start_time(username, 'pairwise')

    # get the starting points from the acquirer
    job1, job2 = utils_jobs.get_next_start_jobs(username)

    # transform the two jobs into dictionaries
    job1 = utils_jobs.job_array_to_job_dict(job1)
    job2 = utils_jobs.job_array_to_job_dict(job2)

    return render_template("query_pairwise_jobs.html", username=username, job1=job1, job2=job2, side_clicked=-1)


@bp_pairwise_jobs.route('/choose_pairwise_jobs', methods=['POST'])
def choose_pairwise():

    # get current user
    username = request.form['username']

    # get the comparison
    winner_job = np.fromstring(request.form['winner'], sep=' ')
    loser_job = np.fromstring(request.form['loser'], sep=' ')

    # normalise jobs again
    winner_job = data_jobs.normalise_job_offer(winner_job)
    loser_job = data_jobs.normalise_job_offer(loser_job)

    # add comparison to user's data and save
    dataset_user = utils_users.get_gp_dataset(username, 'pairwise', num_objectives=specs_jobs.NUM_OBJECTIVES)
    dataset_user.add_single_comparison(winner_job, loser_job)
    utils_users.update_gp_dataset(username, dataset_user, 'pairwise')

    # find out whether user clicked left or right button (0=left, 1=right)
    job_clicked = request.form['job-clicked']

    # display new jobs
    return redirect(url_for('.continue_pairwise', username=username, side_clicked=job_clicked))


@bp_pairwise_jobs.route('/continue_pairwise_jobs/<username>_<side_clicked>', methods=['POST', 'GET'])
def continue_pairwise(username, side_clicked):

    # get the dataset for this user
    dataset_user = utils_users.get_gp_dataset(username, 'pairwise', num_objectives=specs_jobs.NUM_OBJECTIVES)

    # initialise the acquirer which picks new datapoints
    acquirer = DiscreteAcquirer(input_domain=utils_jobs.get_jobs(), query_type='pairwise', seed=specs_jobs.SEED)

    # intialise the GP
    gp = GPPairwise(num_objectives=specs_jobs.NUM_OBJECTIVES, seed=specs_jobs.SEED)

    # add collected datapoints to acquirer
    acquirer.history = dataset_user.datapoints
    # add collected datapoints to GP
    gp.update(dataset_user)

    # get the best job so far
    job_best_idx = dataset_user.comparisons[-1, 0]
    job_best = dataset_user.datapoints[job_best_idx]

    # let acquirer pick new point
    job_new = acquirer.get_next_point(gp, dataset_user)

    # sort according to what user did last round
    if side_clicked == "1":
        job1 = job_best
        job2 = job_new
    else:
        job1 = job_new
        job2 = job_best

    # transform the two jobs into dictionaries
    job1 = utils_jobs.job_array_to_job_dict(job1)
    job2 = utils_jobs.job_array_to_job_dict(job2)

    # get the start time for this user
    start_time = utils_users.get_experiment_start_time(username, 'pairwise')

    if time.time()-start_time < specs_jobs.TIME_EXPERIMENT_SEC:
        return render_template("query_pairwise_jobs.html", username=username, job1=job1, job2=job2,
                               side_clicked=side_clicked)
    else:
        utils_users.update_experiment_status(username=username, query_type='pairwise')
        return redirect('start_experiment/{}'.format(username))
