"""
@author: Luisa M Zintgraf (2017, Vrije Universiteit Brussel)
"""
import numpy as np
from flask import render_template, request, redirect, url_for, Blueprint
import sys
import time
sys.path.insert(0, '../')
from python_backend import specs_jobs, utils_jobs, utils_users
sys.path.insert(0, '../../')
from gaussian_process import GPPairwise
from acquisition_function import DiscreteAcquirer

bp_ranking_jobs = Blueprint('bp_ranking_jobs', __name__)


@bp_ranking_jobs.route('/start_ranking_jobs/<username>', methods=['POST', 'GET'])
def start(username):

    # get the start time for this user
    utils_users.get_experiment_start_time(username, 'ranking')

    # get the dataset for this user
    user_dataset = utils_users.get_gp_dataset(username, 'ranking', num_objectives=specs_jobs.NUM_OBJECTIVES)

    # if no data has been collected yet, we only display two starting jobs
    if user_dataset.comparisons.shape[0] == 0:

        # delete any datapoint in the user's dataset (in case experiment was aborted)
        user_dataset.datapoints = np.empty((0, specs_jobs.NUM_OBJECTIVES))

        # get the starting points from the acquirer
        job1, job2 = utils_jobs.get_next_start_jobs(username)

        # add jobs to dataset of user
        job1_idx = user_dataset._add_single_datapoint(job1)
        job2_idx = user_dataset._add_single_datapoint(job2)

        # save dataset
        utils_users.update_gp_dataset(username, user_dataset, 'ranking')

        # convert into displayable format
        job1 = utils_jobs.job_array_to_job_dict(job1)
        job2 = utils_jobs.job_array_to_job_dict(job2)

        # add ID to the above dictionaries (equals the index in the dataset
        job1['ID'] = job1_idx
        job2['ID'] = job2_idx

        # put jobs we want to display in the respective lists
        jobs_unranked = [job1, job2]
        jobs_ranked = []

    # otherwise, we show the previous ranking and pick a new point according to that
    else:

        # intialise the GP
        gp = GPPairwise(num_objectives=specs_jobs.NUM_OBJECTIVES, seed=specs_jobs.SEED)

        # initialise acquirer
        acquirer = DiscreteAcquirer(input_domain=utils_jobs.get_jobs(), query_type='clustering',
                                    seed=specs_jobs.SEED)

        # add collected datapoints to acquirer
        acquirer.history = user_dataset.datapoints
        # add collected datapoints to GP
        gp.update(user_dataset)

        # let acquirer pick new point
        job_new = acquirer.get_next_point(gp, user_dataset)

        # add that point to the dataset and save
        job_new_idx = user_dataset._add_single_datapoint(job_new)
        utils_users.update_gp_dataset(username, user_dataset, 'ranking')

        # convert job to dictionary
        job_new = utils_jobs.job_array_to_job_dict(job_new)

        # add the ID
        job_new['ID'] = job_new_idx

        # put into list of jobs that need to be ranked
        jobs_unranked = [job_new]

        # get ranking so far
        ranking = utils_users.get_ranking(username)
        # get the job information from that ranking and convert to dictionaries
        jobs_ranked = user_dataset.datapoints[ranking]
        jobs_ranked = [utils_jobs.job_array_to_job_dict(job) for job in jobs_ranked]
        # add the IDs
        for i in range(len(ranking)):
            jobs_ranked[i]['ID'] = ranking[i]

    return render_template("query_ranking_jobs.html", username=username, jobs_unranked=jobs_unranked,
                           jobs_ranked=jobs_ranked)


@bp_ranking_jobs.route('/submit_ranking_jobs', methods=['POST'])
def submit_ranking():

    # get the username and their dataset
    username = request.form['username']
    user_dataset = utils_users.get_gp_dataset(username, 'ranking', num_objectives=specs_jobs.NUM_OBJECTIVES)

    # get the ranking the user submitted
    ranking = np.fromstring(request.form['rankingResult'], sep=',', dtype=np.int)
    # save it
    utils_users.save_ranking(username, ranking)

    # get the actual jobs (ranking returned the indiced in the dataset)
    jobs_ranked = user_dataset.datapoints[ranking]

    # add the ranking to the dataset
    user_dataset.add_ranked_preferences(jobs_ranked)

    # save
    utils_users.update_gp_dataset(username, user_dataset, 'ranking')

    # get the start time for this user
    start_time = utils_users.get_experiment_start_time(username, 'ranking')

    if time.time()-start_time < specs_jobs.TIME_EXPERIMENT_SEC:
        return redirect(url_for('.start', username=username))
    else:
        utils_users.update_experiment_status(username=username, query_type='ranking')
        return redirect('start_experiment/{}'.format(username))
