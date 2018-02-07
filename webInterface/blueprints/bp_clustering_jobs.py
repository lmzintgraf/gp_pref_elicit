"""
@author: Luisa M Zintgraf (2017, Vrije Universiteit Brussel)
"""
import numpy as np
from flask import render_template, request, redirect, url_for, Blueprint
import sys
import time
sys.path.insert(0, '../')
from python_backend import utils_jobs, specs_jobs, utils_users
sys.path.insert(0, '../../')
from gaussian_process import GPPairwise
from acquisition_function import DiscreteAcquirer

bp_clustering_jobs = Blueprint('bp_clustering_jobs', __name__)


@bp_clustering_jobs.route('/start_clustering_jobs/<username>', methods=['POST', 'GET'])
def start(username):

    # get the start time for this user
    _ = utils_users.get_experiment_start_time(username, 'clustering')

    # get the dataset for this user
    user_dataset = utils_users.get_gp_dataset(username, 'clustering', num_objectives=specs_jobs.NUM_OBJECTIVES)

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
        utils_users.update_gp_dataset(username, user_dataset, 'clustering')

        # convert into displayable format
        job1 = utils_jobs.job_array_to_job_dict(job1)
        job2 = utils_jobs.job_array_to_job_dict(job2)

        # add ID to the above dictionaries (equals the index in the dataset
        job1['ID'] = job1_idx
        job2['ID'] = job2_idx

        # put jobs we want to display in the respective lists
        jobs_unclustered = [job1, job2]
        top_job = []
        good_jobs = []
        bad_jobs = []

    # otherwise, we show the previous ranking and pick a new point accordingly
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
        utils_users.update_gp_dataset(username, user_dataset, 'clustering')

        # convert job to dictionary
        job_new = utils_jobs.job_array_to_job_dict(job_new)

        # add the ID
        job_new['ID'] = job_new_idx

        # put into list of jobs that need to be ranked
        jobs_unclustered = [job_new]

        # get clustering
        clustering = utils_users.get_clustering(username)

        clustering_new = []
        for cluster in clustering:
            cluster_new = []
            for idx in range(len(cluster)):
                job_idx = cluster[idx]
                new_job = utils_jobs.job_array_to_job_dict(user_dataset.datapoints[job_idx])
                new_job['ID'] = job_idx
                cluster_new.append(new_job)
            clustering_new.append(cluster_new)

        top_job = clustering_new[0]
        good_jobs = clustering_new[1]
        bad_jobs = clustering_new[2]

    return render_template('query_clustering_jobs.html', username=username, jobs_unclustered=jobs_unclustered,
                           top_job=top_job, good_jobs=good_jobs, bad_jobs=bad_jobs)


@bp_clustering_jobs.route('/submit_clustering_jobs', methods=['POST'])
def submit_clustering_jobs():

    # get the username and their dataset
    username = request.form['username']
    user_dataset = utils_users.get_gp_dataset(username, 'clustering', num_objectives=specs_jobs.NUM_OBJECTIVES)

    # get the clusters the user submitted
    top_job = [int(request.form['top-cluster'])]
    good_jobs = np.fromstring(request.form['good-cluster'], sep=',', dtype=np.int)
    bad_jobs = np.fromstring(request.form['bad-cluster'], sep=',', dtype=np.int)
    clustering = [top_job, good_jobs, bad_jobs]

    # save it
    utils_users.save_clustering(username, clustering)

    # get the actual jobs (ranking returned the indiced in the dataset)
    top_job = user_dataset.datapoints[top_job]
    good_jobs = user_dataset.datapoints[good_jobs]
    bad_jobs = user_dataset.datapoints[bad_jobs]
    clustering = [top_job, good_jobs, bad_jobs]

    # add the ranking to the dataset
    user_dataset.add_clustered_preferences(clustering)

    # save
    utils_users.update_gp_dataset(username, user_dataset, 'clustering')

    # get the start time for this user
    start_time = utils_users.get_experiment_start_time(username, 'clustering')

    if time.time()-start_time < specs_jobs.TIME_EXPERIMENT_SEC:
        return redirect(url_for('.start', username=username))
    else:
        utils_users.update_experiment_status(username=username, query_type='clustering')
        return redirect('start_experiment/{}'.format(username))
