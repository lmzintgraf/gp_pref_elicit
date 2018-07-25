"""
Start script for running the survey webpage.

@author: Luisa M Zintgraf (2017, Vrije Universiteit Brussel)
"""
import re
import argparse
import numpy as np
from os import environ
from flask import Flask, render_template, request, redirect, url_for
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')
from python_backend import utils_jobs, utils_users
# import the blueprints for the tutorials
from blueprints.bp_pairwise_tutorial import bp_pairwise_tutorial
from blueprints.bp_ranking_tutorial import bp_ranking_tutorial
from blueprints.bp_clustering_tutorial import bp_clustering_tutorial
# import the blueprints for the experiment
from blueprints.bp_pairwise_jobs import bp_pairwise_jobs
from blueprints.bp_ranking_jobs import bp_ranking_jobs
from blueprints.bp_clustering_jobs import bp_clustering_jobs

# create application
app = Flask(__name__)

# register the blueprints for the tutorial
app.register_blueprint(bp_pairwise_tutorial)
app.register_blueprint(bp_ranking_tutorial)
app.register_blueprint(bp_clustering_tutorial)

# register the blueprints for the different experiments
app.register_blueprint(bp_pairwise_jobs)
app.register_blueprint(bp_ranking_jobs)
app.register_blueprint(bp_clustering_jobs)

# default settings for running the script locally
default_host = '0.0.0.0'
default_port = environ.get("PORT", 5000)

# set up the command-line options
parser = argparse.ArgumentParser()
parser.add_argument("-H", "--host",
                    help="Hostname of the Flask app " +
                         "[default %s]" % default_host,
                    default=default_host)
parser.add_argument("-P", "--port",
                    help="Port for the Flask app " +
                         "[default %s]" % default_port,
                    default=default_port)
parser.add_argument("-t", "--skip_tutorial",
                    help="Skip tutorial " +
                         "[default False]",
                    action="store_true")
parser.add_argument("-d", "--debug",
                    help="Debug mode " +
                         "[default False]",
                    action="store_true")

args = parser.parse_args()


@app.route("/")
def main():
    return render_template('index.html')


# -- start page  --

@app.route('/register_user', methods=['POST'])
def register_user():
    username = request.form['username']

    # if the input is not OK, set username to None and return
    if not re.match('^\w+$', username):
        return render_template('index.html', username='invalid')

    # check if user exists already and completed the survey
    user_status = utils_users.get_user_status(username)
    if len(user_status['survey results']) > 0:
        return render_template('index.html', username='existing')

    utils_users.register_user(username)

    if args.skip_tutorial:
        return render_template("persona_jobs.html", username=username)
    else:
        return render_template("persona_tutorial.html", username=username)

# -- tutorial --


@app.route('/persona_tutorial', methods=['POST'])
def persona_tutorial():

    # get the username
    username = request.form['username']

    # go to the persona description page (using the username)
    return render_template("persona_tutorial.html", username=username)


@app.route('/start_tutorial', methods=['POST'])
def start_tutorial():

    # get the username
    username = request.form['username']

    # get the user's status on the tutorials
    tutorial_status = utils_users.get_tutorial_status(username)

    # go to the persona description page (using the username)
    return render_template("navi_tutorial.html", username=username, pairwise_fin=tutorial_status[0],
                           clustering_fin=tutorial_status[1], ranking_fin=tutorial_status[2])


@app.route('/finish_tutorial', methods=['POST'])
def finish_tutorial():

    # get the username
    username = request.form['username']

    return render_template('index.html', username=username, tutorial_completed=True)


# -- experiment --

@app.route('/reroute_to_experiment', methods=['POST'])
def reroute_to_experiment():
    # get the username
    username = request.form['username']
    # go to the persona description page (using the username)
    return render_template("persona_jobs.html", username=username)


@app.route('/start_experiment/<username>')
def start_experiment(username):

    # decide which query type the user has to do now (pairwise, clustering, ranking)
    query_type = utils_users.get_next_experiment_type(username)

    if query_type is not None:
        return render_template("navi_experiment_jobs.html", username=username, query_type=query_type)
    else:
        return redirect('start_survey/{}'.format(username))


# -- survey --

@app.route('/start_survey/<username>')
def start_survey(username):

    # get the resulting jobs
    user_status = utils_users.get_user_status(username)

    win_idx_pair = user_status['dataset pairwise'].comparisons[-1, 0]
    win_pair = user_status['dataset pairwise'].datapoints[win_idx_pair]
    win_pair = utils_jobs.job_array_to_job_dict(win_pair)
    win_pair['ID'] = 'pairwise'

    win_idx_clust = user_status['logs clustering'][-1][0][0]
    win_clust = user_status['dataset clustering'].datapoints[win_idx_clust]
    win_clust = utils_jobs.job_array_to_job_dict(win_clust)
    win_clust['ID'] = 'clustering'

    win_idx_rank = user_status['logs ranking'][-1][0]
    win_rank = user_status['dataset ranking'].datapoints[win_idx_rank]
    win_rank = utils_jobs.job_array_to_job_dict(win_rank)
    win_rank['ID'] = 'ranking'

    # shuffle the results
    winners = [win_pair, win_clust, win_rank]
    winners = np.random.permutation(winners)

    return render_template('survey.html', username=username, winners=winners)


@app.route('/submit_survey', methods=['POST'])
def submit_survey():

    # get the user name
    username = request.form['username']

    survey_result = {
        'outcome order': request.form['outcome-ranking'],
        'understanding': [request.form['understand-pairwise'], request.form['understand-ranking'],
                          request.form['understand-clustering']],
        'preference': request.form['preference-ranking'],
        'effort': [request.form['pairwise-effort'], request.form['ranking-effort'], request.form['clustering-effort']],
        'distraction': request.form['distracted'],
        'comment': request.form['comment']
    }

    utils_users.save_survey_result(username, survey_result)

    return render_template('the_end.html', username=username)


# --- run ---

app.run(
    debug=args.debug,
    host=args.host,
    port=int(args.port)
)
