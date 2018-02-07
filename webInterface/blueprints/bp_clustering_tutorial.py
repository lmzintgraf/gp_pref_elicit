"""
@author: Luisa M Zintgraf (2017, Vrije Universiteit Brussel)
"""
import numpy as np
from flask import render_template, request, Blueprint
import sys
sys.path.insert(0, '../')
from python_backend import utils_users


bp_clustering_tutorial = Blueprint('bp_clustering_tutorial', __name__)


NUMBERS = [1, 9, 2, 25, -5, 6]


@bp_clustering_tutorial.route('/start_clustering_tutorial/<username>', methods=['POST', 'GET'])
def start(username):

    start_numbers = [NUMBERS[0], NUMBERS[1]]

    return render_template('query_clustering_tutorial.html', username=username, numbers_unranked=start_numbers,
                           top_number=[], good_numbers=[], bad_numbers=[], number_counter=2)


@bp_clustering_tutorial.route('/submit_clustering_tutorial', methods=['POST'])
def submit_clustering_tutorial():

    # get the username and their dataset
    username = request.form['username']
    number_counter = int(request.form['number_counter'])

    if number_counter < len(NUMBERS):

        # get the clusters the user submitted
        top_number = [int(request.form['top-cluster'])]
        good_numbers = np.fromstring(request.form['good-cluster'], sep=',', dtype=int)
        bad_numbers = np.fromstring(request.form['bad-cluster'], sep=',', dtype=int)

        new_number = NUMBERS[number_counter]

        number_counter += 1

        # save
        return render_template('query_clustering_tutorial.html', username=username, numbers_unranked=[new_number],
                               top_number=top_number, good_numbers=good_numbers, bad_numbers=bad_numbers,
                               number_counter=number_counter)

    else:

        # update the tutorial stats of the user
        utils_users.update_tutorial_status('clustering', username)

        # get the user's status on the tutorials
        tutorial_status = utils_users.get_tutorial_status(username)

        return render_template("navi_tutorial.html", username=username, pairwise_fin=tutorial_status[0],
                               clustering_fin=tutorial_status[1], ranking_fin=tutorial_status[2])
