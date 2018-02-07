"""
@author: Luisa M Zintgraf (2017, Vrije Universiteit Brussel)
"""
import numpy as np
from flask import render_template, request, Blueprint
import sys
sys.path.insert(0, '../')
from python_backend import utils_users


bp_ranking_tutorial = Blueprint('bp_ranking_tutorial', __name__)

NUMBERS = [6, 2, 4, 1]


@bp_ranking_tutorial.route('/start_ranking_tutorial/<username>', methods=['POST', 'GET'])
def start(username):

    start_numbers = [NUMBERS[0], NUMBERS[1]]

    return render_template("query_ranking_tutorial.html", username=username, numbers_unranked=start_numbers,
                           numbers_ranked=[], number_counter=2)


@bp_ranking_tutorial.route('/submit_ranking_tutorial', methods=['POST'])
def submit_ranking():

    # get the username and their dataset
    username = request.form['username']
    number_counter = int(request.form['number_counter'])

    if number_counter < len(NUMBERS):

        # get the ranking the user submitted
        ranking = np.fromstring(request.form['rankingResult'], sep=',', dtype=np.int)

        new_number = NUMBERS[number_counter]

        number_counter += 1

        return render_template("query_ranking_tutorial.html", username=username, numbers_unranked=[new_number],
                               numbers_ranked=ranking, number_counter=number_counter)

    else:

        # update the tutorial stats of the user
        utils_users.update_tutorial_status('ranking', username)

        # get the user's status on the tutorials
        tutorial_status = utils_users.get_tutorial_status(username)

        return render_template("navi_tutorial.html", username=username, pairwise_fin=tutorial_status[0],
                               clustering_fin=tutorial_status[1], ranking_fin=tutorial_status[2])
