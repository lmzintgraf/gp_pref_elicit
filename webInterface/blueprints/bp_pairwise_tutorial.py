"""
@author: Luisa M Zintgraf (2017, Vrije Universiteit Brussel)
"""
from flask import Blueprint
from flask import render_template, request
import sys
sys.path.insert(0, '../')
from python_backend import utils_users


bp_pairwise_tutorial = Blueprint('bp_pairwise_tutorial', __name__)


NUMBERS = [5, 3, 18, 9, 34]


@bp_pairwise_tutorial.route('/start_pairwise_tutorial/<username>', methods=['POST', 'GET'])
def start(username):

    # get the two starting points for the user
    number1 = NUMBERS[0]
    number2 = NUMBERS[1]

    return render_template("query_pairwise_tutorial.html", username=username, number1=number1, number2=number2,
                           number_counter=2, side_clicked=-1)


@bp_pairwise_tutorial.route('/choose_pairwise_tutorial', methods=['POST'])
def choose_pairwise():

    # get current user
    username = request.form['username']

    # find out whether user clicked left or right button (0=left, 1=right)
    winner = request.form['winner']
    number_clicked = request.form['number_clicked']
    number_counter = int(request.form['number_counter'])

    if number_counter < len(NUMBERS):

        # sort according to what user did last round
        if number_clicked == "1":
            number1 = winner
            number2 = NUMBERS[number_counter]
        else:
            number1 = NUMBERS[number_counter]
            number2 = winner

        number_counter += 1

        return render_template("query_pairwise_tutorial.html", username=username, number1=number1, number2=number2,
                               number_counter=number_counter, number_clicked=number_clicked,
                               side_clicked=number_clicked)

    else:

        # update the tutorial stats of the user
        utils_users.update_tutorial_status('pairwise', username)

        # get the user's status on the tutorials
        tutorial_status = utils_users.get_tutorial_status(username)

        return render_template("navi_tutorial.html", username=username, pairwise_fin=tutorial_status[0],
                               clustering_fin=tutorial_status[1], ranking_fin=tutorial_status[2])
