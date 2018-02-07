"""
@author: Luisa M Zintgraf (2017, Vrije Universiteit Brussel)

Helper classes for managing parameters.

Ok yes I could've done this nicer and easier.
There were some unforeseen extensions.
"""
import re


def get_parameter_dict(query_type=None, utility_noise=None, num_objectives=None):
    """ Initialise experiment settings with default values """
    return {
        # query type can be 'pairs' or 'clusters'
        "query type": 'pairwise' if query_type is None else query_type,
        "num queries": 50,
        "seed": 666,

        # - settings for the gaussian process -
        "gp prior mean": "zero",  # zero / linear / linear-zero
        "gp kernel hyperparameter": 0.15,
        "gp noise hyperparameter": 0.01,

        # - settings for the acquisition function -
        "acquisition function": "expected improvement",

        # - settings for the utility function -
        "utility noise": 0.05 if utility_noise is None else utility_noise,
        "num objectives": 2 if num_objectives is None else num_objectives,

        # settings for the CCS
        "ccs size": 5 if num_objectives is None else (num_objectives-1)*5,
        "pcs point dist": 0.01 if num_objectives is None else 0.01*num_objectives,
        "pcs min size": 50 if num_objectives is None else (num_objectives-1) * 50,

        # whether to use reference points (zero and one vector)
        'reference max': False,
        'reference min': False,

        # whether to do transitive closure of entire dataset
        "transitive closure": False,
        # whether to remove inconsistencies in dataset
        "remove inconsistencies": False,
        # whether to keep info from previous queries (important in e.g. ranking)
        "keep previous info": True,

        # - settings important only for clustering -
        "num clusters": 2,  # None means full ranking
        "winner from": 1,
        "headstart clusters": None,  # None: we start with 2 random items

        # - settings for using virtual pairwise comparisons
        "VC grid": False,
        "VC grid begin": False,
        "num VC grid": 3,
        "dist VC grid": None,

        "VC pcs": False,
        "VC pcs begin": False,
        "num VC pcs": 4,
        "dist VC pcs": None,
    }


def get_filename(parameters):
    """
    :param parameters:
    :return:
    """

    filename = ''

    # add the query type
    filename += parameters["query type"]

    # details about this run
    filename += "_quer" + str(parameters["num queries"])
    filename += "_seed" + str(parameters["seed"])

    # settings for gaussian process
    filename += "_" + parameters["gp prior mean"] + "-prior"
    filename += "_theta" + re.sub(r'[^\w]', '', str(parameters["gp kernel hyperparameter"]))
    filename += "_sigma" + re.sub(r'[^\w]', '', str(parameters["gp noise hyperparameter"]))

    # settings for acquisition function
    filename += "_" + parameters["acquisition function"] + "-acq"

    # settings for the utility function
    filename += "_util-noise" + re.sub(r'[^\w]', '', str(parameters["utility noise"]))
    filename += "_obj" + str(parameters["num objectives"])

    if parameters["num objectives"] > 1:
        filename += "-ccs" + str(parameters["ccs size"])
        filename += "_" + str(parameters['pcs min size'])
        filename += "pcs_grid" + re.sub(r'[^\w]', '', str(parameters["pcs point dist"]))

    if parameters['reference max'] == 'full':
        filename += '_refMax'
    elif parameters['reference max'] == 'beginning':
        filename += '_refMaxBegin'
    if parameters['reference min'] == 'full':
        filename += '_refMin'
    elif parameters['reference min'] == 'beginning':
        filename += '_refMinBegin'

    filename += "_trans" + str(parameters["transitive closure"])
    filename += "_remove" + str(parameters["remove inconsistencies"])
    filename += "_keep" + str(parameters["keep previous info"])

    # settings for clustering
    if parameters["query type"] == "clustering":
        filename += "-clust" + str(parameters["num clusters"])
        filename += "-win" + str(parameters["winner from"])
        filename += "-headstart" + str(parameters["headstart clusters"])

    # settings for using virtual points
    if parameters["num objectives"] > 1:

        # virtual comparisons
        if parameters["VC grid"]:
            filename += "_VC-grid" + str(parameters["num VC grid"]) + re.sub(r'[^\w]', '', str(parameters["dist VC grid"]))
        if parameters["VC pcs"]:
            filename += "_VC-pcs" + str(parameters["num VC pcs"]) + re.sub(r'[^\w]', '', str(parameters["dist VC pcs"]))
        if parameters["VC grid begin"]:
            filename += "_VC-grid-begin" + str(parameters["num VC grid"]) + re.sub(r'[^\w]', '', str(parameters["dist VC grid"]))
        if parameters["VC pcs begin"]:
            filename += "_VC-pcs-begin" + str(parameters["num VC pcs"]) + re.sub(r'[^\w]', '', str(parameters["dist VC pcs"]))

    return filename
