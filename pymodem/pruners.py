#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 16:28:51 2016

@author: Diederik M. Roijers (University of Oxford)
"""

from .value import ValueVectorSet
from . import value
import operator
from functools import reduce
from pulp import LpVariable, LpProblem, LpMaximize, value
import numpy as np


def weak_pareto_dominates(vec1, vec2):
    """
    Returns whether vec1 weakly dominates vec2
    """
    for i in range(len(vec1)):
        if vec1[i] < vec2[i]:
            return False
    return True


def remove_weak_dominated_by(vector, vv_list):
    """
    Returns a new list of vectors which contains all vectors from vv_set that
    are not weak Pareto-dominated by 'vector'.
    """
    result = []
    for i in range(len(vv_list)):
        if not weak_pareto_dominates(vector, vv_list[i]):
            result.append(vv_list[i])
    return result


def pareto_prune(vv_set):
    """
    Returns a new ValueVectorSet from which all Pareto-dominated vectors
    have been removed.
    For Pseudo-code, see e.g., PPrune (Algorithm 2, page 34, Chapter 3) from
        Diederik M. Roijers - Multi-Objective Decision-Theoretic Planning,
        PhD Thesis, University of Amsterdam, 2016.
    """
    result = ValueVectorSet()
    V = vv_set.set
    while len(V):
        current_non_dominated = V[0]
        for i in range(len(V)):
            if weak_pareto_dominates(V[i], current_non_dominated):
                current_non_dominated = V[i]
        result.add(current_non_dominated)
        V = remove_weak_dominated_by(current_non_dominated, V)
    return result


def find_weight(v, vv_set):
    """
    Finds the weight where the value of a new value vector V improves most
    upon a given set of value vectors VVS.
    Find weight returns a tuple: weight, improvement
    For Pseudo-code, see e.g., find_weight (Algorithm 3,Chapter 3) from
        Diederik M. Roijers - Multi-Objective Decision-Theoretic Planning,
        PhD Thesis, University of Amsterdam, 2016.
    """
    x = LpVariable('x')
    w_vars = []
    problem = LpProblem('maximization', LpMaximize)
    for i in range(v.shape[0]):
        w_i = LpVariable('w_' + str(i), 0, 1)
        w_vars.append(w_i)
    for vector in vv_set.set:
        diff_list = []
        for index, _ in enumerate(vector):
            diff = v[index] - vector[index]
            diff_list.append(w_vars[index] * diff)
        problem += reduce(operator.add, diff_list, -x) >= 0
    problem += reduce(operator.add, w_vars, 0) == 1
    problem += x #What to optimise, i.e., x
    status = problem.solve()
    if value(x) <= 0:
        if(len(vv_set.set)==0):
            #Special case: in this case x is not in the problem and 
            #any solution in the weight simplex goes. Therefore, x retained
            #its initial value of 0
            return np.array([value(w) for w in w_vars]), True
        return [], False
    else:
        return np.array([value(w) for w in w_vars]), True


def c_prune(vv_set):
    """
    Returns a new ValueVectorSet from which all C-dominated vectors
    have been removed. (CCS(vv_set))
    For Pseudo-code, see e.g., CPrune (Algorithm 1, page 34, Chapter 3) from
        Diederik M. Roijers - Multi-Objective Decision-Theoretic Planning,
        PhD Thesis, University of Amsterdam, 2016.
    """
    vprime = pareto_prune(vv_set)
    ccs = ValueVectorSet()
    # We build up the CCS incrementally by identifying
    # weights where our intermediate CCS is not yet perfect
    # and then adding the best vectors for those weights
    while not vprime.empty():
        v = vprime.set[0]
        tup = find_weight(v, ccs)
        # vpp = ValueVectorSet()
        if tup[1]:
            # find the best vector for the weight (tup[0])
            # where v improves upon the CCS
            # move this best vector from vprime to ccs
            tup2 = vprime.removeMaximisingLinearScalarisedValue(tup[0])
            vprime = tup2[1]
            ccs.add(tup2[0])
        else:
            # remove v from vprime if it isn't better anywhere
            vprime = vprime.removeVec(v)
    return ccs


def max_prune(vv_set, w):
    """
    Returns the maximising vector for a given weightvector w
      argmax_(vec in V)  w . vec
    """
    result = ValueVectorSet()
    V = vv_set.set
    # current = -math.inf
    current = -float("inf")
    current_vec = None
    for i in range(len(V)):
        vec = V[i]
        scal_value = value.inner_product(w, vec)
        if scal_value > current:
            current_vec = vec
            current = scal_value
    result.add(current_vec)
    return result
