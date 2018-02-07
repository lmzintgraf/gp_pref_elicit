#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 15:51:03 2016

@author: Diederik M. Roijers (University of Oxford)
"""

import numpy


def equalVectors(vector1, vector2):
    """
        Returns whether two vectors are equal in all elements
    """
    for i in range(len(vector1)):
        if not vector1[i] == vector2[i]:
            return False
    return True


def inner_product(vector1, vector2):
    """
        Returns vector1 . vector2
    """
    result = 0
    for i in range(len(vector1)):
        result = result + vector1[i]*vector2[i]
    return result


class ValueVectorSet:
    def __init__(self):
        self.set = []

    def add(self, vector):
        self.set.append(vector)

    def addAll(self, vectors):
        """
        Add all vectors in in 'vectors' to the set
        :param vectors:     A list of vectors
        """
        for i in range(len(vectors)):
            self.set.append(vectors[i])

    def empty(self):
        """
        """
        return len(self.set) == 0

    def translate(self, vector):
        """
        Return a new set identical to this set, but translated with 'vector'
        """
        newSet = []
        for i in range(len(self.set)):
            newSet.append((self.set[i]+vector))
        result = ValueVectorSet()
        result.addAll(newSet)
        return result

    def multiplyByScalar(self, scalar):
        """
        Return a new set identical to this set, but translated with 'vector'
        """
        newSet = []
        for i in range(len(self.set)):
            newSet.append((scalar*self.set[i]))
        result = ValueVectorSet()
        result.addAll(newSet)
        return result

    def crossSum(self, vvset):
        """
        Return a new set being the cross sum of this set and another set
        """
        if(len(self.set) == 0):
            result = ValueVectorSet()
            result.addAll(vvset.set)
            return result
        if(len(vvset.set) == 0):
            result = ValueVectorSet()
            result.addAll(self.set)
            return result
        newSet = []
        for i in range(len(self.set)):
            for j in range(len(vvset.set)):
                newSet.append((self.set[i]+vvset.set[j]))
        result = ValueVectorSet()
        result.addAll(newSet)
        return result

    def __str__(self):
        """
        """
        result = ""
        for i in range(len(self.set)):
            result = result + str(i) + ": " + str(self.set[i]) + "\n"
        return result

    def removeVec(self, vector):
        """
        Return a new set identical to self, but from which vector
        has been removed.
        """
        setprime = []
        for i in range(len(self.set)):
            if(not equalVectors(self.set[i], vector)):
                setprime.append(self.set[i])
        vvs = ValueVectorSet()
        vvs.addAll(setprime)
        return vvs

    def removeMaximisingLinearScalarisedValue(self, weight):
        """
        Remove the vector that maximises the scalarised value
        for a given scalarisation weight, and return a tuple
        with this vector and the value set from which this
        vector has been removed
        """
        maxscalval = -numpy.Infinity
        maxvec = None
        for i in range(len(self.set)):
            scalval = inner_product(self.set[i], weight)
            if(scalval > maxscalval):
                maxvec = self.set[i]
                maxscalval = scalval
        vvs = self.removeVec(maxvec)
        return (maxvec, vvs)

    def removeMaximisingForExtrema(self):
        """
        Remove the vectors that maximises the scalarised value
        at the extrema of the weight space, and return a tuple
        with these vectors and the value set from which these
        vectors have been removed
        """
        if(self.empty()):
            return (None, None)
        nObjectives = len(self.set[0])
        rest = self
        maximising = []
        for o in range(nObjectives):
            maxscalval = -numpy.Infinity
            maxvec = None
            for i in range(len(self.set)):
                scalval = self.set[i][o]
                if(scalval > maxscalval):
                    maxvec = self.set[i]
                    maxscalval = scalval
            if(maxvec is not None):
                rest = rest.removeVec(maxvec)
                maximising.append(maxvec)
        vvs = ValueVectorSet()
        vvs.addAll(maximising)
        return (vvs, rest)


class ValueFunction:
    """
    A class containing a multi-objective value function indexed
    with an integer state: V(S), returning a ValueVectorSet
    """

    def __init__(self, nStates):
        """
        """
        self.tables = []
        for i in range(nStates):
            vvs = ValueVectorSet()
            self.tables.append(vvs)

    def addVectorToAllSets(self, vector):
        """
        """
        for i in range(len(self.tables)):
            vvs = self.tables[i]
            vvs.add(vector)

    def __str__(self):
        """
        """
        result = ""
        for i in range(len(self.tables)):
            result = result+str(i)+":\n" + self.tables[i] + "\n\n"
        return result

    def getValue(self, stateIndex):
        """
        """
        return self.tables[stateIndex]

    def setValue(self, stateIndex, vvs):
        """
        """
        self.tables[stateIndex] = vvs
