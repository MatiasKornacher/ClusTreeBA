import math
from collections import defaultdict

from river import base, cluster, stats, utils

class ClusTree(base.Clusterer):
    def __init__(self, root):
        super().__init__()
        self.root=root

    def learn_one(self):
        #ToDo

    def predict_one(self, x):
        #ToDo

    def split(self, node):
        max_dist = -1

class ClusterFeature(base.Base):
    def __init__(self, n=0, LS=None, SS=None):
        self.n = n
        self.LS = LS
        self.SS = SS

        def center(self):
            if self.n == 0:
                return None
            return {k: self.LS[k] / self.n for k in self.LS}

        def addObject(self, object):
            self.n += 1
            if self.LS is None:
                self.LS = object.copy()
                self.SS = {k: v ** 2 for k, v in object.items()}
            else:
                for x in object:
                    self.LS[x] += object[x]
                    self.SS[x] += object[x] ** 2

        def addCluster(self, cf):
            self.n += cf.n
            for x in cf.LS:
                self.LS[x] += cf.LS[x]
                self.SS[x] += cf.SS[x]

        def clear(self):
            self.n = 0
            self.LS = None
            self.SS = None

class Entry(base.Base):
    def __init__(self, cf_data, is_leaf, cf_buffer=None, child=None):
        self.cf_data = cf_data
        self.is_leaf = is_leaf
        self.cf_buffer = cf_buffer
        self.child = child

    def aggregateEntry(self):
        if self.child is None:
            return self.cf_data
        agg_cf = ClusterFeature()
        for entry in self.child.entries:
            agg_cf.addCluster(entry.cf_data)
        return agg_cf

    def mergeWith(self, other):
        self.cf_data.addCluster(other.cf_data)

class Node(base.Base):
    MAX_ENTRIES = 3

    def __init__(self, parent=None):
        self.entries = []
        self.parent = parent

    def is_leaf(self):
        return all(entry.child is None for entry in self.entries)

    def add_entry(self, entry):
        if len(self.entries) >= self.MAX_ENTRIES:
        # exception

        self.entries.append(entry)
        if entry.child is not None:
            entry.child.parent = self

    def aggregate_cf(self):
        cf = ClusterFeature()
        for entry in self.entries:
            cf.addCluster(entry.cf_data)
        return cf

    def is_full(self):
        return len(self.entries) >= self.MAX_ENTRIES

    def mergeEntries(self):
        while len(self.entries) > self.MAX_ENTRIES:
            min_dist = float('inf')
            pair = None
            for i in range(len(self.entries)):
                for j in range(i + 1, len(self.entries)):
                    d = self._euclidean_distance(
                        self.entries[i].cf_data.center(),
                        self.entries[j].cf_data.center()
                    )
                    if d < min_dist:
                        min_dist = d
                        pair = (i, j)

            i, j = pair
            self.entries[i].mergeWith(self.entries[j])
            del self.entries[j]