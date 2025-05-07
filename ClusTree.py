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

class ClusterFeature(base.Base):
    def __init__(self, n=0, LS=None, SS=None):
        self.n = n
        self.LS = LS
        self.SS = SS

        def center(self):
            if self.n == 0:
                return None
            return [x / self.n for x in self.LS]  # ToDo add weight

        def addObject(self, object):
            self.n += 1

        def addCluster(self, cf):
            self.n += cf.n
            # ToDo

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

    def aggregateEntry(self):  # ToDo

    def mergeWith(self):  # ToDo

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

    def is_full(self):
        return len(self.entries) >= self.MAX_ENTRIES

    def mergeEntries(self):  # ToDo