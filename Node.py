class Node:
    def __init__(self, level, is_leaf=True):
        self.level = level
        self.entries = []
        self.is_leaf = is_leaf

    def isLeaf(self):
        for i in entries:
            entry = entries[i]
            if (entry.getChild() != null):
                self.is_leaf = False
                return False

    def nearestEntry(self,buffer):
        if not self.entries:
            return None
        #ToDo

    def mergeEntries(self, pos1, pos2):
        #ToDo

    def makeOlder(self, currentTime, negLambda):
        #ToDo