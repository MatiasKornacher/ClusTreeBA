class ClusKernel:
    def __init__(self, point, dim):
        self.totalN
        self.N = 1.0
        self.LS = np.array(point, dtype=np.float64)
        self.SS = np.square(self.LS)

    def get_center(self):
        return self.LS / self.N

    def add(self, other):
        self.N += other.N
        #toDo update LS and SS

    def makeOlder(self, timeDiff, negLambda):
        if (timeDiff == 0):
            return
        weightFactor = pow(2.0, negLambda * timeDiff)
        self.N *= weightFactor
        #toDo for update LS and SS

    def aggregate(self, other, timeDiff, negLambda):
        makeOlder(self, timeDiff, negLambda)
        add(other)

    def calcDistance(self, other):
        #ToDo

    def clear(self):
        self.totalN = 0
        self.N = 0.0
        #ToDo clear LS and SS

    def overwriteOldCluster(other):
        #ToDo

