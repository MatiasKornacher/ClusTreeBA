from river import base, cluster, stats, utils
import math


class ClusTree(base.Clusterer):
    def __init__(self, root, lambda_=0.1, max_radius=1.0, tsnap=100, beta=2):
        super().__init__()
        self.root=root
        self.lambda_ = lambda_
        self.max_radius = max_radius
        self.time = 0
        self.tsnap = tsnap
        self.beta = beta

    def learn_one(self, x):
        self.time += 1
        t = self.time
        node = self.root
        hitchhiker = None

        #Handling of empty tree:
        if not node.entries:
            new_cf=ClusterFeature(n=1, LS=x.copy(),SS={k: v ** 2 for k, v in x.items()}, timestamp=t)
            node.add_entry(Entry(cf_data=new_cf, is_leaf=True))
            return self

        while not node.is_leaf():
            # updating current node’s timestamp
            # ToDo maybe own method
            for entry in node.entries:
                entry.cf_data.decay(t, self.lambda_)
                if entry.cf_buffer:
                    entry.cf_buffer.decay(t, self.lambda_)

            closest_to_x = min(node.entries, key=lambda e: self._euclidean_distance(x, e.cf_data.center()))

            
            if closest_to_x.cf_buffer:
                #ToDo check if same closest entry
                hitchhiker = closest_to_x.cf_buffer
                closest_to_x.cf_buffer = None

        #reached leaf level
        for entry in node.entries:
            entry.cf_data.decay(t, self.lambda_)

        if hitchhiker:
            #check
            if node.is_full():
                closest_to_hitchhiker = min(node.entries,key=lambda e: self._euclidean_distance(hitchhiker.center(), e.cf_data.center()))
                closest_to_hitchhiker.cf_data.add_cluster(hitchhiker, t, self.lambda_)
            else:
                node.add_entry(Entry(cf_data=hitchhiker, is_leaf=True))


        closest_to_x = min(node.entries, key=lambda e: self._euclidean_distance(x, e.cf_data.center()))

        if self._euclidean_distance(x, closest_to_x.cf_data.center()) <= self.max_radius:
            closest_to_x.cf_data.add_object(x, t, self.lambda_)
        elif node.is_full():
            node.merge_entries(self._euclidean_distance)
            if node.is_full():  # still too full after merging
                if self.try_discard_insignificant_entry(node):
                    return self.learn_one(x) #bc true means we removed one
                self.split(node)
                return self.learn_one(x)
        else:
            new_cf = ClusterFeature(n=1,LS=x.copy(),SS={k: v ** 2 for k, v in x.items()},timestamp=t)
            node.add_entry(Entry(cf_data=new_cf, is_leaf=True))
        return self

    def predict_one(self, x):
        leaf_entries = [] #get all leafs, maybe own method

        def traverse(node):
            if node.is_leaf():
                leaf_entries.extend(node.entries)
            else:
                for entry in node.entries:
                    if entry.child:
                        traverse(entry.child)

        traverse(self.root)

        if not leaf_entries:
            return 0  # fallback
        closest_entry = min(
            leaf_entries,
            key=lambda e: self._euclidean_distance(x, e.cf_data.center())
        )
        return closest_entry.cf_data.center()

    def try_discard_insignificant_entry(self, node):#check if entry can be removed. return true if removes something, false else
        if len(node.entries) < node.MAX_ENTRIES: #check if even necessary
            return False

        threshold = self.beta ** (-self.lambda_ * self.tsnap)
        least_significant = min(node.entries, key=lambda e: e.cf_data.n)

        if least_significant.cf_data.n < threshold:
            node.entries.remove(least_significant)
            self._remove_entry_stats_up_tree(least_significant, node)
            return True

        return False

    def _remove_entry_stats_up_tree(self, entry, node):
        cf_to_remove = entry.cf_data
        while node is not None:
            for parent_entry in node.entries:
                if parent_entry.child == node:
                    parent_entry.cf_data.subtract_cluster(cf_to_remove)
            node = node.parent

    def split(self, node):
        t=self.time
        max_dist = -1

        entry1, entry2 = None, None #ToDo maybe own method
        for i in range(len(node.entries)):
            for j in range(i + 1, len(node.entries)):
                c1 = node.entries[i].cf_data.center()
                c2 = node.entries[j].cf_data.center()
                d = self._euclidean_distance(c1, c2)
                if d > max_dist:
                    max_dist = d
                    entry1, entry2 = node.entries[i], node.entries[j]

        node1 = Node(parent=node.parent)
        node2 = Node(parent=node.parent)
        node1.add_entry(entry1)
        node2.add_entry(entry2)

        for entry in node.entries:
            if entry in (entry1, entry2):
                continue
            d1 = self._euclidean_distance(entry.cf_data.center(), entry1.cf_data.center())
            d2 = self._euclidean_distance(entry.cf_data.center(), entry2.cf_data.center())
            if d1 < d2:
                node1.add_entry(entry)
            else:
                node2.add_entry(entry)

        if node.parent is None:#check for if parent is the root
            new_root = Node()
            new_root.add_entry(Entry(cf_data=node1.aggregate_cf(current_time=t, lambda_=self.lambda_), is_leaf=False, child=node1))
            new_root.add_entry(Entry(cf_data=node2.aggregate_cf(current_time=t, lambda_=self.lambda_), is_leaf=False, child=node2))
            self.root = new_root
        else:
            parent = node.parent
            parent.entries = [e for e in parent.entries if e.child != node]
            parent.add_entry(Entry(cf_data=node1.aggregate_cf(current_time=t, lambda_=self.lambda_), is_leaf=False, child=node1))
            parent.add_entry(Entry(cf_data=node2.aggregate_cf(current_time=t, lambda_=self.lambda_), is_leaf=False, child=node2))

            if parent.is_full():
                if self.try_discard_insignificant_entry(parent):
                    return
                self.split(parent)  #recursive split if parent now overflows

    @staticmethod
    def _euclidean_distance(a, b):
        if a is None or b is None:
            return float('inf')
        if len(a) != len(b):
            return float('inf')#error but return inf to avoid crash
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a.values(), b.values())))


class ClusterFeature(base.Base):
    def __init__(self, n=0, LS=None, SS=None, timestamp=0):
        self.n = n
        self.LS = LS
        self.SS = SS
        self.timestamp = timestamp

    def center(self):
        if self.n == 0 or self.LS is None:
            return None
        return {k: v / self.n for k, v in self.LS.items()}

    def add_object(self, object_, current_time, lambda_):
        self.decay(current_time, lambda_)
        self.n += 1
        if self.LS is None:
            self.LS = object_.copy()
            self.SS = {k: v ** 2 for k, v in object_.items()}
        else:
            for x in object_:
                self.LS[x] = self.LS.get(x, 0.0) + object_[x]
                self.SS[x] = self.SS.get(x, 0.0) + object_[x] ** 2
        self.timestamp = current_time

    def add_cluster(self, cf, current_time, lambda_):
        cf.decay(current_time, lambda_)
        self.decay(current_time, lambda_)
        self.n += cf.n

        if self.LS is None:
            self.LS = cf.LS.copy()
            self.SS = cf.SS.copy()
        else:
            for k in cf.LS:
                self.LS[k] = self.LS.get(k, 0.0) + cf.LS[k]
                self.SS[k] = self.SS.get(k, 0.0) + cf.SS[k]

    def subtract_cluster(self, cf):
        self.n -= cf.n
        for k in cf.LS:
            self.LS[k] -= cf.LS[k]
        for k in cf.SS:
            self.SS[k] -= cf.SS[k]

    def decay(self, current_time, lambda_):
        dt = current_time - self.timestamp
        if dt <= 0: #check for no time passed
            return
        decay_factor = 2 ** (-lambda_ * dt)#beta=2 like in paper
        self.n *= decay_factor
        if self.LS is not None:
            for x in self.LS:
                self.LS[x] *= decay_factor
            #self.LS = [x * decay_factor for x in self.LS]
        if self.SS is not None:
            for x in self.SS:
                self.SS[x] *= decay_factor
            #self.SS = [x * decay_factor for x in self.SS]
        self.timestamp = current_time


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

    def aggregate_entry(self, current_time, lambda_):
        if self.child is None:
            return self.cf_data
        agg_cf = ClusterFeature()
        for entry in self.child.entries:
            entry.cf_data.decay(current_time, lambda_)
            agg_cf.add_cluster(entry.cf_data, current_time, lambda_)
        return agg_cf

    def merge_with(self, other):
        self.cf_data.add_cluster(other.cf_data)

class Node(base.Base):
    MAX_ENTRIES = 3

    def __init__(self, parent=None):
        self.entries = []
        self.parent = parent

    def is_leaf(self):
        return all(entry.child is None for entry in self.entries)

    def add_entry(self, entry):
        if len(self.entries) >= self.MAX_ENTRIES:
            raise ValueError("Node is full")
        self.entries.append(entry)
        if entry.child is not None:
            entry.child.parent = self

    def aggregate_cf(self, current_time, lambda_):
        cf = ClusterFeature()
        for entry in self.entries:
            entry.cf_data.decay(current_time, lambda_)
            cf.add_cluster(entry.cf_data, current_time, lambda_)
        return cf

    def is_full(self):
        return len(self.entries) >= self.MAX_ENTRIES

    def merge_entries(self,distance_calc):
        while len(self.entries) > self.MAX_ENTRIES:
            min_dist = float('inf')
            pair = None
            for i in range(len(self.entries)):
                for j in range(i + 1, len(self.entries)):
                    d = distance_calc(
                        self.entries[i].cf_data.center(),
                        self.entries[j].cf_data.center()
                    )
                    if d < min_dist:
                        min_dist = d
                        pair = (i, j)

            i, j = pair
            self.entries[i].merge_with(self.entries[j])
            del self.entries[j]