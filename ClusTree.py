from river import base, cluster, stats, utils
from sklearn.metrics.pairwise import euclidean_distances
import copy


class ClusTree(base.Clusterer):
    def __init__(self, root, lambda_=0.1, max_radius=1.0, tsnap=100, beta=2, use_aggregation=False, max_aggregates=10):
        super().__init__()
        self.root=root
        self.current_node=self.root
        self.hitchhiker=None
        self.pending=None
        self.lambda_ = lambda_
        self.max_radius = max_radius
        self.time = 0
        self.tsnap = tsnap
        self.snapshots = []
        self.last_snapshot_time = 0
        self.beta = beta
        self.aggregates = []
        self.use_aggregation = use_aggregation
        self.max_aggregates = max_aggregates
        self.new_arrival=False

    def learn_one(self, x):
        self.time += 1
        if self.use_aggregation:
            self.aggregate_or_update(x)
        else:
            #conversion to cf
            input_cf = ClusterFeature(n=1, LS=x.copy(), SS={k: v * v for k, v in x.items()}, timestamp=self.time)
            self.pending = input_cf
        return self

    def update_one(self):
        t = self.time
        node = self.current_node

        # Handling of empty tree:
        if not node.entries:
            node.add_entry(Entry(cf_data=self.pending))
            self.pending = None
            return

        node.decay_all_entries(t, self.lambda_)

        if not node.is_leaf():
            closest = min(node.entries, key=lambda e: self._euclidean_distance(self.pending.center(), e.cf_data.center()))

            if self.hitchhiker is not None:
                #check for same closest entries
                closest_to_hitch = min(node.entries,key=lambda e: self._euclidean_distance(self.hitchhiker.cf_data.center(), e.cf_data.center()))

                if closest_to_hitch is not closest:
                    closest_to_hitch.cf_buffer.add_cluster(self.hitchhiker, t, self.lambda_)
                    self.hitchhiker = None
            if self.new_arrival:#TODO check if new arrival.
                closest.cf_buffer.add_cluster(self.hitchhiker, t, self.lambda_)
                self.hitchhiker = None
                closest.cf_buffer.add_cluster(self.pending, t, self.lambda_)
                self.pending = None
                return
            if closest.cf_buffer:
                if self.hitchhiker is None:
                    self.hitchhiker = closest.cf_buffer
                else:
                    self.hitchhiker.cf_data.add_cluster(closest.cf_buffer.cf_data, t, self.lambda_)
                closest.cf_buffer = None
            self.current_node = closest.child
            return

        if node.is_leaf():
            if self.hitchhiker:
                # check
                if node.is_full():
                    closest_to_hitch = min(node.entries, key=lambda e: self._euclidean_distance(self.hitchhiker.center(),e.cf_data.center()))
                    closest_to_hitch.cf_data.add_cluster(self.hitchhiker, t, self.lambda_)
                else:
                    node.add_entry(Entry(cf_data=self.hitchhiker))
                self.hitchhiker=None

            # closest = min(node.entries, key=lambda e: self._euclidean_distance(self.pending.center(), e.cf_data.center()))

            # if self._euclidean_distance(self.pending.center(), closest.cf_data.center()) <= self.max_radius:
            #     closest.cf_data.add_cluster(self.pending, t, self.lambda_)

            elif node.is_full():
                if self.new_arrival:# TODO check if new arrival.
                    node.merge_entries(self._euclidean_distance)
                    node.add_entry(Entry(cf_data=self.pending))
                    self.pending = None
                    return
                elif node.is_full():
                    if self.try_discard_insignificant_entry(node):#check before merge???
                        node.add_entry(Entry(cf_data=self.pending))
                        self.pending = None
                        return
                    self.current_node = self.split(node) #returns parent
                    self.update_one()
                    return

            else:
                node.add_entry(Entry(cf_data=self.pending))
                self.pending=None
                return



    def update_max_radius_from_leaves(self):
        leaf_vars = []

        def collect_leaf_vars(node):
            if node.is_leaf():
                for entry in node.entries:
                    cf = entry.cf_data
                    if cf.n > 0:
                        dim_vars = []
                        for k in cf.LS:
                            mean = cf.LS[k] / cf.n
                            var = (cf.SS[k] / cf.n) - (mean ** 2)
                            if var > 0:
                                dim_vars.append(var)
                        if dim_vars:
                            avg_var = sum(dim_vars) / len(dim_vars)
                            leaf_vars.append(avg_var)
            else:
                for entry in node.entries:
                    if entry.child:
                        collect_leaf_vars(entry.child)

        collect_leaf_vars(self.root)

        if leaf_vars:
            self.max_radius = sum(leaf_vars) / len(leaf_vars)

    def aggregate_or_update(self, x):#misleading name now, update happens separately
        t = self.time
        closest_cf = None
        closest_dist = float('inf')

        self.update_max_radius_from_leaves()

        # Find the closest aggregate within max_radius
        for cf in self.aggregates:
            dist = self._euclidean_distance(x, cf.center())
            if dist < closest_dist and dist <= self.max_radius:
                closest_cf = cf
                closest_dist = dist

        # If a close enough aggregate was found, update it
        if closest_cf:
            closest_cf.add_object(x, current_time=t, lambda_=self.lambda_)

        else:
            new_cf = ClusterFeature(n=1,LS=x.copy(),SS={k: v ** 2 for k, v in x.items()},timestamp=t)
            self.aggregates.append(new_cf)

        if len(self.aggregates) > self.max_aggregates:
            self.aggregates.sort(key=lambda agg: (-agg.n, agg.timestamp))
            cf_to_insert = self.aggregates.pop(0)
            self.pending = cf_to_insert


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
            raise ValueError("Tree is empty.")
        closest_entry = min(
            leaf_entries,
            key=lambda e: self._euclidean_distance(x, e.cf_data.center())
        )
        return closest_entry.cf_data.center()

    def try_discard_insignificant_entry(self, node):#check if entry can be removed. return true if removes something, false else
        if len(node.entries) < node.MAX_ENTRIES: #check if even necessary
            return False

        tsince = self.time - self.last_snapshot_time
        threshold = self.beta ** (-self.lambda_ * tsince)
        least_significant = min(node.entries, key=lambda e: e.cf_data.n)

        if least_significant.cf_data.n < threshold:
            node.entries.remove(least_significant)
            self._remove_entry_stats_up_tree(least_significant, node)
            return True

        return False

    @staticmethod
    def _remove_entry_stats_up_tree(entry, node):
        cf_to_remove = entry.cf_data
        while node is not None:
            for parent_entry in node.entries:
                if parent_entry.child == node:
                    parent_entry.cf_data.subtract_cluster(cf_to_remove)
            node = node.parent

    def split(self, node):
        t=self.time
        max_dist = -1

        entry1, entry2 = None, None
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
                if entry.child:
                    entry.child.parent = node1
            else:
                node2.add_entry(entry)
                if entry.child:
                    entry.child.parent = node2

        node.entries.clear()

        if node.parent is None:#check for if parent is the root
            new_root = Node()
            new_root.add_entry(Entry(cf_data=node1.aggregate_cf(current_time=t, lambda_=self.lambda_), child=node1))
            new_root.add_entry(Entry(cf_data=node2.aggregate_cf(current_time=t, lambda_=self.lambda_), child=node2))
            self.root = new_root
            self.take_snapshot()
            return new_root
        else:
            parent = node.parent
            parent.entries = [e for e in parent.entries if e.child != node]
            parent.add_entry(Entry(cf_data=node1.aggregate_cf(current_time=t, lambda_=self.lambda_), child=node1))
            parent.add_entry(Entry(cf_data=node2.aggregate_cf(current_time=t, lambda_=self.lambda_), child=node2))
            if parent.is_full():
                if self.try_discard_insignificant_entry(parent):
                    self.take_snapshot()
                    return node.parent
                self.split(parent)  #recursive split if parent now overflows

            self.take_snapshot()
            return node.parent

    def take_snapshot(self):
         self.snapshots.append(copy.deepcopy(self.root))
         self.last_snapshot_time = self.time

    # @staticmethod
    # def _euclidean_distance(a, b):
    #     if a is None or b is None:
    #          return float('inf')
    #     keys = sorted(set(a.keys()) | set(b.keys()))
    #     vec_a = np.array([a.get(k, 0.0) for k in keys]).reshape(1, -1)
    #     vec_b = np.array([b.get(k, 0.0) for k in keys]).reshape(1, -1)
    #     return float(euclidean_distances(vec_a, vec_b)[0, 0])

    @staticmethod
    def _euclidean_distance(a, b):
        if a is None or b is None:
            return float('inf')

        keys = sorted(set(a) | set(b))
        vec_a = [[a.get(k, 0.0) for k in keys]]
        vec_b = [[b.get(k, 0.0) for k in keys]]
        return float(euclidean_distances(vec_a, vec_b)[0, 0])


class ClusterFeature(base.Base):
    # noinspection PyPep8Naming
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
        if self.timestamp is None:
            self.timestamp = current_time
            return
        dt = current_time - self.timestamp
        if dt <= 0: #check for no time passed
            return
        decay_factor = 2 ** (-lambda_ * dt)#beta=2 like in paper
        self.n *= decay_factor
        if self.LS is not None:
            for x in self.LS:
                self.LS[x] *= decay_factor
        if self.SS is not None:
            for x in self.SS:
                self.SS[x] *= decay_factor
        self.timestamp = current_time



    def clear(self):
        self.n = 0
        self.LS = None
        self.SS = None

class Entry(base.Base):
    def __init__(self, cf_data, cf_buffer=None, child=None):
        self.cf_data = cf_data
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

    def decay_all_entries(self, current_time, lambda_):
        for entry in self.entries:
            entry.cf_data.decay(current_time, lambda_)
            if entry.cf_buffer:
                entry.cf_buffer.decay(current_time, lambda_)