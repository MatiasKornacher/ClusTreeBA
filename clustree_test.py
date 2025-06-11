import pytest
import math
from ClusTree import ClusTree, ClusterFeature, Entry, Node


def test_decay_applies_exponential_weight():
    cf = ClusterFeature(n=2, LS={'x': 6.0}, SS={'x': 18.0}, timestamp=0)
    cf.decay(current_time=5, lambda_=0.1)

    decay_factor = 2 ** (-0.1 * 5)
    assert math.isclose(cf.n, 2 * decay_factor, rel_tol=1e-6)
    assert math.isclose(cf.LS['x'], 6.0 * decay_factor, rel_tol=1e-6)
    assert math.isclose(cf.SS['x'], 18.0 * decay_factor, rel_tol=1e-6)

def test_add_object_updates_cluster_stats():
    cf = ClusterFeature(n=0, LS={}, SS={}, timestamp=0)
    cf.add_object({'x': 3.0}, current_time=1, lambda_=0.1)

    assert cf.n > 0
    assert math.isclose(cf.LS['x'], 3.0, rel_tol=1e-6)
    assert math.isclose(cf.SS['x'], 9.0, rel_tol=1e-6)
    assert cf.timestamp == 1

def test_add_cluster_combines_stats():
    cf1 = ClusterFeature(n=1, LS={'x': 2.0}, SS={'x': 4.0}, timestamp=0)
    cf2 = ClusterFeature(n=1, LS={'x': 3.0}, SS={'x': 9.0}, timestamp=0)

    cf1.add_cluster(cf2, current_time=1, lambda_=0.0)

    assert cf1.n == 2
    assert math.isclose(cf1.LS['x'], 5.0, rel_tol=1e-6)
    assert math.isclose(cf1.SS['x'], 13.0, rel_tol=1e-6)


def test_node_entry_management():
    node = Node()
    assert not node.is_full()

    for i in range(3):
        cf = ClusterFeature(n=1, LS={'x': i}, SS={'x': i ** 2}, timestamp=0)
        node.add_entry(Entry(cf_data=cf, is_leaf=True))

    assert node.is_full()

def test_learn_and_predict():
    root = Node()
    clustree = ClusTree(root=root, lambda_=0.1, max_radius=2.0)

    clustree.learn_one({'x': 1.0})
    clustree.learn_one({'x': 1.2})
    clustree.learn_one({'x': 3.0})

    pred = clustree.predict_one({'x': 1.1})
    assert isinstance(pred, int)

def test_discard_insignificant_entry():
    from ClusTree import ClusTree, Node, Entry, ClusterFeature

    # Create a node with 3 entries (full)
    node = Node()
    node.MAX_ENTRIES = 3  # explicitly reinforce for test clarity

    # Create low-significance entries (n very small)
    e1 = Entry(cf_data=ClusterFeature(n=0.0001, LS={'x': 0.0001}, SS={'x': 0.00000001}, timestamp=0), is_leaf=True)
    e2 = Entry(cf_data=ClusterFeature(n=0.02, LS={'x': 0.02}, SS={'x': 0.0004}, timestamp=0), is_leaf=True)
    e3 = Entry(cf_data=ClusterFeature(n=0.03, LS={'x': 0.03}, SS={'x': 0.0009}, timestamp=0), is_leaf=True)

    node.entries.extend([e1, e2, e3])

    # Create a fake root to simulate upward stats
    root = Node()
    root.entries.append(Entry(cf_data=ClusterFeature(n=0.06, LS={'x': 0.06}, SS={'x': 0.0014}, timestamp=0), is_leaf=False, child=node))
    node.parent = root

    # Create ClusTree instance
    ct = ClusTree(root=root, lambda_=0.1, tsnap=100)
    ct.time = 100  # simulate time to trigger decay

    # Should discard the lowest-n entry
    removed = ct.try_discard_insignificant_entry(node)

    assert removed is True
    assert len(node.entries) == 2

def test_no_discard_when_all_significant():
    from ClusTree import ClusTree, Node, Entry, ClusterFeature

    # Create a full node with significant entries (n > threshold)
    node = Node()
    node.MAX_ENTRIES = 3

    # Use n values that are clearly above threshold (e.g., n = 10)
    e1 = Entry(cf_data=ClusterFeature(n=10, LS={'x': 10}, SS={'x': 100}, timestamp=0), is_leaf=True)
    e2 = Entry(cf_data=ClusterFeature(n=12, LS={'x': 12}, SS={'x': 144}, timestamp=0), is_leaf=True)
    e3 = Entry(cf_data=ClusterFeature(n=15, LS={'x': 15}, SS={'x': 225}, timestamp=0), is_leaf=True)

    node.entries.extend([e1, e2, e3])

    # Setup fake parent
    root = Node()
    root.entries.append(Entry(cf_data=ClusterFeature(n=37, LS={'x': 37}, SS={'x': 469}, timestamp=0), is_leaf=False, child=node))
    node.parent = root

    # Create ClusTree instance
    ct = ClusTree(root=root, lambda_=0.1, tsnap=100)
    ct.time = 100

    # Should NOT discard any entry
    removed = ct.try_discard_insignificant_entry(node)

    assert removed is False
    assert len(node.entries) == 3  # still full