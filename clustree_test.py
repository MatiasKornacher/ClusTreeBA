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

def test_node_decay_all_entries():
    node = Node()
    cf_main = ClusterFeature(n=2, LS={'x': 4.0}, SS={'x': 10.0}, timestamp=0)
    cf_buffer = ClusterFeature(n=1, LS={'x': 2.0}, SS={'x': 4.0}, timestamp=0)

    entry = Entry(cf_data=cf_main, is_leaf=True, cf_buffer=cf_buffer)
    node.add_entry(entry)

    node.decay_all_entries(current_time=10, lambda_=0.1)

    # Expected decay factor: 2^(-λ * Δt) = 2^(-0.1 * 10) = 2^-1 = 0.5
    expected_decay = 0.5

    assert abs(entry.cf_data.n - 2 * expected_decay) < 1e-6
    assert abs(entry.cf_data.LS['x'] - 4.0 * expected_decay) < 1e-6
    assert abs(entry.cf_data.SS['x'] - 10.0 * expected_decay) < 1e-6

    assert abs(entry.cf_buffer.n - 1 * expected_decay) < 1e-6
    assert abs(entry.cf_buffer.LS['x'] - 2.0 * expected_decay) < 1e-6
    assert abs(entry.cf_buffer.SS['x'] - 4.0 * expected_decay) < 1e-6

def test_predict_one_uses_nearest_leaf():

    root = Node()
    leaf = Node(parent=root)

    cf1 = ClusterFeature(n=2, LS={'x': 2.0}, SS={'x': 4.0}, timestamp=0)  # center = 1.0
    cf2 = ClusterFeature(n=2, LS={'x': 8.0}, SS={'x': 32.0}, timestamp=0)  # center = 4.0

    entry1 = Entry(cf_data=cf1, is_leaf=True)
    entry2 = Entry(cf_data=cf2, is_leaf=True)
    leaf.entries.extend([entry1, entry2])

    root.entries.append(Entry(cf_data=None, is_leaf=False, child=leaf))
    ct = ClusTree(root=root, lambda_=0.1, max_radius=1.0)

    # Prediction target closer to entry1 (center=1.0) than entry2 (center=4.0)
    x = {'x': 1.1}
    prediction = ct.predict_one(x)

    # nearest neighbour should be cf1
    expected_center = cf1.center()

    assert abs(prediction['x'] - expected_center['x']) < 1e-6


def test_no_aggregation_in_normal_mode():
    from ClusTree import ClusTree, Node

    root = Node()
    ct = ClusTree(root=root, lambda_=0.1, max_radius=1.0, use_aggregation=False)

    ct.learn_one({'x': 1.0})
    ct.learn_one({'x': 1.1})
    ct.learn_one({'x': 1.2})

    # No aggregation in normal mode
    assert len(ct.aggregates) == 0


def test_aggregate_flush_when_exceeding_capacity():
    from ClusTree import ClusTree, Node, ClusterFeature

    root = Node()
    ct = ClusTree(root=root, lambda_=0.1, max_radius=1.0, use_aggregation=True)
    ct.time = 100  # simulate current time

    # Create 11 aggregates to exceed capacity of 10
    for i in range(10):
        cf = ClusterFeature(
            n=1.0,
            LS={'x': 1.0 + i},
            SS={'x': (1.0 + i) ** 2},
            timestamp=90 + i
        )
        ct.aggregates.append(cf)

    # Add one more that should trigger a flush
    cf_target = ClusterFeature(
        n=3.0,  # highest n
        LS={'x': 3.0},
        SS={'x': 9.0},
        timestamp=80  # oldest
    )
    ct.aggregates.append(cf_target)

    # Verify precondition
    assert len(ct.aggregates) == 11

    # Insert a new object (should trigger flush)
    ct.aggregate_or_insert({'x': 5.0})

    # Check that one aggregate was flushed
    assert len(ct.aggregates) == 10
    assert cf_target not in ct.aggregates

    # Root should have one new entry from flushed aggregate
    assert len(root.entries) == 1
    assert abs(root.entries[0].cf_data.n - cf_target.n) < 1e-6

def test_snapshot_creates_deep_copy():
    from ClusTree import ClusTree, Node, ClusterFeature, Entry

    # Set up a simple tree
    root = Node()
    cf = ClusterFeature(n=1.0, LS={'x': 1.0}, SS={'x': 1.0}, timestamp=0)
    root.add_entry(Entry(cf_data=cf, is_leaf=True))
    clustree = ClusTree(root=root, lambda_=0.1, max_radius=1.0)
    clustree.time = 50

    # Take a snapshot
    clustree.take_snapshot()

    # Check snapshot list updated
    assert len(clustree.snapshots) == 1
    snapshot_root = clustree.snapshots[0]

    # Modify original tree
    new_cf = ClusterFeature(n=1.0, LS={'x': 2.0}, SS={'x': 4.0}, timestamp=50)
    root.add_entry(Entry(cf_data=new_cf, is_leaf=True))

    # Ensure snapshot is unaffected
    assert len(snapshot_root.entries) == 1  # Still only has the original entry
    assert snapshot_root.entries[0].cf_data.LS['x'] == 1.0