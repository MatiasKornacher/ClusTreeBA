from ClusTree import ClusTree, Node

# Sample streaming data
stream = [
    {'x': 1.0},
    {'x': 1.1},
    {'x': 4.8},
    {'x': 5.0},
    {'x': 5.1},
    {'x': 1.2}
]

# Initialize ClusTree with or without aggregation
root_node = Node()
ct = ClusTree(root=root_node, lambda_=0.1, max_radius=1.0, use_aggregation=True)

# Simulate a stream
for x in stream:
    ct.learn_one(x)
    print(f"Processed: {x}, Tree time: {ct.time}, Aggregates: {len(ct.aggregates)}")

