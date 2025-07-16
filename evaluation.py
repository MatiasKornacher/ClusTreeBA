from river import datasets
from river.datasets import synth
from ClusTree import ClusTree, Node
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, accuracy_score
from collections import defaultdict

stream = synth.Agrawal(seed=42).take(100)



root = Node()
clustree = ClusTree(root=root, lambda_=0.1, max_radius=2.0)

y_true = []
y_pred = []

for x, y in stream:
    clustree.learn_one(x)
    pred_cluster = clustree.predict_one(x)
    y_true.append(y)
    y_pred.append(str(pred_cluster))

def evaluate_clustering(y_true, y_pred):
    cluster_to_label = {}
    clusters = defaultdict(list)
    for pred, true in zip(y_pred, y_true):
        clusters[pred].append(true)
    for cluster_id, trues in clusters.items():
        if trues:
            cluster_to_label[cluster_id] = max(set(trues), key=trues.count)
    mapped_preds = [cluster_to_label.get(p, -1) for p in y_pred]

    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    acc = accuracy_score(y_true, mapped_preds)
    return {"ARI": ari, "AMI": ami, "Accuracy": acc}

metrics = evaluate_clustering(y_true, y_pred)
print(metrics)