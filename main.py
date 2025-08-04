import itertools
import time
import csv
import argparse

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from TSindex import tempsil

import numpy as np
from clustpy.metrics.clustering_metrics import unsupervised_clustering_accuracy

import river.cluster.clustream as clustream
import river.cluster.dbstream as dbstream
import river.cluster.denstream as denstream
import river.cluster.streamkmeans as streamkmeans

from ClusTree import ClusTree

def parse_args():
	parser = argparse.ArgumentParser(
		prog='main',
		description = 'mainparser'
	)

	parser.add_argument(
		"--data", "-d",
		required=True,
		help="Dataset to use (real or synth)."
	)

	parser.add_argument(
        "--algorithm", "-a",
        required=True, choices=["CluStream","DenStream","DBSTREAM","STREAMKMeans", "ClusTreeAgg", "ClusTreeNoAgg"],
        help="Which clustering algorithm to run."
    )

	parser.add_argument(
		"--timestep", "-t",
		type=float, choices=[0.0001, 0.001, 0.01],
		help="Window size (number of points) between evaluations."
	)

	parser.add_argument(
        "--eval-timestep", "-e",
        type=int, default=1000,
        help="Window size (number of points) between evaluations."
    )

	parser.add_argument(
        "--convdict",
        action="store_true",
        help="If set, convert tuple/array datapoints to dicts via `dict(enumerate(x))`."
    )

	return parser.parse_args()

def main(alg, datastream, timesteps, eval_timestep, convdict=True):

	if alg.__class__ in [ClusTree]:
		hasUpdate = True
	else:#river algos
		hasUpdate = False

	if isinstance(timesteps, list):
		timelist = True
	else:
		timelist = False

	i = 0

	timebalance = 0
	timebalance_list = []
	learned_list = []
	datastream_store = []
	raw_features_list = []
	predict_list = []
	true_list = []
	timestamp_list = []


	for dp, label in datastream:

		timestamp_list.append(time.time())
		raw_dp = list(dp) if not isinstance(dp, dict) else list(dp.values())
		raw_features_list.append(raw_dp)


		if convdict:
			dp = dict(enumerate(dp))

		if timelist:
			nexttime = timesteps[i % len(timesteps)]
		else:
			nexttime = timesteps

		timebalance += nexttime
		i += 1

		start = time.time()
		if timebalance >= 0:
			alg.learn_one(dp)

		stop = time.time()
		learned = timebalance >= 0
		timebalance = timebalance + start - stop
		done = not hasUpdate
		while timebalance >= 0 and not done:
			start = time.time()
			done = alg.update_one(final=False)
			stop = time.time()
			timebalance += start - stop

		if not done:
			start = time.time()
			done = alg.update_one(final=False)
			stop = time.time()
			timebalance += start - stop

		if done:
			timebalance = min(0, timebalance)

		timebalance_list.append(timebalance)
		learned_list.append(learned)
		datastream_store.append((dp, label))

		if (i + 1) % eval_timestep == 0:
			for j in range(len(datastream_store)):
				dp, label = datastream_store[j]
				timebalance = timebalance_list[j]
				learned = learned_list[j]
				pred = alg.predict_one(dp)
				predict_list.append(pred if pred is not None else -1)
				true_list.append(label)
				correct = pred == label
				print((i-len(datastream_store)+j), dp, pred, label, timebalance, learned, correct)

			ari = adjusted_rand_score(true_list, predict_list)

			ami = adjusted_mutual_info_score(true_list, predict_list)

			tl = np.array(true_list, dtype=int)
			pl = np.array(predict_list, dtype=int)
			acc = unsupervised_clustering_accuracy(tl, pl)


			features_np = np.array(raw_features_list[-len(predict_list):])
			labels_np = np.array(predict_list)
			t = np.array(timestamp_list[-len(labels_np):])

			if len(set(labels_np)) > 1:
				_, _, ts_score = tempsil(t,features_np, labels_np)
			else:
				ts_score = float('nan')


			print(f"[{i + 1}] Window, ARI: {ari}, AMI: {ami}, ACC: {acc}, TS: {ts_score}")


			datastream_store = []
			timebalance_list = []
			learned_list = []
			true_list = []
			predict_list = []
			raw_features_list = raw_features_list[-len(predict_list):]




if __name__ == '__main__':
	args = parse_args()
	MAX_ROWS = 5000

	if args.data.lower()=="synth":

		DATA_FILE1 = 'RBF3_40000.csv'
		with open(DATA_FILE1, newline='') as fp1:
			reader = csv.DictReader(fp1)
			x, y = [], []
			for row in itertools.islice(reader,MAX_ROWS	):
				raw_lbl = row.pop('class')
				try:
					lbl = int(raw_lbl)
				except ValueError:
					lbl = -1
				y.append(lbl)
				x.append({k: float(v) for k, v in row.items()})
		datastream = iter(zip(x, y))
		clu_num = len(set(y))


	elif args.data.lower() == "real":
		DATA_FILE2 ='fert_vs_gdp.arff'
		with open(DATA_FILE2) as fp2:
			for line in fp2:
				if line.startswith('@data'):
					break
			reader = csv.DictReader(fp2, fieldnames=['children_per_women', 'GDP_per_capita', 'class'])
			x, y = [], []
			for row in itertools.islice(reader, MAX_ROWS):
				raw_lbl = row.pop('class')
				try:
					lbl = int(raw_lbl)
				except ValueError:
					lbl = -1
				y.append(lbl)
				x.append({k: float(v) for k, v in row.items()})

		datastream = iter(zip(x, y))
		clu_num = len(set(y))

	elif args.data.lower() == "complex":
		DATA_FILE2 ='complex8.arff'
		with open(DATA_FILE2) as fp2:
			for line in fp2:
				if line.startswith('@DATA'):
					break
			reader = csv.DictReader(fp2, fieldnames=['xval', 'yval', 'class'])
			x, y = [], []
			for row in itertools.islice(reader, MAX_ROWS):
				raw_lbl = row.pop('class')
				try:
					lbl = int(raw_lbl)
				except ValueError:
					lbl = -1
				y.append(lbl)
				x.append({k: float(v) for k, v in row.items()})

		datastream = iter(zip(x, y))
		clu_num = len(set(y))

	else:
		raise ValueError(f"Unknown dataset: {args.data}")

	alg_map = {
		"CluStream": clustream.CluStream(n_macro_clusters=clu_num),
		"DBSTREAM": dbstream.DBSTREAM(clustering_threshold=0.2),
		"DenStream": denstream.DenStream(epsilon=0.05, beta=0.5, mu=5, decaying_factor=0.01, n_samples_init=500, stream_speed=100),
		"STREAMKMeans": streamkmeans.STREAMKMeans(n_clusters=clu_num, chunk_size=500),
		"ClusTreeAgg": ClusTree(use_aggregation=True),
		"ClusTreeNoAgg": ClusTree(use_aggregation=False	)
	}
	algorithm = alg_map[args.algorithm]

	main(alg=algorithm, datastream=datastream, timesteps=args.timestep, eval_timestep=args.eval_timestep, convdict = args.convdict)
	# timesteps = [0.0001, 0.001, 0.01],
	# main(alg = clustream.CluStream(n_macro_clusters=synth_clu_num), datastream=synth, timesteps = [0.0001, 0.001, 0.01], eval_timestep=1000, convdict = False)
	# main(alg = dbstream.DBSTREAM(clustering_threshold=0.2), datastream=synth, timesteps=[0.0001, 0.001, 0.01], eval_timestep=1000, convdict=False)
	# main(alg=denstream.DenStream(decaying_factor=0.01, beta=0.5, mu=5, epsilon=0.05, n_samples_init=500, stream_speed=100), datastream=synth, timesteps=[0.0001],eval_timestep=1000, convdict=False)
	# main(alg=streamkmeans.STREAMKMeans(n_clusters=synth_clu_num, chunk_size=500), datastream=synth, timesteps=[0.0001, 0.001, 0.01],eval_timestep=1000, convdict=False)
	# main(alg=ClusTree(use_aggregation=False), datastream=synth,timesteps=[0.0001], eval_timestep=1000, convdict=False)
