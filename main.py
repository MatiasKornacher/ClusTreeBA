import itertools
import time
import csv

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

import numpy as np
from clustpy.metrics.clustering_metrics import unsupervised_clustering_accuracy

import river.cluster.clustream as clustream
import river.cluster.dbstream as dbstream
import river.cluster.denstream as denstream
import river.cluster.streamkmeans as streamkmeans

from ClusTree import ClusTree


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
	predict_list = []
	true_list = []


	for dp, label in datastream:
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
				print((i-len(datastream_store)+j), pred, label, timebalance, learned, correct)

			ari = adjusted_rand_score(true_list, predict_list)
			ami = adjusted_mutual_info_score(true_list, predict_list)
			tl = np.array(true_list, dtype=int)
			pl = np.array(predict_list, dtype=int)
			acc = unsupervised_clustering_accuracy(tl, pl)
			print(f"[{i + 1}] Window, ARI: {ari}, AMI: {ami}, ACC: {acc}")


			datastream_store = []
			timebalance_list = []
			learned_list = []
			true_list = []
			predict_list = []




if __name__ == '__main__':

	MAX_ROWS = 5000
	DATA_FILE1 = 'RBF3_40000.csv'
	with open(DATA_FILE1, newline='') as fp1:
		reader = csv.DictReader(fp1)
		x1, y1 = [], []
		for row in itertools.islice(reader,MAX_ROWS	):
			raw_lbl = row.pop('class')
			try:
				lbl = int(raw_lbl)
			except ValueError:
				lbl = -1
			y1.append(lbl)
			x1.append({k: float(v) for k, v in row.items()})



	synth = iter(zip(x1, y1))
	synth_clu_num = len(set(y1))

	DATA_FILE2 ='fert_vs_gdp.arff'
	with open(DATA_FILE2) as fp2:
		for line in fp2:
			if line.startswith('@data'):
				break
		reader = csv.DictReader(fp2, fieldnames=['children_per_women', 'GDP_per_capita', 'class'])
		x2, y2 = [], []
		for row in itertools.islice(reader, MAX_ROWS):
			raw_lbl = row.pop('class')
			try:
				lbl = int(raw_lbl)
			except ValueError:
				lbl = -1
			y2.append(lbl)
			x2.append({k: float(v) for k, v in row.items()})

	real = iter(zip(x2, y2))
	real_clu_num = len(set(y2))




	# timesteps = [0.0001, 0.001, 0.01],
	# main(alg = clustream.CluStream(n_macro_clusters=synth_clu_num), datastream=synth, timesteps = [0.0001, 0.001, 0.01], eval_timestep=1000, convdict = False)
	# main(alg = dbstream.DBSTREAM(clustering_threshold=0.2), datastream=synth, timesteps=[0.0001, 0.001, 0.01], eval_timestep=1000, convdict=False)
	# main(alg=denstream.DenStream(decaying_factor=0.01, beta=0.5, mu=5, epsilon=0.05, n_samples_init=500, stream_speed=100), datastream=synth, timesteps=[0.0001, 0.001, 0.01],eval_timestep=1000, convdict=False)
	# main(alg=streamkmeans.STREAMKMeans(n_clusters=synth_clu_num, chunk_size=500), datastream=synth, timesteps=[0.0001, 0.001, 0.01],eval_timestep=1000, convdict=False)
	main(alg=ClusTree(), datastream=real,timesteps=[0.0001], eval_timestep=1000, convdict=False)