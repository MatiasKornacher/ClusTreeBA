import itertools
import time
import csv

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

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
			alg.update_one(final=True)

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
			print(f"[{i + 1}] Window, ARI: {ari}, AMI: {ami}")


			datastream_store = []
			timebalance_list = []
			learned_list = []
			true_list = []
			predict_list = []




if __name__ == '__main__':
	DATA_FILE = 'RBF3_40000.csv'
	with open(DATA_FILE, newline='') as fp:
		reader = csv.DictReader(fp)
		x, y = [], []
		for row in itertools.islice(reader,100):
			label = row.pop('class')
			try:
				lbl = int(label)
			except ValueError:
				lbl = -1
			y.append(lbl)
			x.append({k: float(v) for k, v in row.items()})

	datastream = iter(zip(x, y))
	clu_num = len(set(y))

	# main(alg = clustream.CluStream(n_macro_clusters=clu_num), datastream=datastream, timesteps = [0.0001, 0.001, 0.01], eval_timestep=1000, convdict = False)
	# main(alg = dbstream.DBSTREAM(clustering_threshold=0.2), datastream=datastream, timesteps=[0.0001, 0.001, 0.01], eval_timestep=1000, convdict=False)
	# main(alg=denstream.DenStream(decaying_factor=0.01, beta=0.5, mu=5, epsilon=0.05, n_samples_init=500, stream_speed=100), datastream=datastream, timesteps=[0.0001, 0.001, 0.01],eval_timestep=1000, convdict=False)
	# main(alg=streamkmeans.STREAMKMeans(n_clusters=clu_num, chunk_size=500), datastream=datastream, timesteps=[0.0001, 0.001, 0.01],eval_timestep=1000, convdict=False)
	main(alg=ClusTree(), datastream=datastream,timesteps=[0.0001, 0.001, 0.01], eval_timestep=10, convdict=False)