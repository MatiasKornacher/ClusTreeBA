import itertools
import time
import csv

import river.cluster.clustream as clustream
import river.cluster.dbstream as dbstream
import river.cluster.denstream as denstream
import river.cluster.streamkmeans as streamkmeans

from ClusTree import ClusTree as clustree, Node


def main(alg, datastream, timesteps, eval_timestep, convdict=True):

	if alg.__class__ in [clustream, dbstream, denstream, clustree]:
		hasUpdate = True
	else:#only streamkmeans
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
			done = alg.update_one()
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
				# TODO evaluation
				correct = pred == label
				print((i-len(datastream_store)+j), pred, label, timebalance, learned, correct)

			datastream_store = []
			timebalance_list = []
			learned_list = []



if __name__ == '__main__':
	DATA_FILE = 'RBF3_40k.csv'
	with open(DATA_FILE, newline='') as fp:
		reader = csv.DictReader(fp)
		x, y = [], []
		for row in itertools.islice(reader,5000):
			label = row.pop('class')
			y.append(int(label))
			x.append({k: float(v) for k, v in row.items()})

	datastream = iter(zip(x, y))
	clu_num = len(set(y))

	main(alg = clustream.CluStream(n_macro_clusters=clu_num), datastream=datastream, timesteps = [0.0001, 0.001, 0.01], eval_timestep=1000)