import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage import io

import pathfinder

rng = np.random.RandomState(37145)

def _test_read_image():
    print read_image('train', 1237648702966726812).shape

def read_image(dataset, id):
    if dataset == 'train':
        prefix = 'train-csv/train/'
    elif dataset == 'test':
        prefix = 'test-tif/test_'
    else:
        raise
    csv_g_path = pathfinder.DATA_PATH + prefix + str(id) + '-g.csv'
    csv_i_path = pathfinder.DATA_PATH + prefix + str(id) + '-i.csv'
    i_data = pd.read_csv(csv_i_path)
    g_data = pd.read_csv(csv_g_path)
    # image = io.imread(path)
    # image = np.swapaxes(image,0,2)
    image = np.zeros((2, i_data.shape[0],i_data.shape[1]),dtype=np.float32)
    image[0] = i_data
    image[1] = g_data
    return image


def get_labels():
	df_train = pd.read_csv(pathfinder.DATA_PATH+'train.csv')
	df_train = pd.concat([df_train['image_name'], df_train.tags.str.get_dummies(sep=' ')], axis=1)
	return df_train

def get_labels_array():
	df = get_labels()
	only_labels = df.drop(['image_name'], axis = 1, inplace = False)
	only_labels = only_labels.as_matrix()
	return only_labels
	
def chunkIt(seq, num):
  avg = len(seq) / float(num)
  out = []
  last = 0.0

  while last < len(seq):
    out.append(seq[int(last):int(last + avg)])
    last += avg

  return out

def top_occ(feat_comb, n_top = 5):
	# a method for printing top occurences of feature combinations
	# built for checking if split is stratified
	n_total_samples = len(feat_comb)
	feat_comb_occ = np.bincount(feat_comb)
	top = feat_comb_occ.argsort()[-n_top:][::-1]
	for idx, fc in enumerate(top):
		print idx, fc, 1.0*feat_comb_occ[fc]/n_total_samples
	print 'checksum', sum(feat_comb)

def make_stratified_split(no_folds=5, verbose=False):
	df = get_labels()
	only_labels = df.drop(['image_name'], axis = 1, inplace = False)
	only_labels = only_labels.as_matrix()
	if verbose: print 'labels shape', only_labels.shape
	feat_comb = only_labels.dot(1 << np.arange(only_labels.shape[-1] - 1, -1, -1))
	feat_comb_set = set(feat_comb)
	feat_comb_occ = np.bincount(feat_comb)
	feat_comb_high = np.where(feat_comb_occ >= no_folds)[0]
	n_total_samples = 0
	folds = [[] for _ in range(no_folds)]
	for fc in feat_comb_high:
		idcs = np.where(feat_comb == fc)[0]
		chunks = chunkIt(idcs,no_folds)
		# print len(idcs), [len(chunk) for chunk in chunks]
		rng.shuffle(chunks)
		for idx, chunk in enumerate(chunks):
			folds[idx].extend(chunk)

	feat_comb_low = np.where(np.logical_and(feat_comb_occ < no_folds, feat_comb_occ > 0))[0]
	low_idcs = []
	for fc in feat_comb_low:
		idcs = np.where(feat_comb == fc)[0]
		low_idcs.extend(idcs)

	chunks = chunkIt(low_idcs,no_folds)
	rng.shuffle(chunks)
	for idx, chunk in enumerate(chunks):
		folds[idx].extend(chunk)


	n_samples_fold = 0
	for f in folds:
		n_samples_fold += len(f)

	if verbose:
		print 'n_samples_fold', n_samples_fold
		top_occ(feat_comb)
		for f in folds:
			top_occ(feat_comb[f])

	return folds





if __name__ == "__main__":
    _test_read_image()
    #make_stratified_split(verbose=True)

