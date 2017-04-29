import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage import io
import os

import pathfinder

rng = np.random.RandomState(37145)

def _test_read_image():
    img_ids = temporary_get_img_ids()
    for img_id in img_ids:
        print read_image('train', img_id).shape

def read_image(dataset, id, n_channels = 2):
    if dataset == 'train':
        prefix = 'train-csv/'
    elif dataset == 'test':
        prefix = 'test-csv/'
    else:
        raise
    csv_g_path = pathfinder.DATA_PATH + prefix + str(id) + '-g.csv'
    csv_i_path = pathfinder.DATA_PATH + prefix + str(id) + '-i.csv'
    g_data = pd.read_csv(csv_g_path)
    
    image = np.zeros((n_channels, g_data.shape[0],g_data.shape[1]),dtype=np.float32)
    image[0] = g_data
    if n_channels==2:
        i_data = pd.read_csv(csv_i_path)
        if i_data.shape != g_data.shape:
            print 'img id', id, 'i_data.shape', i_data.shape, 'g_data.shape', g_data.shape
        image[1] = i_data
    return image

def read_image_fast(dataset, id, n_channels = 2):

    image_path = pathfinder.COMPRESSED_DATA_PATH + str(id) + '.npz'
    image = np.load(image_path)['arr_0']
    works = id

    return image

def get_pd_labels(dataset = 'train'):
    df = pd.read_csv(pathfinder.DATA_PATH+dataset+'.csv', sep=';')
    return df

def get_d_labels():
    d_label =  get_pd_labels().set_index('SDSS_ID').T.to_dict('list')
    return d_label

def _test_get_labels():
    print get_pd_labels('train').describe()

def temporary_get_img_ids():
    filenames =  os.listdir(pathfinder.DATA_PATH+'train-csv')
    img_ids = [int(x.split('.')[0].split('-')[0]) for x in filenames]
    img_ids = list(set(img_ids))

    return img_ids

def make_random_split(img_ids,no_folds=3):
    chunks = chunkIt(img_ids,no_folds)
    return chunks

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


def get_bad_img_ids():
    bad_img_ids = []
    for line in open('bad_img_ids.lst') :
        bad_img_ids.append(int(line.split(';')[0]))
    for line in open('bad_img_ids2.lst') :
        bad_img_ids.append(int(line.split(';')[0]))

    return bad_img_ids



if __name__ == "__main__":
    bad_img_ids = get_bad_img_ids()
    d_labels = get_d_labels()

    masses = []
    errors = []
    distances = []
    for img_id, lv in d_labels.iteritems():
        if img_id not in bad_img_ids:
            masses.append(lv[0])
            errors.append(lv[1])
            distances.append(lv[2])

    print 'mass', np.amin(masses), np.amax(masses), np.average(masses)
    print 'errors', np.amin(errors), np.amax(errors), np.average(errors)
    print 'distances', np.amin(distances), np.amax(distances), np.average(distances)





