import numpy as np
import lasagne as nn
from collections import namedtuple
from functools import partial
import lasagne.layers.dnn as dnn
import lasagne
import theano.tensor as T

import data_transforms
import data_iterators
import pathfinder
import utils
import app

restart_from_save = None
rng = np.random.RandomState(42)

# transformations
p_transform = {'patch_size': (128, 128),
               'channels': 4,
               'n_labels': 17}


p_augmentation = {
    'zoom_range': (1 / 1.1, 1.1),
    'rotation_range': (0, 360),
    'shear_range': (0, 0),
    'translation_range': (-4, 4),
    'do_flip': True,
    'allow_stretch': False,
}


channel_zmuv_stats = {
    'avg': [4970.55, 4245.35, 3064.64, 6360.08],
    'std': [1785.79, 1576.31, 1661.19, 1841.09]}

# data preparation function
def data_prep_function_train(x, p_transform=p_transform, p_augmentation=p_augmentation, **kwargs):
    x = data_transforms.perturb(x, p_augmentation, p_transform['patch_size'], rng)
    x = data_transforms.channel_zmuv(x, img_stats = channel_zmuv_stats, no_channels=4)
    return x

def data_prep_function_valid(x, p_transform=p_transform, **kwargs):
    #take a patch in the middle of the chip
    x = x[:,64:192,64:192]
    x = data_transforms.channel_zmuv(x, img_stats = channel_zmuv_stats, no_channels=4)
    return x


# data iterators
batch_size = 32
nbatches_chunk = 1
chunk_size = batch_size * nbatches_chunk

folds = app.make_stratified_split(no_folds=5)
print len(folds)
train_ids = folds[0] + folds[1] + folds[2] + folds[3]
valid_ids = folds[4]

bad_ids = [18772]
print (18772 in valid_ids)

train_ids = [x for x in train_ids if x not in bad_ids]
valid_ids = [x for x in valid_ids if x not in bad_ids]


train_data_iterator = data_iterators.DataGenerator(dataset='train',
                                                    batch_size=chunk_size,
                                                    img_ids = train_ids,
                                                    p_transform=p_transform,
                                                    data_prep_fun = data_prep_function_train,
                                                    rng=rng,
                                                    full_batch=True, random=True, infinite=True)

valid_data_iterator = data_iterators.DataGenerator(dataset='train',
                                                    batch_size=chunk_size,
                                                    img_ids = valid_ids,
                                                    p_transform=p_transform,
                                                    data_prep_fun = data_prep_function_valid,
                                                    rng=rng,
                                                    full_batch=False, random=False, infinite=False)

nchunks_per_epoch = train_data_iterator.nsamples / chunk_size
max_nchunks = nchunks_per_epoch * 100

validate_every = int(1. * nchunks_per_epoch)
save_every = int(1. * nchunks_per_epoch)

learning_rate_schedule = {
    0: 5e-4,
    int(max_nchunks * 0.5): 2e-4,
    int(max_nchunks * 0.6): 1e-4,
    int(max_nchunks * 0.7): 5e-5,
    int(max_nchunks * 0.8): 2e-5,
    int(max_nchunks * 0.9): 1e-5
}

# model
conv = partial(dnn.Conv2DDNNLayer,
                 filter_size=3,
                 pad='same',
                 W=nn.init.Orthogonal(),
                 nonlinearity=nn.nonlinearities.very_leaky_rectify)

max_pool = partial(dnn.MaxPool2DDNNLayer,
                     pool_size=2)

drop = lasagne.layers.DropoutLayer

dense = partial(lasagne.layers.DenseLayer,
                W=lasagne.init.Orthogonal(),
                nonlinearity=lasagne.nonlinearities.very_leaky_rectify)


def inrn_v2(lin):
    n_base_filter = 32

    l1 = conv(lin, n_base_filter, filter_size=1)

    l2 = conv(lin, n_base_filter, filter_size=1)
    l2 = conv(l2, n_base_filter, filter_size=3)

    l3 = conv(lin, n_base_filter, filter_size=1)
    l3 = conv(l3, n_base_filter, filter_size=3)
    l3 = conv(l3, n_base_filter, filter_size=3)

    l = lasagne.layers.ConcatLayer([l1, l2, l3])

    l = conv(l, lin.output_shape[1], filter_size=1)

    l = lasagne.layers.ElemwiseSumLayer([l, lin])

    l = lasagne.layers.NonlinearityLayer(l, nonlinearity=lasagne.nonlinearities.rectify)

    return l


def inrn_v2_red(lin):
    # We want to reduce our total volume /4

    den = 16
    nom2 = 4
    nom3 = 5
    nom4 = 7

    ins = lin.output_shape[1]

    l1 = max_pool(lin)

    l2 = conv(lin, ins // den * nom2, filter_size=3, stride=2)

    l3 = conv(lin, ins // den * nom2, filter_size=1)
    l3 = conv(l3, ins // den * nom3, filter_size=3, stride=2)

    l4 = conv(lin, ins // den * nom2, filter_size=1)
    l4 = conv(l4, ins // den * nom3, filter_size=3)
    l4 = conv(l4, ins // den * nom4, filter_size=3, stride=2)

    l = lasagne.layers.ConcatLayer([l1, l2, l3, l4])

    return l


def feat_red(lin):
    # We want to reduce the feature maps by a factor of 2
    ins = lin.output_shape[1]
    l = conv(lin, ins // 2, filter_size=1)
    return l


def build_model(l_in=None):
    l_in = nn.layers.InputLayer((None, p_transform['channels'],) + p_transform['patch_size']) if l_in is None else l_in
    l_target = nn.layers.InputLayer((None,p_transform['n_labels']))

    l = conv(l_in, 64)
    l = inrn_v2_red(l)
    l = inrn_v2(l)
    l = feat_red(l)
    l = inrn_v2(l)

    l = inrn_v2_red(l)
    l = inrn_v2(l)
    l = feat_red(l)
    l = inrn_v2(l)

    l = feat_red(l)

    l = dense(drop(l), 128)

    l_out = nn.layers.DenseLayer(l, num_units=p_transform['n_labels'],
                                 W=nn.init.Constant(0.),
                                 nonlinearity=nn.nonlinearities.sigmoid)

    return namedtuple('Model', ['l_in', 'l_out', 'l_target'])(l_in, l_out, l_target)


def build_objective(model, deterministic=False):
    predictions = nn.layers.get_output(model.l_out, deterministic=deterministic)
    targets = T.flatten(nn.layers.get_output(model.l_target))
    objectives = lasagne.objectives.squared_error(predictions,targets)
    ojbective = lasagne.objectives.aggregate(objectives, weights=None, mode='mean')
    return ojbective


def build_updates(train_loss, model, learning_rate):
    updates = nn.updates.adam(train_loss, nn.layers.get_all_params(model.l_out, trainable=True), learning_rate)
    return updates
