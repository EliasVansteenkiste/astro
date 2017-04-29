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
p_transform = {'patch_size': (64, 64),
               'channels': 2,
               'n_labels': 3}


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
    x = data_transforms.ch_norm_center(x)
    x = data_transforms.perturb(x, p_augmentation, p_transform['patch_size'], rng)
    return x

def data_prep_function_valid(x, p_transform=p_transform, **kwargs):
    #take a patch in the middle of the chip
    x = data_transforms.ch_norm_center(x)

    d1_r = p_transform['patch_size'][0]/2
    d2_r = p_transform['patch_size'][1]/2

    d1 = slice(x.shape[1]/2-d1_r,x.shape[1]/2+d1_r,1)
    d2 = slice(x.shape[2]/2-d2_r,x.shape[2]/2+d2_r,1)
    x = x[:,d1,d2]
    return x


# data iterators
batch_size = 32
nbatches_chunk = 1
chunk_size = batch_size * nbatches_chunk


all_ids = app.temporary_get_img_ids()
folds = app.make_random_split(all_ids, no_folds=3)
print len(folds)
train_ids = folds[0] + folds[1]
valid_ids = folds[2]

bad_ids = app.get_bad_img_ids() + [1237665532796272810, 1237667106885665484]
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
max_nchunks = nchunks_per_epoch * 20

validate_every = int(1. * nchunks_per_epoch)
save_every = int(5. * nchunks_per_epoch)

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



def build_model(l_in=None):
    l_in = nn.layers.InputLayer((None, p_transform['channels'],) + p_transform['patch_size']) if l_in is None else l_in
    l_target = nn.layers.InputLayer((None,p_transform['n_labels']))

    l = conv(l_in, 64)
    l = conv(l, 64)
    l = max_pool(l)

    l = conv(l, 64)
    l = conv(l, 64)
    l = max_pool(l)

    l = conv(l, 64)
    l = conv(l, 64)
    l = max_pool(l)

    l = conv(l, 64)
    l = conv(l, 64)
    l = max_pool(l)

    l = conv(l, 64)
    l = conv(l, 64)
    l = max_pool(l)

    l = dense(drop(l), 128)

    l_out = nn.layers.DenseLayer(l, num_units=1,
                                 W=nn.init.Constant(0.),
                                 nonlinearity=nn.nonlinearities.softplus)

    return namedtuple('Model', ['l_in', 'l_out', 'l_target'])(l_in, l_out, l_target)


# def build_objective(model, deterministic=False):
#     predictions = nn.layers.get_output(model.l_out, deterministic=deterministic)
#     targets = nn.layers.get_output(model.l_target)[:,0]
#     errors = nn.layers.get_output(model.l_target)[:,1]
    
#     common_exp = 10.5
#     qpred = 2.*predictions - errors - common_exp
#     qtarg = 2.*targets - errors - common_exp
#     mix_pred_targ = targets + predictions - errors - common_exp

#     objectives = 10.**qpred + 10.**qtarg -2.*10.**mix_pred_targ
#     objective = T.mean(objectives)
#     return objective


def build_objective(model, deterministic=False):
    predictions = nn.layers.get_output(model.l_out, deterministic=deterministic)
    targets = nn.layers.get_output(model.l_target)[:,0]
    errors = nn.layers.get_output(model.l_target)[:,1]

    objectives = lasagne.objectives.squared_error(predictions,targets)
    ojbective = lasagne.objectives.aggregate(objectives, weights=10**errors, mode='mean')
    return ojbective

def build_updates(train_loss, model, learning_rate):
    updates = nn.updates.adam(train_loss, nn.layers.get_all_params(model.l_out, trainable=True), learning_rate)
    return updates
