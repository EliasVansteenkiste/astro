import numpy as np
import lasagne as nn
from collections import namedtuple
from functools import partial
import lasagne.layers.dnn as dnn
import lasagne
from theano import printing, theano
import theano.tensor as T

import data_transforms
import data_iterators
import pathfinder
import theano_printer
import utils
import app

restart_from_save = None
rng = np.random.RandomState(42)

# transformations
p_transform = {'patch_size': (64, 64),
               'channels': 2,
               'n_labels': 3}


p_augmentation = {
    'zoom_range': (1.,1.),
    'rotation_range': (0, 360),
    'shear_range': (0, 0),
    'translation_range': (0, 0),
    'do_flip': True,
    'allow_stretch': False,
}


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
batch_size = 1
nbatches_chunk = 32
chunk_size = batch_size * nbatches_chunk


all_ids = app.temporary_get_img_ids()
folds = app.make_random_split(all_ids, no_folds=3)
print len(folds)
train_ids = folds[0] + folds[1]
valid_ids = folds[2]

bad_ids = app.get_bad_img_ids() + [1237665532796272810, 1237667106885665484, 1237651755092148341, 1237660634909835612, 1237679167157895402, 1237661083199079644, 1237652946384257075, 1237679167157895402, 
1237661059574464751,
1237667106885665484,
1237661435384365240,
1237667912749678730,
1237664853650178099,
1237656538053083251,
1237662337871577447,
1237667255063412964,
1237648705137934473,
1237652936184102946,
1237673705576202561, 1237652947995066417]+[
   1237664286172053681
]

train_ids = [x for cnt,x in enumerate(train_ids) if x not in bad_ids]
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

nchunks_per_epoch = train_data_iterator.nsamples / chunk_size if train_data_iterator.nsamples > chunk_size else 1
max_nchunks = 25 * nchunks_per_epoch

validate_every = int(5 * nchunks_per_epoch)
save_every = int(5 * nchunks_per_epoch)

learning_rate_schedule = {
    0:    0.001,
    5 * nchunks_per_epoch:    0.003,
    10 * nchunks_per_epoch:    0.001,
    15 * nchunks_per_epoch:    0.0003,
    20 * nchunks_per_epoch:    0.0001,
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

    # l = drop(l)

    l = dense(l, 128)

    l_out = nn.layers.DenseLayer(l, num_units=1,
                                 W=nn.init.Constant(0.),b=nn.init.Constant(3.5),
                                 nonlinearity=nn.nonlinearities.identity)

    return namedtuple('Model', ['l_in', 'l_out', 'l_target'])(l_in, l_out, l_target)


def build_objective(model, deterministic=False):
    predictions = nn.layers.get_output(model.l_out, deterministic=deterministic)
    targets = nn.layers.get_output(model.l_target)[:,0]
    errors = nn.layers.get_output(model.l_target)[:,1]
    distance = nn.layers.get_output(model.l_target)[:,2]

    predictions += 3*T.log(distance) / np.float32(np.log(10))


    ten_pt = 10.**(predictions-targets)
    objectives = (1 - ten_pt)**2 / (np.float32(np.log(10.))*errors)**2
    objective = T.mean(objectives)
    # theano_printer.print_me_this('err', T.mean(predictions-targets))
    # theano_printer.print_me_this('targets', nn.layers.get_output(model.l_target))
    # theano_printer.print_me_this('errors', errors)
    # theano_printer.print_me_this('objective', objective)
    return objective


def fixed_norm_constraint(tensor_vars, max_norm, epsilon=0,
                          return_norm=False):
    norm = T.sqrt(sum(T.sum(tensor**2) for tensor in tensor_vars))
    dtype = np.dtype(theano.config.floatX).type
    target_norm = max_norm
    multiplier = norm / target_norm
    tensor_vars_scaled = [step/multiplier for step in tensor_vars]

    if return_norm:
        return tensor_vars_scaled, norm
    else:
        return tensor_vars_scaled

def build_updates(train_loss, model, learning_rate):
    if True:
        grad, grad_norm = fixed_norm_constraint(theano.grad(train_loss, nn.layers.get_all_params(model.l_out, trainable=True)), max_norm=1.0,return_norm=True )
        # theano_printer.print_me_this('grad', grad_norm)
    else:
        grad = theano.grad(train_loss, nn.layers.get_all_params(model.l_out, trainable=True))
    updates = nn.updates.nesterov_momentum(grad, nn.layers.get_all_params(model.l_out, trainable=True), learning_rate)
    return updates
