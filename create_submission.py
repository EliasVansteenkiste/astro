import string
import sys
import lasagne as nn
import numpy as np
import theano
import buffering
import pathfinder
import utils
from configuration import config, set_configuration
import logger

theano.config.warn_float64 = 'raise'

if len(sys.argv) < 2:
    sys.exit("Usage: test.py <configuration_name>")

config_name = sys.argv[1]
set_configuration('configs', config_name)

# metadata
metadata_dir = utils.get_dir_path('models', pathfinder.METADATA_PATH)
metadata_path = utils.find_model_metadata(metadata_dir, config_name)

metadata = utils.load_pkl(metadata_path)
expid = metadata['experiment_id']

# logs
logs_dir = utils.get_dir_path('logs', pathfinder.METADATA_PATH)
sys.stdout = logger.Logger(logs_dir + '/%s-test.log' % expid)
sys.stderr = sys.stdout

# predictions path
submissions_dir = utils.get_dir_path('submissions', pathfinder.METADATA_PATH)
utils.auto_make_dir(submissions_dir)
output_csv_file = submissions_dir + '/%s.csv' % expid

print 'Build model'
model = config().build_model()
all_layers = nn.layers.get_all_layers(model.l_out)
all_params = nn.layers.get_all_params(model.l_out)
num_params = nn.layers.count_params(model.l_out)
print '  number of parameters: %d' % num_params
print string.ljust('  layer output shapes:', 36),
print string.ljust('#params:', 10),
print 'output shape:'
for layer in all_layers:
    name = string.ljust(layer.__class__.__name__, 32)
    num_param = sum([np.prod(p.get_value().shape) for p in layer.get_params()])
    num_param = string.ljust(num_param.__str__(), 10)
    print '    %s %s %s' % (name, num_param, layer.output_shape)

nn.layers.set_all_param_values(model.l_out, metadata['param_values'])

x_shared = nn.utils.shared_empty(dim=len(model.l_in.shape))
y_shared = nn.utils.shared_empty(dim=len(model.l_target.shape))

givens_test = {}
givens_test[model.l_in.input_var] = x_shared
givens_test[model.l_target.input_var] = y_shared

# theano functions
iter_get_predictions = theano.function([], config().get_preds(model),
                                       givens=givens_test)

test_data_iterator = config().test_data_iterator

print
print 'Data'
print 'n test: %d' % test_data_iterator.nsamples

pid2prediction = {}
for n, (x_chunk, y_chunk, id_chunk) in enumerate(buffering.buffered_gen_threaded(test_data_iterator.generate())):
    # load chunk to GPU
    x_shared.set_value(x_chunk)
    y_shared.set_value(y_chunk)
    predictions = iter_get_predictions()
    for i in xrange(predictions.shape[0]):
        pid2prediction[id_chunk[i]] = predictions[i][0]

utils.write_submission(pid2prediction, submission_path=output_csv_file)
