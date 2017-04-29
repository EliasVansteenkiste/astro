import lasagne as nn
import theano.tensor as T
import numpy as np
from lasagne import nonlinearities




class MassLayer(nn.layers.MergeLayer):
    """
    takes elementwise product between 2 layers
    """

    def __init__(self, input1, input2, **kwargs):
        super(MassLayer, self).__init__([input1, input2], **kwargs)

    def get_output_shape_for(self, input_shapes):
        assert(len(input_shapes[1])==2)
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        return inputs[0] + T.log10(inputs[1][:,2:3])

class LogDistLayer(nn.layers.Layer):
    def __init__(self, incoming, **kwargs):
        super(LogDistLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        assert(input_shape[1]==3)
        assert(len(input_shape)==2)
        return (input_shape[0], 1)

    def get_output_for(self, input, **kwargs):
        result = T.log(input[:,2:3])
        return result
