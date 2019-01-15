from keras import initializers, regularizers, activations
from keras.engine.topology import Layer
import tensorflow as tf


class GCNGraphConv(Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""

    def __init__(self,
                 filters,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.filters = filters

        super(GCNGraphConv, self).__init__(**kwargs)

    def build(self, input_shape):
        num_features = input_shape[0][-1]
        self.weight = self.add_weight(shape=(num_features, self.filters),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      name='w')

        self.bias = self.add_weight(shape=(self.filters,),
                                    name='b',
                                    initializer=self.bias_initializer)

        super(GCNGraphConv, self).build(input_shape)

    def call(self, inputs, mask=None):
        # Import graph tensors
        # features = (samples, max_atoms, features)
        # adjacency = (samples, max_atoms, max_atoms)
        features, adjacency = inputs

        # Get parameters
        max_atoms = int(features.shape[1])
        num_features = int(features.shape[-1])

        # Calculate AX
        features = tf.matmul(adjacency, features)

        # Calculate (AX)W
        features = tf.reshape(features, [-1, num_features])
        features = tf.matmul(features, self.weight) + self.bias
        features = tf.reshape(features, [-1, max_atoms, self.filters])

        # Activation
        features = self.activation(features)

        return features

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self.filters


class GCNGraphGather(Layer):
    def __init__(self,
                 pooling="sum",
                 activation=None,
                 **kwargs):
        self.activation = activations.get(activation)
        self.pooling = pooling

        super(GCNGraphGather, self).__init__(**kwargs)

    def build(self, inputs_shape):
        super(GCNGraphGather, self).build(inputs_shape)

    def call(self, inputs, mask=None):
        # Import graph tensors
        # features = (samples, max_atoms, features)
        features = inputs

        # Integrate over atom axis
        if self.pooling == "sum":
            features = tf.reduce_sum(features, axis=1)
        elif self.pooling == "max":
            features = tf.reduce_max(features, axis=1)

            # Activation
            features = self.activation(features)

        return features

    def compute_output_shape(self, inputs_shape):
        return inputs_shape[0], inputs_shape[-1]
