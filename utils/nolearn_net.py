import numpy as np
import lasagne as nn
import theano
from nolearn.lasagne import NeuralNet as BaseNeuralNet


class NeuralNet(BaseNeuralNet):
    def transform(self, X, target_layer_name, y=None):
        target_layer = self.layers_[target_layer_name]

        layers = self.layers_
        input_layers = [
            layer for layer in layers.values()
            if isinstance(layer, nn.layers.InputLayer)
        ]
        X_inputs = [
            theano.Param(input_layer.input_var, name=input_layer.name)
            for input_layer in input_layers
        ]

        target_layer_output = nn.layers.get_output(
            target_layer, None, deterministic=True
        )

        transform_iter = theano.function(
            inputs=X_inputs,
            outputs=target_layer_output,
            allow_input_downcast=True,
        )

        outputs = []
        for Xb, yb in self.batch_iterator_test(X):
            outputs.append(self.apply_batch_func(transform_iter, Xb))
        return np.vstack(outputs)
