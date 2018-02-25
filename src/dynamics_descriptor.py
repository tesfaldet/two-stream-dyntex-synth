from layers import gramian
import tensorflow as tf


class DynamicsDescriptor(object):

    def __init__(self, name, prefix, input, model):
        """
        :param input: A 5D-tensor of shape [batch_size, 2, H, W, 1]
                [0, :, :, :] holds the target dynamic texture,
                [1:, :, :, :] holds synthesized dynamic textures
        """
        self.name = name
        self.full_name = prefix + '/' + name

        # TODO: cache target grammians

        with open(model, mode='rb') as f:
            file_content = f.read()

        graph_def = tf.GraphDef()
        graph_def.ParseFromString(file_content)

        # 'pack' is the name of the input for the imported MSOEnet graph
        tf.import_graph_def(graph_def, input_map={'input': input},
                            name=self.name)

    def get_output(self):
        return tf.get_default_graph() \
                 .get_tensor_by_name('{0}/MSOEmultiscale/reshape/Reshape:0'
                                     .format(self.full_name))

    def gramian_for_layer(self, layer):
        """
        Returns a matrix of cross-correlations between the activations of
        convolutional channels in a given layer.
        """
        activations = self.activations_for_layer(layer)

        # drop depth because it'll always be a singleton dimension
        activations = tf.squeeze(activations, axis=[1])

        # Reshape from (batch, height, width, channels) to
        # (batch, channels, height, width)
        shuffled_activations = tf.transpose(activations, perm=[0, 3, 1, 2])
        return gramian(shuffled_activations, normalize_method='ulyanov')

    def activations_for_layer(self, layer):
        return tf.get_default_graph() \
                 .get_tensor_by_name('{0}/MSOEmultiscale/{1}:0'
                                     .format(self.full_name, layer))
