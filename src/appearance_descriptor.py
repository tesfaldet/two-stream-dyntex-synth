from layers import gramian
import tensorflow as tf


class AppearanceDescriptor(object):

    def __init__(self, name, prefix, input):
        """
        :param input: A 4D-tensor of shape [batchSize, H, W, 3]
                [0, :, :, :] holds the target static texture
                [1:1+j, :, :, :] holds synthesized static textures
        """
        self.name = name
        self.full_name = prefix + '/' + name

        # TODO: cache target grammians

        with open('models/vgg19_normalized.tfmodel', mode='rb') as f:
            file_content = f.read()

        graph_def = tf.GraphDef()
        graph_def.ParseFromString(file_content)

        # 'images' is the name of the input for the imported vgg graph
        tf.import_graph_def(graph_def, input_map={'images': input},
                            name=self.name)

    def gramian_for_layer(self, layer):
        """
        Returns a matrix of cross-correlations between the activations of
        convolutional channels in a given layer.
        """
        activations = self.activations_for_layer(layer)

        # Reshape from (batch, height, width, channels) to
        # (batch, channels, height, width)
        shuffled_activations = tf.transpose(activations, perm=[0, 3, 1, 2])
        return gramian(shuffled_activations, normalize_method='ulyanov')

    def activations_for_layer(self, layer):
        """
        :param layer: A tuple that indexes into the convolutional blocks of
        the VGG Net
        """
        return tf.get_default_graph() \
                 .get_tensor_by_name('{0}/{1}:0'
                                     .format(self.full_name, layer))
