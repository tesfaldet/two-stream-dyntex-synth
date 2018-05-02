import tensorflow as tf
from utilities import load_image, load_images, vgg_process, vgg_deprocess
from appearance_descriptor import AppearanceDescriptor
from dynamics_descriptor import DynamicsDescriptor
from optimizer import Optimizer
from layers import stop_gradients
import numpy as np


class SynthesizerMasking(Optimizer):

    # TODO: let spatiotemporal size be user-definable
    def __init__(self, target_dynamic_path, config):
        Optimizer.__init__(self, tf.Graph(), 256, 512, 3, 3,
                           target_dynamic_path, '',
                           config)

        with self.graph.as_default():
            with tf.device('/gpu:' + str(self.user_config['gpu'])):
                # load dynamic texture
                imgs = load_images(target_dynamic_path,
                                   size=(self.input_frame_count,
                                         self.input_dimension,
                                         self.input_dimension))
                self.target_dynamic_texture = [tf.to_float(
                        tf.constant(img.reshape(1, self.input_dimension,
                                                self.input_dimension, 3)))
                                               for img in imgs]

                # previously synthesized dynamic texture
                init = load_images('data/dynamic_textures/init',
                                   size=(self.input_frame_count,
                                         self.input_dimension,
                                         self.input_dimension))
                init = np.expand_dims(init, 0)
                print init.shape

                # initialize noise
                initial_noise = np.random.randn(self.user_config['batch_size'],
                                                self.output_frame_count,
                                                self.output_dimension,
                                                self.output_dimension, 3)
                print initial_noise.shape

                # prevent gradients from modifying the masked regions
                start = ((self.output_dimension - self.input_dimension)
                         / 2) - 1
                end = start + self.input_dimension
                mask = np.ones_like(initial_noise)
                mask[:, :, start:end, start:end, :] = 0.0
                mask = tf.to_float(mask, name='mask')

                # initialize center volume of noise to be a dynamic texture
                initial_noise[:, :, start:end, start:end, :] = init[...]

                self.output = tf.Variable(tf.to_float(initial_noise),
                                          name='output')
                # self.output = stop_gradients('gradient_mask', self.output,
                                             # mask)

                # dts weights, app: 1e9, dyn: 1e15
                # dts weights (flow decode layer), app: 1e9, dyn: 1e3
                # TODO: let weight be user-definable
                # build appearance descriptors (one for each frame)
                self.appearance_loss = \
                    self.build_appearance_descriptors(
                        'appearance_descriptors', 1e9)

                # TODO: let weight be user-definable
                # build dynamics descriptors (one for each pair of
                # frames)
                self.dynamics_loss = \
                    self.build_dynamics_descriptors('dynamics_descriptors',
                                                    1e15)

                # evaluate dynamic texture loss
                self.dyntex_loss = tf.add(self.appearance_loss,
                                          self.dynamics_loss)

                # averaging loss over batch
                self.dyntex_loss = tf.div(self.dyntex_loss,
                                          self.user_config['batch_size'])

                # attach summaries
                self.attach_summaries('summaries')

    # TODO: decouple temporal length between target and output
    def build_appearance_descriptors(self, name, weight):
        with tf.get_default_graph().name_scope(name):
            # TODO: make this user-definable
            loss_layers = ['conv1_1/Relu', 'pool1', 'pool2',
                           'pool3', 'pool4']
            gramians = []
            for i in range(self.input_frame_count):
                # texture target is in RGB [0,1], but VGG
                # accepts BGR [0-mean,255-mean] mean subtracted
                target = vgg_process(self.target_dynamic_texture[i])
                output = self.output[:, i]
                a_t = AppearanceDescriptor('appearance_descriptor_target_' +
                                           str(i+1), name, target)
                a_o = AppearanceDescriptor('appearance_descriptor_output_' +
                                           str(i+1), name, output)
                g = ([a_t.gramian_for_layer(l) for l in loss_layers],
                     [a_o.gramian_for_layer(l) for l in loss_layers])
                gramians.append(g)
            return tf.multiply(self.style_loss('appearance_style_loss',
                                               gramians), weight)

    # TODO: decouple temporal length between target and output
    def build_dynamics_descriptors(self, name, weight):
        with tf.get_default_graph().name_scope(name):
            loss_layers = ['MSOEnet_concat/concat']  # concat layer
            # loss_layers = ['MSOEnet_conv4/BiasAdd']  # flow decode layer
            gramians = []
            for i in range(self.input_frame_count - 1):
                # input is in BGR [0-mean,255-mean] mean subtracted, but
                # MSOEnet accepts grayscale [0,1]
                target = tf.image.rgb_to_grayscale(
                            tf.stack(self.target_dynamic_texture[i:i+2], 1))
                output = tf.image.rgb_to_grayscale(
                    vgg_deprocess(self.output[:, i:i+2], no_clip=True,
                                  unit_scale=True))
                d_t = DynamicsDescriptor('dynamics_descriptor_target_' +
                                         str(i+1), name, target,
                                         self.user_config['dynamics_model'])
                d_o = DynamicsDescriptor('dynamics_descriptor_output_' +
                                         str(i+1), name, output,
                                         self.user_config['dynamics_model'])
                g = ([d_t.gramian_for_layer(l) for l in loss_layers],
                     [d_o.gramian_for_layer(l) for l in loss_layers])
                gramians.append(g)
            return tf.multiply(self.style_loss('dynamics_style_loss',
                                               gramians), weight)

    def style_loss(self, name, gramians):
        with tf.get_default_graph().name_scope(name):
            num_layers = len(gramians[0][0])
            target_gramians = [g[0] for g in gramians]
            output_gramians = [g[1] for g in gramians]
            avg_target_grams = []
            style_losses = []
            for layer in range(num_layers):
                avg_target_grams.append(
                    tf.add_n([g[layer] for g in target_gramians]) /
                    tf.to_float(len(target_gramians)))
            for frame in range(len(gramians)):
                gramian_diffs = [
                    tf.tile(avg_target_grams[layer],
                            [self.user_config['batch_size'], 1, 1]) -
                    output_gramians[frame][layer] for layer in
                    range(num_layers)]

                # MSE
                scaled_diffs = [tf.square(g) for g in gramian_diffs]
                style_losses.append(tf.add_n([tf.reduce_sum(d) for d in
                                    scaled_diffs]) / tf.to_float(num_layers))
            return tf.add_n(style_losses) / tf.to_float(len(style_losses))

    def attach_summaries(self, name):
        with tf.get_default_graph().name_scope(name):
            tf.summary.scalar('appearance_loss', self.appearance_loss)
            tf.summary.scalar('dynamics_loss',
                              self.dynamics_loss)
            tf.summary.scalar('dynamic_texture_loss', self.dyntex_loss)
            self.summaries = tf.summary.merge_all()
