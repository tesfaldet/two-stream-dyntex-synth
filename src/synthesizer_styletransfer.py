import tensorflow as tf
from utilities import load_image, load_images, vgg_process, vgg_deprocess
from appearance_descriptor import AppearanceDescriptor
from dynamics_descriptor import DynamicsDescriptor
from optimizer import Optimizer
import numpy as np


class SynthesizerStyleTransfer(Optimizer):

    # TODO: let spatiotemporal size be user-definable
    def __init__(self, target_dynamic_path, target_static_path, config):
        Optimizer.__init__(self, tf.Graph(), 256, 12,
                           target_dynamic_path, target_static_path,
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

                # load static texture (for dynamics style transfer)
                img = load_image(target_static_path,
                                 size=(self.input_dimension,
                                       self.input_dimension))
                self.target_static_texture = tf.to_float(
                       tf.constant(img.reshape(1, self.input_dimension,
                                               self.input_dimension, 3)))

                # TODO: check for b/w input
                # initialize noise
                initial_noise = tf.random_normal([self.user_config
                                                  ['batch_size'],
                                                  self.input_frame_count,
                                                  self.input_dimension,
                                                  self.input_dimension, 3])
                self.output = tf.Variable(initial_noise, name='output')

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

    def build_appearance_descriptors(self, name, weight):
        with tf.get_default_graph().name_scope(name):
            loss_layers = ['conv1_1/Relu', 'pool1', 'pool2',
                           'pool3', 'pool4']
            gramians = []
            for i in range(self.input_frame_count):
                # texture target is in RGB [0,1], but VGG
                # accepts BGR [0-mean,255-mean] mean subtracted
                input = [vgg_process(self.target_static_texture),
                         self.output[:, i]]
                a = AppearanceDescriptor('appearance_descriptor_' + str(i+1),
                                         name, tf.concat(axis=0, values=input))
                gramians.append([a.gramian_for_layer(l) for l in loss_layers])
            return tf.multiply(self.style_loss('appearance_style_loss',
                                               gramians), weight)

    def build_dynamics_descriptors(self, name, weight):
        with tf.get_default_graph().name_scope(name):
            loss_layers = ['MSOEnet_concat/concat']
            gramians = []
            for i in range(self.input_frame_count - 1):
                # input is in BGR [0-mean,255-mean] mean subtracted, but
                # MSOEnet accepts grayscale [0,1]
                target = tf.image.rgb_to_grayscale(
                            tf.stack(self.target_dynamic_texture[i:i+2], 1))
                output = tf.image.rgb_to_grayscale(
                    vgg_deprocess(self.output[:, i:i+2], no_clip=True,
                                  unit_scale=True))
                input = [target, output]
                d = DynamicsDescriptor('dynamics_descriptor_' + str(i+1),
                                       name, tf.concat(axis=0, values=input),
                                       self.user_config['dynamics_model'])
                gramians.append([d.gramian_for_layer(l) for l in loss_layers])
            return tf.multiply(self.style_loss('dynamics_style_loss',
                                               gramians), weight)

    def style_loss(self, name, gramians):
        with tf.get_default_graph().name_scope(name):
            num_layers = len(gramians[0])
            target_gramians = [[g[:1] for g in grams] for grams in gramians]
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
                    gramians[frame][layer][1:] for layer in range(num_layers)]

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
