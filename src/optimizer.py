import tensorflow as tf
import skimage.io
from utilities import check_snapshots, vgg_deprocess
import time
import datetime
import numpy as np
import os


class Optimizer(object):

    def __init__(self, graph, input_dimension, output_dimension,
                 input_frame_count, output_frame_count, target_dynamic_path,
                 target_static_path, config):
        self.graph = graph
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.input_frame_count = input_frame_count
        self.output_frame_count = output_frame_count
        self.target_dynamic_path = target_dynamic_path
        self.target_static_path = target_static_path
        # import config
        self.user_config = config['user']
        self.tf_config = config['tf']

    def print_info(self, losses):
        i = self.iterations_so_far
        iterations = self.user_config['iterations']
        run_id = self.user_config['run_id']

        time_diff = time.time() - self.last_print
        it_per_sec = 1 / time_diff
        remaining_it = iterations - i
        eta = remaining_it / it_per_sec
        eta_string = str(datetime.timedelta(seconds=eta))

        print_string = '(%s) Iteration %d: dynamic texture loss: %f ' \
                       'appearance loss: %f dynamics ' \
                       'loss: %f ' \
                       'iter per/s: %f ETA: %s' % (run_id, i + 1,
                                                   losses[0],
                                                   losses[1],
                                                   losses[2],
                                                   it_per_sec,
                                                   eta_string)
        print print_string
        self.last_print = time.time()

    def minimize_callback(self, dyntex_loss, appearance_loss,
                          dynamics_loss, output, summaries):
        # if hasattr(self, 'current_loss'):
        #     self.past_loss = self.current_loss
        # self.current_loss = dyntex_loss
        # for cleanliness
        i = self.iterations_so_far
        snapshot_frequency = self.user_config['snapshot_frequency']
        network_out_frequency = self.user_config['network_out_frequency']
        log_frequency = self.user_config['log_frequency']
        run_id = self.user_config['run_id']

        # print training information
        self.print_info([dyntex_loss, appearance_loss, dynamics_loss])

        if (i + 1) % snapshot_frequency == 0:
            print 'Saving snapshot...'
            try:
                os.makedirs('snapshots/' + run_id)
            except OSError:
                if not os.path.isdir('snapshots/' + run_id):
                    raise
            self.saver.save(self.sess, 'snapshots/' + run_id + '/iter',
                            global_step=i+1)
        if (i + 1) % log_frequency == 0:
            print 'Saving log file...'
            self.summary_writer.add_summary(summaries, i + 1)
            self.summary_writer.flush()

        if (i + 1) % network_out_frequency == 0:
            print 'Saving image(s)...'
            try:
                os.makedirs('data/out/' + run_id)
            except OSError:
                if not os.path.isdir('data/out/' + run_id):
                    raise
            network_out = output.reshape((-1,
                                          self.output_frame_count,
                                          self.output_dimension,
                                          self.output_dimension, 3))
            img_count = 1
            for out in network_out:
                frame_count = 1
                for frame in out:
                    img_out = vgg_deprocess(frame, no_clip=False,
                                            unit_scale=False)
                    filename = 'data/out/' + run_id + \
                        '/iter_%d_frame_%d_%d.png'
                    skimage.io.imsave(filename %
                                      (i + 1, frame_count,
                                       img_count),
                                      img_out)
                    frame_count += 1
                img_count += 1
        self.iterations_so_far += 1

    def step_callback(self, args):
        if hasattr(self, 'past_loss'):
            loss_diff = (self.past_loss - self.current_loss) / \
                np.amax([np.abs(self.past_loss), np.abs(self.current_loss), 1])
            print 'f diff = ' + str(loss_diff)

    def optimize(self):
        iterations = self.user_config['iterations']
        run_id = self.user_config['run_id']

        with self.graph.as_default():
            """
            Instantiate optimizer
            """
            with tf.device('/gpu:' + str(self.user_config['gpu'])):
                optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                    self.dyntex_loss, method='L-BFGS-B',
                    options={'maxfun': iterations,
                             'disp': True})
                             #'ftol': 1e-5})

            """
            Train over iterations, printing loss at each one
            """
            self.saver = tf.train.Saver(max_to_keep=0, pad_step_number=16)
            with tf.Session(config=self.tf_config) as self.sess:

                # TODO: change snapshot and log folders to be in a single
                # location
                # check snapshots
                resume, self.iterations_so_far = check_snapshots(run_id)

                # start summary writer
                self.summary_writer = tf.summary.FileWriter('logs/' + run_id,
                                                            self.sess.graph)

                if resume:
                    self.saver.restore(self.sess, resume)
                else:
                    self.sess.run(tf.global_variables_initializer())

                # initialize start time
                self.last_print = time.time()

                # start train loop
                print '-------OPTIMIZING USING L-BFGS-B-------'
                # scipy optimizer needs a callback for printing iter info
                optimizer.minimize(self.sess,
                                   fetches=[self.dyntex_loss,
                                            self.appearance_loss,
                                            self.dynamics_loss,
                                            self.output,
                                            self.summaries],
                                   loss_callback=self.minimize_callback)
