import tensorflow as tf
import argparse
from src.synthesizer import Synthesizer
from src.synthesizer_styletransfer import SynthesizerStyleTransfer
from src.synthesizer_infinite import SynthesizerInfinite
from src.synthesizer_incremental import SynthesizerIncremental
from src.synthesizer_static import SynthesizerStatic
from src.synthesizer_masking import SynthesizerMasking

# set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=1,
                    help='input batch size')
parser.add_argument('--iter', type=int, default=6000,
                    help='number of iterations to optimize for')
parser.add_argument('--sfreq', type=int, default=1000,
                    help='number of iterations before saving a snapshot')
parser.add_argument('--noutfreq', type=int, default=100,
                    help='number of iterations before saving synthesized \
                          textures to disk')
parser.add_argument('--lfreq', type=int, default=100,
                    help='number of iterations before logging to disk')
parser.add_argument('--gpu', type=int, default=0, help='which GPU to use')
parser.add_argument('--runid', default='synthesized',
                    help='id assigned to this run')
parser.add_argument('--dynamics_target', default='',
                    help='path to target dynamic texture')
parser.add_argument('--appearance_target', default='',
                    help='path to target static texture')
parser.add_argument('--dynamics_model', default='MSOEnet',
                    help='path to dynamics modeling network')
parser.add_argument('--type', required=True, help='dts -> dynamic texture \
                    synthesis | dst -> dynamics style transfer | inf -> \
                    infinite/endless dynamic texture synthesis | inc -> \
                    incremental dynamic texture synthesis | sta -> static \
                    texture synthesis')

opt = parser.parse_args()
print(opt)

# config
config_proto = tf.ConfigProto()
config_proto.gpu_options.allow_growth = True
config_proto.allow_soft_placement = True
config_proto.log_device_placement = False
my_config = {}
my_config['batch_size'] = opt.batchsize
my_config['iterations'] = opt.iter
my_config['snapshot_frequency'] = opt.sfreq
my_config['network_out_frequency'] = opt.noutfreq
my_config['log_frequency'] = opt.lfreq
my_config['gpu'] = opt.gpu
my_config['run_id'] = opt.runid
my_config['dynamics_model'] = opt.dynamics_model

if opt.type == 'dts':
    assert opt.dynamics_target != ''
    s = Synthesizer(opt.dynamics_target,
                    config={'tf': config_proto,
                            'user': my_config})
elif opt.type == 'dst':
    assert opt.appearance_target != ''
    assert opt.dynamics_target != ''
    s = SynthesizerStyleTransfer(opt.dynamics_target,
                                 opt.appearance_target,
                                 config={'tf': config_proto,
                                         'user': my_config})
elif opt.type == 'inf':
    assert opt.dynamics_target != ''
    s = SynthesizerInfinite(opt.dynamics_target,
                            config={'tf': config_proto,
                                    'user': my_config})
elif opt.type == 'inc':
    assert opt.appearance_target != ''
    assert opt.dynamics_target != ''
    s = SynthesizerIncremental(opt.dynamics_target,
                               opt.appearance_target,
                               config={'tf': config_proto,
                                       'user': my_config})
elif opt.type == 'sta':
    s = SynthesizerStatic(opt.appearance_target,
                          config={'tf': config_proto,
                                  'user': my_config})

if opt.type == 'mas':
    assert opt.dynamics_target != ''
    s = SynthesizerMasking(opt.dynamics_target,
                           config={'tf': config_proto,
                                   'user': my_config})

s.optimize()
