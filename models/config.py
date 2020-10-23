import os
import tensorflow as tf

flags = tf.app.flags


############################
#    hyper parameters      #
############################

# For margin loss
flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')

# For spread loss
flags.DEFINE_float('m_scheduler', 1, '.')
flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')

# for training
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('num_steps', 300000, 'The number of training epochs')
flags.DEFINE_integer('train_sum_every', 1000, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('val_sum_every', 1000, 'the frequency of saving evaluation summary(step)')
flags.DEFINE_integer('save_ckpt_every', 1000, 'the frequency of saving model(step)')

flags.DEFINE_float('learning_rate', 0.0001, 'learning rate for adam optimizer')
flags.DEFINE_float('decay_step', -1, 'steps after decay of learning rate (if negative, no decay)')
flags.DEFINE_float('regularization_scale', 0.3, 'regularization coefficient for reconstruction loss')


############################
#   environment setting    #
############################
flags.DEFINE_string('model', 'ResCaps',
                    'The model to use. Default: ResCaps')

supported_datasets = ["mnist", "fashion_mnist", "cifar10", "cifar100", "small-norb", "imagenette"]
flags.DEFINE_string('dataset', 'cifar10',
                    'The name of dataset, one of [' + ", ".join(supported_datasets) + ']. Default: cifar10')

data_dir = os.path.abspath("data")
results_dir = os.path.join('models', 'results')
logdir = os.path.join(results_dir, 'logdir')

flags.DEFINE_string('data_dir', data_dir,
                    'The directory containing dataset. Default: ' + data_dir)
flags.DEFINE_string('results_dir', results_dir,
                    'The directory to save all results. Default: ' + results_dir)
flags.DEFINE_string('logdir', logdir,
                    'Logs directory for saving checkpoint. Default: ' + logdir)
flags.DEFINE_string('splitting', "TVT",
                    'One of "TVT" or "TT" (case-insensitive). \
                     "TVT" for training-validation-test data splitting, and "TT" for training-test splitting. \
                     Default: TVT')
flags.DEFINE_boolean('is_training', True,
                     'Boolean, train or predict mode. Default: True')
flags.DEFINE_integer('num_works', 8,
                     'The number of works for processing data')
flags.DEFINE_boolean('summary_verbose', True, 'Use tensorflow summary')

flags.DEFINE_boolean('verbose', True, 'verbose output')

############################
#   distributed setting    #
############################
flags.DEFINE_integer('num_gpus', 1, 'The number of gpus for distributed training')
flags.DEFINE_integer('batch_size_per_gpu', 128, 'batch size on 1 gpu')
flags.DEFINE_integer('thread_per_gpu', 8, 'Number of preprocessing threads per tower.')

cfg = tf.app.flags.FLAGS

# Uncomment this line to run in debug mode
# tf.logging.set_verbosity(tf.logging.INFO)
