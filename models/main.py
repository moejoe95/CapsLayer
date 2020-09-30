# Copyright 2018 The CapsLayer Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==========================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from importlib import import_module
from capslayer.plotlib import plot_activation

from config import cfg


def get_file_descriptors(result_dir, filenames):
    fds = {}
    for filename in filenames:
        file_path = os.path.join(result_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        fd = open(file_path, 'w')
        fd.write('step,'+ filename +'\n')
        fds[filename] = fd
    return fds


def fd_write_log(fds, step, values):
    i = 0
    for filename in fds:
        if filename == 'val_acc.csv':
            continue
        fds[filename].write("{:d},{:.4f}\n".format(step, values[i]))
        fds[filename].flush()
        i += 1


def save_to(result_dir):
    os.makedirs(os.path.join(cfg.results_dir, "activations"), exist_ok=True)
    os.makedirs(os.path.join(cfg.results_dir, "timelines"), exist_ok=True)

    if cfg.is_training:
        log_files = ['loss.csv', 'train_acc.csv', 'val_acc.csv', 't_score.csv', 'd_score.csv']
        fd = get_file_descriptors(result_dir, log_files)

    else:
        test_acc = os.path.join(cfg.results_dir, 'test_acc.csv')
        if os.path.exists(test_acc):
            os.remove(test_acc)
        fd_test_acc = open(test_acc, 'w')
        fd_test_acc.write('test_acc\n')
        fd = {"test_acc": fd_test_acc}

    return fd


def train(model, data_loader):
    # Setting up model
    training_iterator = data_loader(cfg.batch_size, mode="train")
    validation_iterator = data_loader(cfg.batch_size, mode="eval")
    inputs = data_loader.next_element["images"]
    labels = data_loader.next_element["labels"]

    model.create_network(inputs, labels)

    loss, train_ops, summary_ops = model.train(cfg.num_gpus)

    # Creating files, saver and summary writer to save training results
    fd = save_to(model.model_result_dir)
    summary_writer = tf.compat.v1.summary.FileWriter(cfg.logdir)
    summary_writer.add_graph(tf.compat.v1.get_default_graph())

    saver = tf.compat.v1.train.Saver(max_to_keep=3)

    # Setting up training session
    #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:

        last_checkpoint = tf.train.latest_checkpoint(cfg.logdir)

        if last_checkpoint is None:
            tf.logging.info('Train model from scratch!')
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
        else:
            saver.restore(sess, last_checkpoint)
            tf.logging.info('Model restored!')
        

        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())

        print("\nNote: all of results will be saved to directory: " + cfg.results_dir)
        for step in range(1, cfg.num_steps):
            start_time = time.time()
            if step % cfg.train_sum_every == 0:
                _, loss_val, train_acc, summary_str, T, D = sess.run([train_ops,
                                                               loss,
                                                               model.accuracy,
                                                               summary_ops,
                                                               model.T,
                                                               model.D],
                                                               feed_dict={data_loader.handle: training_handle})
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                out_path = os.path.join(cfg.results_dir, "timelines/timeline_%d.json" % step)
                with open(out_path, "w") as f:
                    f.write(ctf)
                summary_writer.add_summary(summary_str, step)
                fd_write_log(fd, step, [loss_val, train_acc, T, D])
            else:
                _, loss_val, T, D = sess.run([train_ops, loss, model.T, model.D], feed_dict={data_loader.handle: training_handle})
                assert not np.isnan(loss_val), 'Something wrong! loss is nan...'

            if step % cfg.val_sum_every == 0:
                if cfg.verbose:
                    print("evaluating, it will take a while...")
                sess.run(validation_iterator.initializer)
                probs = []
                targets = []
                total_acc = 0
                n = 0
                while True:
                    try:
                        val_acc, prob, label = sess.run([model.accuracy, model.probs, labels], feed_dict={data_loader.handle: validation_handle})
                        probs.append(prob)
                        targets.append(label)
                        total_acc += val_acc
                        n += 1
                    except tf.errors.OutOfRangeError:
                        break
                probs = np.concatenate(probs, axis=0)
                targets = np.concatenate(targets, axis=0).reshape((-1, 1))
                avg_acc = total_acc / n
                path = os.path.join(os.path.join(cfg.results_dir, "activations"))
                plot_activation(np.hstack((probs, targets)), step=step, save_to=path)
                fd['val_acc.csv'].write("{:d},{:.4f}\n".format(step, avg_acc))
                fd['val_acc.csv'].flush()
            if step % cfg.save_ckpt_every == 0:
                saver.save(sess,
                           save_path=os.path.join(cfg.logdir, 'model.ckpt'),
                           global_step=step)

            duration = time.time() - start_time
            log_str = ' step: {:d}, loss: {:.3f}, time: {:.3f} sec/step, T: {:.3f}, D: {:.3f}' \
                      .format(step, loss_val, duration, T, D)
            if cfg.verbose:
                print(log_str)


def evaluate(model, data_loader):
    # Setting up model
    test_iterator = data_loader(cfg.batch_size, mode="test")
    inputs = data_loader.next_element["images"]
    labels = data_loader.next_element["labels"]
    model.create_network(inputs, labels)

    # Create files to save evaluating results
    fd = save_to()
    saver = tf.compat.v1.train.Saver()

    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        test_handle = sess.run(test_iterator.string_handle())
        saver.restore(sess, tf.train.latest_checkpoint(cfg.logdir))
        tf.logging.info('Model restored!')

        probs = []
        targets = []
        total_acc = 0
        n = 0
        while True:
            try:
                test_acc, prob, label = sess.run([model.accuracy, model.probs, labels], feed_dict={data_loader.handle: test_handle})
                probs.append(prob)
                targets.append(label)
                total_acc += test_acc
                n += 1
            except tf.errors.OutOfRangeError:
                break
        probs = np.concatenate(probs, axis=0)
        targets = np.concatenate(targets, axis=0).reshape((-1, 1))
        avg_acc = total_acc / n
        out_path = os.path.join(cfg.results_dir, 'prob_test.txt')
        np.savetxt(out_path, np.hstack((probs, targets)), fmt='%1.2f')
        print('Classification probability for each category has been saved to ' + out_path)
        fd["test_acc"].write(str(avg_acc))
        fd["test_acc"].close()
        out_path = os.path.join(cfg.results_dir, 'test_accuracy.txt')
        print('Test accuracy has been saved to ' + out_path)


def main(_):

    model_list = ['baseline', 'vectorCapsNet', 'matrixCapsNet', 'ResCaps', 'capsNet_big']

    # Deciding which model to use
    if cfg.model == 'baseline':
        model = import_module(cfg.model).Model
    elif cfg.model in model_list:
        model = import_module(cfg.model).CapsNet
    else:
        raise ValueError('Unsupported model, please check the name of model:', cfg.model)

    # load data set
    dataset = "capslayer.data.datasets." + cfg.dataset
    data_loader = import_module(dataset).DataLoader(path=cfg.data_dir,
                                                    splitting=cfg.splitting,
                                                    num_works=cfg.num_works)

    # Deciding which dataset to use
    if cfg.dataset == 'mnist' or cfg.dataset == 'fashion_mnist':
        height = 28
        width = 28
        channels = 1
        num_label = 10
    elif cfg.dataset == 'cifar10' or cfg.dataset == 'cifar100':
        height = 32
        width = 32
        channels = 3
        num_label = 10
    elif cfg.dataset == 'small_norb':
        num_label = 5
        height = 32
        width = 32
        channels = 1
    elif cfg.dataset == 'imagenette':
        num_label = 10
        height = 160
        width = 160
        channels = 3

    # Initializing model and data loader
    net = model(height=height, width=width, channels=channels, num_label=num_label)

    # Deciding to train or evaluate model
    if cfg.is_training:
        train(net, data_loader)
    else:
        evaluate(net, data_loader)


if __name__ == "__main__":

    try:
        import cluster_setup
    except ImportError:
        print('IMPORT ERROR')
        pass

    tf.compat.v1.app.run()
