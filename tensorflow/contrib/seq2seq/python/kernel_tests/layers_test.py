# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Tests for contrib.seq2seq.python.seq2seq.layers_ops."""
# pylint: disable=unused-import,g-bad-import-order
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# pylint: enable=unused-import

import tensorflow as tf


class LayersTest(tf.test.TestCase):

  # test time_major=False
  def test_rnn_decoder1(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        BATCH_SIZE = 2
        INPUT_SIZE = 3
        DECODER_INPUT_SIZE = 4
        ENCODER_SIZE = 6
        DECODER_SIZE = 7
        INPUT_SEQUENCE_LENGTH = 8
        DECODER_SEQUENCE_LENGTH = 9

        inputs = tf.constant(0.5, shape=[BATCH_SIZE,
                                         INPUT_SEQUENCE_LENGTH,
                                         INPUT_SIZE])
        _, encoder_state = tf.nn.dynamic_rnn(
            cell=tf.nn.rnn_cell.GRUCell(ENCODER_SIZE), inputs=inputs,
            dtype=tf.float32, time_major=False)

        decoder_inputs = tf.constant(0.4, shape=[BATCH_SIZE,
                                                 DECODER_SEQUENCE_LENGTH,
                                                 DECODER_INPUT_SIZE])
        decoder_length = tf.constant(DECODER_SEQUENCE_LENGTH, dtype=tf.int32,
                                     shape=[BATCH_SIZE,])
        decoder_fn = lambda state: tf.constant(0.5,
                                               shape=[BATCH_SIZE,
                                                      DECODER_INPUT_SIZE])
        decoder_outputs, valid_decoder_outputs = tf_utils.rnn_decoder(
                cell=tf.nn.rnn_cell.GRUCell(DECODER_SIZE),
                decoder_inputs=decoder_inputs, initial_state=encoder_state,
                decoder_fn=decoder_fn, sequence_length=decoder_length,
                time_major=False)

        tf.initialize_all_variables().run()
        decoder_outputs_res = sess.run(decoder_outputs)
        valid_decoder_outputs_res = sess.run(valid_decoder_outputs)
        self.assertEqual((BATCH_SIZE, DECODER_SEQUENCE_LENGTH, DECODER_SIZE),
                         decoder_outputs_res.shape)
        self.assertEqual((BATCH_SIZE, DECODER_SEQUENCE_LENGTH, DECODER_SIZE),
                         valid_decoder_outputs_res.shape)

  # test time_major=True
  def test_rnn_decoder2(self):
    with tf.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        BATCH_SIZE = 2
        INPUT_SIZE = 3
        DECODER_INPUT_SIZE = 4
        ENCODER_SIZE = 6
        DECODER_SIZE = 7
        INPUT_SEQUENCE_LENGTH = 8
        DECODER_SEQUENCE_LENGTH = 9

        inputs = tf.constant(0.5, shape=[INPUT_SEQUENCE_LENGTH,
                                         BATCH_SIZE,
                                         INPUT_SIZE])
        _, encoder_state = tf.nn.dynamic_rnn(
            cell=tf.nn.rnn_cell.GRUCell(ENCODER_SIZE), inputs=inputs,
            dtype=tf.float32, time_major=True)

        decoder_inputs = tf.constant(0.4, shape=[DECODER_SEQUENCE_LENGTH,
                                                 BATCH_SIZE,
                                                 DECODER_INPUT_SIZE])
        decoder_length = tf.constant(DECODER_SEQUENCE_LENGTH, dtype=tf.int32,
                                     shape=[BATCH_SIZE,])
        decoder_fn = lambda state: tf.constant(0.5,
                                               shape=[BATCH_SIZE,
                                                      DECODER_INPUT_SIZE])
        decoder_outputs, valid_decoder_outputs = tf_utils.rnn_decoder(
                cell=tf.nn.rnn_cell.GRUCell(DECODER_SIZE),
                decoder_inputs=decoder_inputs, initial_state=encoder_state,
                decoder_fn=decoder_fn, sequence_length=decoder_length,
                time_major=True)

        tf.initialize_all_variables().run()
        decoder_outputs_res = sess.run(decoder_outputs)
        valid_decoder_outputs_res = sess.run(valid_decoder_outputs)
        self.assertEqual((DECODER_SEQUENCE_LENGTH, BATCH_SIZE, DECODER_SIZE),
                         decoder_outputs_res.shape)
        self.assertEqual((DECODER_SEQUENCE_LENGTH, BATCH_SIZE, DECODER_SIZE),
                         valid_decoder_outputs_res.shape)


  def test_rnn_decoder_attention(self):
    pass


if __name__ == '__main__':
  tf.test.main()
