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

"""Seq2seq layer operations for use in neural networks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import rnn

from tensorflow.contrib.layers import fully_connected

from functools import partial

__all__ = ["rnn_decoder",
           "rnn_decoder_attention"]

"""Used to project encoder state in `rnn_decoder`"""
encoder_projection = partial(fully_connected, activation_fn=math_ops.tanh)


def rnn_decoder(cell, decoder_inputs, initial_state,
                sequence_length, decoder_fn,
                encoder_projection=encoder_projection,
                parallel_iterations=None, swap_memory=False,
                time_major=False, scope=None):
  """RNN decoder for a sequence-to-sequence model specified by RNNCell 'cell'.

  The 'rnn_decoder' is similar to the 'tf.python.ops.rnn.dynamic_rnn'. As the
  decoder does not make any assumptions of sequence length of the input or how
  many steps it can decode, since 'rnn_decoder' uses dynamic unrolling. This
  allows `decoder_inputs` to have [None] in the sequence length of the decoder
  inputs.

  The parameter decoder_inputs is nessesary for both training and evaluation.
  During training it is feed at every timestep. During evaluation it is only
  feed at time==0, as the decoder needs the `start-of-sequence` symbol, known
  from Sutskever et al., 2014 https://arxiv.org/abs/1409.3215, at the
  beginning of decoding.

  The parameter sequence length is nessesary as it determines how many
  timesteps to decode for each sample. TODO: Could make it optional for
  training.

  Args:
    cell: An instance of RNNCell.
    inputs: The RNN inputs.
      If `time_major == False` (default), this must be a `Tensor` of shape:
        `[batch_size, max_time, ...]`.
      If `time_major == True`, this must be a `Tensor` of shape:
        `[max_time, batch_size, ...]`.
      The input to `cell` at each time step will be a `Tensor` with dimensions
      `[batch_size, ...]`.
    sequence_length: An int32/int64 vector sized `[batch_size]`.
    initial_state: An initial state for the RNN.
      Must be [batch_size, num_features], where num_features does not have to
      match the cell.state_size. As a projection is performed at the beginning
      of the decoding.
    decoder_fn: A function that takes a state and returns an embedding.
      The decoder function is closely related to `_extract_argmax_and_embed`.
      Here is an example of a `decoder_fn`:
      def decoder_fn(embeddings, weight, bias):
        def dec_fn(state):
          prev = tf.matmul(state, weight) + bias
          return tf.gather(embeddings, tf.argmax(prev, 1))
        return dec_fn
    encoder_projection: (optional) given that the encoder might have a
      different size than the decoder, we project the intial state as
      described in Bahdanau, 2014 (https://arxiv.org/abs/1409.0473).
      The optional `encoder_projection` is a
      `tf.contrib.layers.fully_connected` with
      `activation_fn=tf.python.ops.nn.tanh`.
    parallel_iterations: (Default: 32).  The number of iterations to run in
      parallel.  Those operations which do not have any temporal dependency
      and can be run in parallel, will be.  This parameter trades off
      time for space.  Values >> 1 use more memory but take less time,
      while smaller values use less memory but computations take longer.
    swap_memory: Transparently swap the tensors produced in forward inference
      but needed for back prop from GPU to CPU.  This allows training RNNs
      which would typically not fit on a single GPU, with very minimal (or no)
      performance penalty.
    time_major: The shape format of the `inputs` and `outputs` Tensors.
      If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
      If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
      Using `time_major = True` is a bit more efficient because it avoids
      transposes at the beginning and end of the RNN calculation.  However,
      most TensorFlow data is batch-major, so by default this function
      accepts input and emits output in batch-major form.
    scope: VariableScope for the created subgraph; defaults to "RNN".
  Returns:
    A pair (outputs, state) where:
      outputs: The RNN output `Tensor`.
        If time_major == False (default), this will be a `Tensor` shaped:
          `[batch_size, max_time, cell.output_size]`.
        If time_major == True, this will be a `Tensor` shaped:
          `[max_time, batch_size, cell.output_size]`.
      state: The final state.  If `cell.state_size` is an int, this
        will be shaped `[batch_size, cell.state_size]`.  If it is a
        `TensorShape`, this will be shaped `[batch_size] + cell.state_size`.
  Raises:
    TypeError: If `cell` is not an instance of RNNCell.
  """
  with vs.variable_scope(scope or "decoder") as varscope:
    # Project initial_state as described in Bahdanau et al. 2014
    # https://arxiv.org/abs/1409.0473
    state = encoder_projection(initial_state, cell.output_size)
    # Setup of RNN (dimensions, sizes, length, initial state, dtype)
    # Setup dtype
    dtype = state.dtype
    if not time_major:
      # [batch, seq, features] -> [seq, batch, features]
      decoder_inputs = array_ops.transpose(decoder_inputs, perm=[1, 0, 2])
    # Get data input information
    batch_size = array_ops.shape(decoder_inputs)[1]
    decoder_input_depth = int(decoder_inputs.get_shape()[2])
    # Setup decoder inputs as TensorArray
    decoder_inputs_ta = tensor_array_ops.TensorArray(dtype, size=0,
                                                     dynamic_size=True)
    decoder_inputs_ta = decoder_inputs_ta.unpack(decoder_inputs)

    # Define RNN: loop function for training.
    # This will run in the while_loop of 'raw_rnn'
    def loop_fn_train(time, cell_output, cell_state, loop_state):
      emit_output = cell_output
      if cell_output is None:
        next_cell_state = state # use projection of prev encoder state
      else:
        next_cell_state = cell_state
      elements_finished = (time >= sequence_length) #TODO handle seq_len=None
      finished = math_ops.reduce_all(elements_finished)
      # Next input must return zero state for last element explanation below
      # https://github.com/tensorflow/tensorflow/issues/4519
      next_input = control_flow_ops.cond(
          finished,
          lambda: array_ops.zeros([batch_size, decoder_input_depth],
                                  dtype=dtype),
          lambda: decoder_inputs_ta.read(time))
      next_loop_state = None
      return (elements_finished, next_input, next_cell_state,
              emit_output, next_loop_state)

    # Define RNN: loop function for evaluation.
    # This will run in the while_loop of 'raw_rnn'
    def loop_fn_eval(time, cell_output, cell_state, loop_state):
      emit_output = cell_output
      if cell_output is None:
        next_cell_state = state # use projection of prev encoder state
      else:
        next_cell_state = cell_state
      elements_finished = (time >= sequence_length) #TODO handle seq_len=None
      finished = math_ops.reduce_all(elements_finished)
      # Next input must return zero state for last element explanation below
      # https://github.com/tensorflow/tensorflow/issues/4519
      next_input = control_flow_ops.cond(
          finished,
          lambda: array_ops.zeros([batch_size, decoder_input_depth],
                                  dtype=dtype),
          lambda: control_flow_ops.cond(math_ops.greater(time, 0),
              lambda: decoder_fn(next_cell_state), # Gather max prediction.
              lambda: decoder_inputs_ta.read(0))) # Read <EOS> tag
      next_loop_state = None
      return (elements_finished, next_input, next_cell_state,
              emit_output, next_loop_state)

    # Run raw_rnn function
    outputs_ta_train, _, _ = \
      rnn.raw_rnn(cell, loop_fn_train,
                  parallel_iterations=parallel_iterations,
                  swap_memory=swap_memory, scope=varscope)
    # Reuse the cell for evaluation
    varscope.reuse_variables()
    outputs_ta_eval, _, _ = \
      rnn.raw_rnn(cell, loop_fn_eval,
                  parallel_iterations=parallel_iterations,
                  swap_memory=swap_memory, scope=varscope)
    outputs_train = outputs_ta_train.pack()
    outputs_eval = outputs_ta_eval.pack()
    if not time_major:
      # [seq, batch, features] -> [batch, seq, features]
      outputs_train = array_ops.transpose(outputs_train, perm=[1, 0, 2])
      outputs_eval = array_ops.transpose(outputs_eval, perm=[1, 0, 2])
    return outputs_train, outputs_eval


def rnn_decoder_attention(*args, **kwargs):
  pass
