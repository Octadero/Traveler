"""
    Travaller neural network
    Author: Volodymyr Pavliukevych
"""

import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import urban_area as ua

TF_FLAGS = tf.app.flags
TF_FLAGS.DEFINE_integer('samples', 5000, 'Number of samples to run.')
TF_FLAGS.DEFINE_string('logdir', '/tmp/traveler/', 'Path to log folder.')
TF_FLAGS.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
TF_FLAGS.DEFINE_string('checkpoint', './saved/model.ckpt', 'Checkpoint path')
TF_FLAGS.DEFINE_integer('epochs', 100, 'Number of Epochs')
TF_FLAGS.DEFINE_integer('batch_size', 75, 'Batch Size')
TF_FLAGS.DEFINE_integer('rnn_size', 300, 'RNN Size')
TF_FLAGS.DEFINE_integer('num_layers', 2, 'Number of Layers')
TF_FLAGS.DEFINE_integer('encoding_embedding_size', 10, 'Embedding Size')
TF_FLAGS.DEFINE_integer('decoding_embedding_size', 10, 'Embedding Size')
TF_FLAGS.DEFINE_bool('debug_print', True, 'Debug print')
TF_FLAGS.DEFINE_integer('gpu', 1, 'Number of GPU')

def debug_print(*args):
    "Debug print function"
    timestamp = time.time()
    if TF_FLAGS.FLAGS.debug_print:
        print('[{:.4f}] '.format(timestamp), *args)

class TravellerNeuralNetwork(object):
    """
        Simple traveller neural network class.
    """

    def __init__(self, session, graph):
        "Init attribute"
        super().__init__()
        debug_print("batch_size: ", TF_FLAGS.FLAGS.batch_size, ", rnn_size: ", TF_FLAGS.FLAGS.rnn_size, "checkpoint: ", TF_FLAGS.FLAGS.checkpoint)
        self.session = session
        self.graph = graph
        self.source_sequences = []
        self.target_sequences = []
        self.vocabulary_size = len(ua.UrbanArea.chars)
        debug_print("Creating network.")
        self.fill_samples()
        self.build_model()
        self.visualize()
        debug_print("Network is ready.")

    def fill_samples(self):
        "Feading samples."
        for _ in range(0, TF_FLAGS.FLAGS.samples):
            area = ua.UrbanArea()
            self.source_sequences.append(area.output())
            self.target_sequences.append(area.get_road())

    def build_model(self):
        "Build the graph."
        debug_print("Building network model.")
        # Set the graph to default to ensure that it is ready for training
        with self.graph.as_default():
            # Load the model inputs
            self.input_data_input = tf.placeholder(tf.int32, [None, None], name='input')
            self.targets = tf.placeholder(tf.int32, [None, None], name='targets')
            self.learning_rate_input = tf.placeholder(tf.float32, name='learning_rate')
            self.target_sequences_length_input = tf.placeholder(tf.int32, (None,), name='target_sequences_length')
            self.max_target_sequences_length = tf.reduce_max(self.target_sequences_length_input, name='max_target_len')
            self.source_sequences_length_input = tf.placeholder(tf.int32, (None,), name='source_sequences_length')

            # Pass the input data through the encoder. We'll ignore the encoder output, but use the state
            _, enc_state = self.encoding_layer(self.input_data_input, self.vocabulary_size, self.source_sequences_length_input)

            # Prepare the target sequences we'll feed to the decoder in training mode
            # Process the input we'll feed to the decoder
            # Remove the last word id from each batch and concat the <GO> to the begining of each batch
            ending = tf.strided_slice(self.targets, [0, 0], [TF_FLAGS.FLAGS.batch_size, -1], [1, 1])
            decoder_input = tf.concat([tf.fill([TF_FLAGS.FLAGS.batch_size, 1], ua.UrbanArea.vacab_go_key), ending], 1)

            # Pass encoder state and decoder inputs to the decoders
            training_decoder_output, inference_decoder_output = self.decoding_layer(enc_state, decoder_input)

        debug_print("Creating loss and optimization ...")
        # Create the training and inference logits
        # Create tensors for the training logits and inference logits
        training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
        _ = tf.identity(inference_decoder_output.sample_id, name='predictions')

        # Create the weights for sequence_loss
        masks = tf.sequence_mask(self.target_sequences_length_input, self.max_target_sequences_length, dtype=tf.float32, name='masks')

        with tf.name_scope("optimization"):
            # Loss function
            self.cost = tf.contrib.seq2seq.sequence_loss(training_logits, self.targets, masks)
            tf.summary.scalar("loss", self.cost)
            # Optimizer
            optimizer = tf.train.AdamOptimizer(self.learning_rate_input)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(self.cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            self.train_op = optimizer.apply_gradients(capped_gradients)

    def encoding_layer(self, input_data_input, vocabulary_size, source_sequences_length_input):
        "Encoder"
        debug_print("Building encoder.")
        # 1. Encoder embedding
        encoder_embed_input = tf.contrib.layers.embed_sequence(input_data_input,
                                                               vocabulary_size,
                                                               TF_FLAGS.FLAGS.encoding_embedding_size)
        # 2. Construct the encoder layer
        encoder_cell = tf.contrib.rnn.MultiRNNCell([self.make_cell() for _ in range(TF_FLAGS.FLAGS.num_layers)])
        enc_output, enc_state = tf.nn.dynamic_rnn(encoder_cell, encoder_embed_input, sequence_length=source_sequences_length_input, dtype=tf.float32)
        debug_print("Encoder is ready.")
        return enc_output, enc_state

    def decoding_layer(self, encoder_state, decoder_input):
        "Decoder"
        target_sequences_length = self.target_sequences_length_input
        max_target_sequences_length = self.max_target_sequences_length
        # 1. Decoder Embedding
        target_vocab_size = self.vocabulary_size
        decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, TF_FLAGS.FLAGS.decoding_embedding_size]))
        decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

        # 2. Construct the decoder layer
        dec_cell = tf.contrib.rnn.MultiRNNCell([self.make_cell() for _ in range(TF_FLAGS.FLAGS.num_layers)])

        # 3. Dense layer to translate the decoder's output at each time
        # step into a choice from the target vocabulary
        output_layer = Dense(target_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        # 4. Set up a training decoder and an inference decoder
        # Training Decoder
        with tf.variable_scope("decode"):

            # Helper for the training process. Used by BasicDecoder to read inputs.
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input, sequence_length=target_sequences_length, time_major=False)

            # Basic decoder
            training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, training_helper, encoder_state, output_layer)

            # Perform dynamic decoding using the decoder
            training_decoder_output = tf.contrib.seq2seq.dynamic_decode(training_decoder, impute_finished=True, maximum_iterations=max_target_sequences_length)[0]

        # 5. Inference Decoder
        # Reuses the same parameters trained by the training process
        with tf.variable_scope("decode", reuse=True):
            start_tokens = tf.tile(tf.constant([ua.UrbanArea.vacab_go_key], dtype=tf.int32), [TF_FLAGS.FLAGS.batch_size], name='start_tokens')

            # Helper for the inference process.
            inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings, start_tokens, ua.UrbanArea.vacab_eos_key)

            # Basic decoder
            inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, inference_helper, encoder_state, output_layer)

        # Perform dynamic decoding using the decoder
        inference_decoder_output = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                     impute_finished=True,
                                                                     maximum_iterations=max_target_sequences_length)[0]
        debug_print("Decoder is ready.")
        return training_decoder_output, inference_decoder_output

    @classmethod
    def make_cell(cls):
        "Cell helper"
        dec_cell = tf.contrib.rnn.LSTMCell(TF_FLAGS.FLAGS.rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
        return dec_cell

    @classmethod
    def pad_sentence_batch(cls, sentence_batch, pad_int):
        """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

    @classmethod
    def get_batches(cls, targets, sources, source_pad_int, target_pad_int):
        """Batch targets, sources, and the lengths of their sentences together"""
        batch_size = TF_FLAGS.FLAGS.batch_size
        for batch_i in range(0, len(sources)//batch_size):
            start_i = batch_i * batch_size
            sources_batch = sources[start_i:start_i + batch_size]
            targets_batch = targets[start_i:start_i + batch_size]
            pad_sources_batch = np.array(cls.pad_sentence_batch(sources_batch, source_pad_int))
            pad_targets_batch = np.array(cls.pad_sentence_batch(targets_batch, target_pad_int))

            # Need the lengths for the _lengths parameters
            pad_targets_lengths = []
            for target in pad_targets_batch:
                pad_targets_lengths.append(len(target))

            pad_source_lengths = []
            for source in pad_sources_batch:
                pad_source_lengths.append(len(source))

            yield pad_targets_batch, pad_sources_batch, pad_targets_lengths, pad_source_lengths

    def visualize(self):
        "Visualisation"
        debug_print("Visualize model at: `", TF_FLAGS.FLAGS.logdir, "` ...")
        self.summary_writer = tf.summary.FileWriter(TF_FLAGS.FLAGS.logdir, graph=self.graph)
        self.summaries = tf.summary.merge_all()
        debug_print("Visualisation was saved.")

    @classmethod
    def restore(cls, session):
        "Load saved model"
        debug_print("Restoring saved model.")
        loader = tf.train.import_meta_graph(TF_FLAGS.FLAGS.checkpoint + '.meta')
        loader.restore(session, TF_FLAGS.FLAGS.checkpoint)

    def evaluation(self):
        "Evaluation"
        # Split data to training and validation sets
        train_source = self.source_sequences[TF_FLAGS.FLAGS.batch_size:]
        train_target = self.target_sequences[TF_FLAGS.FLAGS.batch_size:]
        validation_source = self.source_sequences[:TF_FLAGS.FLAGS.batch_size]
        validation_target = self.target_sequences[:TF_FLAGS.FLAGS.batch_size]
        validation_batche = next(self.get_batches(validation_target,
                                                  validation_source,
                                                  ua.UrbanArea.vacab_pad_key,
                                                  ua.UrbanArea.vacab_pad_key))
        (validation_targets_batch, validation_sources_batch, validation_targets_lengths, validation_sources_lengths) = validation_batche
        display_step = 20 # Check training loss after every 20 batches
        timestamp = time.time()
        evaluation_global_step = 0
        self.session.run(tf.global_variables_initializer())

        for epoch_i in range(1, TF_FLAGS.FLAGS.epochs+1):
            batche_iterator = self.get_batches(train_target, train_source, ua.UrbanArea.vacab_pad_key, ua.UrbanArea.vacab_pad_key)
            for batch_i, batch in enumerate(batche_iterator):
                (targets_batch, sources_batch, targets_lengths, sources_lengths) = batch
                evaluation_global_step += 1
                # Training step
                _, loss = self.session.run([self.train_op, self.cost],
                                           {self.input_data_input : sources_batch,
                                            self.targets : targets_batch,
                                            self.learning_rate_input : TF_FLAGS.FLAGS.learning_rate,
                                            self.target_sequences_length_input : targets_lengths,
                                            self.source_sequences_length_input : sources_lengths})

                # Debug message updating us on the status of the training
                if batch_i % display_step == 0 and batch_i > 0:

                    # Calculate validation cost
                    validation_loss, summary = self.session.run([self.cost, self.summaries],
                                                                {self.input_data_input : validation_sources_batch,
                                                                 self.targets : validation_targets_batch,
                                                                 self.learning_rate_input : TF_FLAGS.FLAGS.learning_rate,
                                                                 self.target_sequences_length_input : validation_targets_lengths,
                                                                 self.source_sequences_length_input : validation_sources_lengths})
                    self.summary_writer.add_summary(summary, global_step=(evaluation_global_step * (TF_FLAGS.FLAGS.batch_size / 25)))
                    newtimestamp = time.time()
                    diff = (newtimestamp - timestamp) / float(TF_FLAGS.FLAGS.batch_size * display_step)
                    timestamp = newtimestamp
                    print('Epoch {}/{} \t Batch {}/{} \t Loss: {:.3f} \t Validation loss: {:.3f} \t Time: {:.4f}s'\
                          .format(epoch_i,
                                  TF_FLAGS.FLAGS.epochs,
                                  batch_i,
                                  len(train_source) // TF_FLAGS.FLAGS.batch_size,
                                  loss,
                                  validation_loss,
                                  diff))

        # Save Model
        saver = tf.train.Saver()
        saver.save(self.session, TF_FLAGS.FLAGS.checkpoint)
        debug_print('Model Trained and Saved')

    def inference(self):
        "Inference network"
        area = ua.UrbanArea()
        area.debug()
        text = area.output()
        loaded_graph = tf.Graph()
        with tf.Session(graph=loaded_graph) as session:
            self.restore(session)
            input_data = loaded_graph.get_tensor_by_name('input:0')
            logits = loaded_graph.get_tensor_by_name('predictions:0')
            source_sequences_length = loaded_graph.get_tensor_by_name('source_sequences_length:0')
            target_sequences_length = loaded_graph.get_tensor_by_name('target_sequences_length:0')

            #Multiply by batch_size to match the model's input parameters
            response = session.run(logits,
                                   {input_data: [text]*TF_FLAGS.FLAGS.batch_size,
                                    target_sequences_length: [len(text)]*TF_FLAGS.FLAGS.batch_size,
                                    source_sequences_length: [len(text)]*TF_FLAGS.FLAGS.batch_size})
            answer_logits = response[0]
        print('Response steps: {}'.format(" ".join([ua.UrbanArea.chars[i] for i in answer_logits if i != ua.UrbanArea.vacab_pad_key])))

def main(argv):
    "Main function"
    debug_print(argv)
    graph = tf.Graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(device_count={'CPU' : 2, 'GPU' : TF_FLAGS.FLAGS.gpu}, allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
    with tf.Session(config=config, graph=graph) as session:
        network = TravellerNeuralNetwork(session=session, graph=graph)
        network.evaluation()
        network.inference()

if __name__ == "__main__":
    main(sys.argv)
