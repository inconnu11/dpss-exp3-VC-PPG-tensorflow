import tensorflow as tf
from tensorflow.contrib import rnn
import pickle
import config
from load_data import LoadedData
from data_set import DataSet


class MyError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class DblstmModel:
    def __init__(self):
        self.batch_size = config.batch_size
        self.num_train_steps = config.train_steps
        self.session = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
        self.learning_rate_ph = tf.placeholder(tf.float64,
                                               shape=[],
                                               name='learning_rate')
        # loaded data
        self.loaded_data = None
        # data set
        self.ds = None
        # data set iterator
        self.train_dataset_iter = None
        self.test_dataset_iter = None
        self.vali_dataset_iter = None
        # data set handle
        self.train_handle = None
        self.test_handle = None
        self.validation_handle = None
        # data set iter
        self.dataset_iter = None
        self.dataset_handle = tf.placeholder(tf.string, shape=[])
        self.batch_features = None
        self.batch_labels = None
        self.batch_lengths = None
        self.batch_uttids = None
        self.batch_outputs = None
        self.batch_mse_loss = None
        self.saver = None
        # training parameters
        self.train_step = None
        self.learning_rate_0 = config.initial_lr
        self.learning_rate_1 = config.final_lr
        self.lr_decay_step = config.change_lr_step
        # tensorboard writer
        self.train_writer = None
        self.test_writer = None
        self.vali_writer = None
        self.tensorboard_merger = None

    def get_session(self):
        if self.session is None:
            raise MyError('Session should be initialized!')
        return self.session

    def close_session(self):
        if not (self.session is None):
            self.session.close()
        self.session = None

    def get_dataset(self, is_train=True):
        ld = LoadedData()
        if is_train:
            ld.load_data()
            ld.label_normalize()
        else:
            # load source speaker data
            ld.load_data(filename=config.src_data_dir,
                         test_set_size=config.src_test_size,
                         vali_set_size=config.src_vali_size)
        # sort the data
        # ld.sort_data()
        ld.print_info()
        dataset = DataSet(ld)
        self.ds = dataset
        self.loaded_data = ld
        self.train_dataset_iter = dataset.train_iterator
        self.vali_dataset_iter = dataset.vali_iterator
        self.test_dataset_iter = dataset.test_iterator
        self.dataset_iter = tf.data.Iterator.from_string_handle(
            self.dataset_handle,
            dataset.train_set.output_types,
            dataset.train_set.output_shapes)
        with tf.name_scope('batch_data'):
            self.batch_features, \
            self.batch_labels, \
            self.batch_lengths, \
            self.batch_uttids = self.dataset_iter.get_next()

    def build_graph(self, is_train=True, is_load_model=False):
        with tf.device('/cpu:0'):
            self.get_dataset(is_train)
        self.batch_outputs = self.model(self.batch_features,
                                        self.batch_lengths,
                                        drop_rate=config.dropout_rate,
                                        is_train=is_train)
        # choose a loss
        self.batch_mse_loss = self.mse_loss_normalized()
        tf.summary.scalar(name='MSE_loss', tensor=self.batch_mse_loss)
        self.train_step = self.get_train_step()
        self.tensorboard_merger = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(config.log_dir + '/train',
                                                  self.session.graph)
        self.test_writer = tf.summary.FileWriter(config.log_dir + '/test')
        self.vali_writer = tf.summary.FileWriter(config.log_dir + '/validation')
        # initialize model saver
        self.saver = tf.train.Saver(
            max_to_keep=int(config.train_steps / config.save_ckpt_period) + 1)
        if is_load_model:
            self.restore_model()
        else:
            self.session.run(tf.global_variables_initializer())
        self.train_handle = self.session.run(self.train_dataset_iter.string_handle())
        self.test_handle = self.session.run(self.test_dataset_iter.string_handle())
        self.validation_handle = self.session.run(self.vali_dataset_iter.string_handle())
        self.session.run(self.train_dataset_iter.initializer)
        self.session.run(self.test_dataset_iter.initializer)
        self.session.run(self.vali_dataset_iter.initializer)

    def train_model(self):
        print('==========================Training========================')
        self.build_graph()
        for i in range(config.train_steps):
            if i <= config.change_lr_step:
                lr = config.initial_lr
            else:
                lr = config.final_lr
            if i % config.evaluate_period == 0:
                summary, loss = self.session.run([self.tensorboard_merger,
                                                  self.batch_mse_loss],
                                                 feed_dict={self.dataset_handle:
                                                                self.validation_handle})
                self.vali_writer.add_summary(summary, i)
                print("Validation set MSE loss at %s: %s" % (i, loss))
            else:
                if i % config.run_meta_period == (config.run_meta_period - 1):
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = self.session.run([self.tensorboard_merger,
                                                   self.train_step],
                                                  feed_dict={self.dataset_handle:
                                                             self.train_handle,
                                                             self.learning_rate_ph: lr},
                                                  options=run_options,
                                                  run_metadata=run_metadata
                                                  )
                    self.train_writer.add_run_metadata(run_metadata,
                                                       'step%03d' % i)
                    self.train_writer.add_summary(summary, i)
                    print('Adding run metadata for', i)
                else:
                    summary, _ = self.session.run([self.tensorboard_merger,
                                                   self.train_step],
                                                  feed_dict={
                                                      self.dataset_handle:
                                                          self.train_handle,
                                                      self.learning_rate_ph: lr})
                    self.train_writer.add_summary(summary, i)
            if i % config.save_ckpt_period == 0 and i != 0:
                self.save_model(str(i))
        self.train_writer.close()
        self.vali_writer.close()
        self.test_writer.close()
        test_loss = self.session.run(self.batch_mse_loss,
                                     feed_dict={self.dataset_handle:
                                                    self.test_handle})
        print('\nFinal test MSE loss is %s' % test_loss)
        self.save_model('_final')
        self.close_session()

    def test_model(self):
        print('============================Test============================')
        self.build_graph(is_train=False, is_load_model=True)
        loss, outputs = self.session.run([self.batch_mse_loss,
                                          self.batch_outputs],
                                         feed_dict={self.dataset_handle:
                                                    self.test_handle})
        outputs = self.loaded_data.label_inverse_normalize(outputs)
        print('MSE loss on the test set is %s' % loss)
        print('Saving data')
        with open(config.test_results_dir + '/predicted_mceps.pkl', 'wb') as f:
            pickle.dump(outputs, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(config.test_results_dir + '/test_ground_truth.pkl', 'wb') as f:
            pickle.dump(self.loaded_data.test_y, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(config.test_results_dir + '/test_ppgs.pkl', 'wb') as f:
            pickle.dump(self.loaded_data.test_x, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(config.test_results_dir + '/test_uttids.pkl', 'wb') as f:
            pickle.dump(self.loaded_data.test_uttids, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.loaded_data.test_uttids, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(config.test_results_dir + '/test_lengths.pkl', 'wb') as f:
            pickle.dump(self.loaded_data.test_lens, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.close_session()

    def inference_model(self):
        self.build_graph(is_train=False, is_load_model=True)
        if config.src_test_size % self.ds.test_batch_size:
            num_iter = int(config.src_test_size // self.ds.test_batch_size) + 1
        else:
            num_iter = int(config.src_test_size / self.ds.test_batch_size)
        mceps = []
        for i in range(num_iter):
            outputs = self.session.run(self.batch_outputs,
                                       feed_dict={self.dataset_handle:
                                                  self.test_handle})
            mceps.append(outputs)
        predicted_mceps = [self.loaded_data.label_inverse_normalize(mcep)
                           for mcep in mceps]
        print('Saving data')
        with open(config.test_results_dir + '/predicted_mceps.pkl', 'wb') as f:
            pickle.dump(predicted_mceps, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(config.test_results_dir + '/test_ground_truth.pkl', 'wb') as f:
            pickle.dump(self.loaded_data.test_y, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(config.test_results_dir + '/test_ppgs.pkl', 'wb') as f:
            pickle.dump(self.loaded_data.test_x, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(config.test_results_dir + '/test_uttids.pkl', 'wb') as f:
            pickle.dump(self.loaded_data.test_uttids, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(config.test_results_dir + '/test_lengths.pkl', 'wb') as f:
            pickle.dump(self.loaded_data.test_lens, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.close_session()

    def save_model(self, name):
        # name is model dir's suffix
        if self.session is None:
            raise MyError('No Session initialized!')
        else:
            save_path = self.saver.save(self.session,
                                        config.model_dir + name + '/model.ckpt')
            print('model saved in %s' % save_path)

    def restore_model(self):
        model_dir = config.ckpt_dir
        if tf.gfile.Exists(model_dir):
            self.saver.restore(self.session, model_dir + '/model.ckpt')
            print('Model restored')
        else:
            print('No saved model found!')
            raise MyError('No saved model found!')

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float64)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.random_normal(shape=shape, dtype=tf.float64)
        return tf.Variable(initial)

    @staticmethod
    def bidirectional_lstm(inputs, seq_lens, number_hiddens):
        fw_cell_lst = [rnn.BasicLSTMCell(num_hidden)
                       for num_hidden in number_hiddens]
        bw_cell_lst = [rnn.BasicLSTMCell(num_hidden)
                       for num_hidden in number_hiddens]
        outputs, states_fw, states_bw = \
            rnn.stack_bidirectional_dynamic_rnn(
                fw_cell_lst,
                bw_cell_lst,
                inputs,
                dtype=tf.float64,
                sequence_length=seq_lens)
        return outputs

    @staticmethod
    def bidirectional_gru(inputs, seq_lens, number_hiddens):
        fw_cell_lst = [rnn.GRUCell(num_hidden)
                       for num_hidden in number_hiddens]
        bw_cell_lst = [rnn.GRUCell(num_hidden)
                       for num_hidden in number_hiddens]
        outputs, states_fw, states_bw = \
            rnn.stack_bidirectional_dynamic_rnn(
                fw_cell_lst,
                bw_cell_lst,
                inputs,
                dtype=tf.float64,
                sequence_length=seq_lens)
        return outputs

    def fc(self, inputs, input_dim, output_dim, activation='identity'):
        # pre_fc_dim = config.rnn_hidden_size[-1]*2
        # outputs_dim = config.output_feature_dim
        with tf.name_scope('fc'):
            with tf.name_scope('weights_and_biases'):
                w_var = self.weight_variable(
                    shape=[input_dim, output_dim])
                b_var = self.bias_variable(
                    shape=[output_dim])
        final_outputs = tf.add(
            tf.tensordot(inputs, w_var, axes=1), b_var)
        if activation == 'identity':
            return final_outputs
        elif activation == 'tanh':
            return tf.nn.tanh(final_outputs)
        elif activation == 'relu':
            return tf.nn.relu(final_outputs)
        else:
            raise MyError('Not defined activation type!')

    def model(self, inputs, seq_lens, drop_rate, is_train=True):
        pre_lstm_outputs = self.fc(inputs,
                                   input_dim=config.input_feature_dim,
                                   output_dim=config.input_act_dim,
                                   activation='tanh')
        
        # add dropout layer
        pre_lstm_outputs = tf.layers.dropout(pre_lstm_outputs,
                                             rate=drop_rate,
                                             training=is_train)

        rnn_outputs = self.bidirectional_gru(
            pre_lstm_outputs, seq_lens, config.rnn_hidden_size)
        # dropout layer
        rnn_outputs = tf.layers.dropout(rnn_outputs,
                                        rate=drop_rate,
                                        training=is_train)
        outputs = self.fc(rnn_outputs,
                          input_dim=config.rnn_hidden_size[-1] * 2,
                          output_dim=config.output_feature_dim,
                          activation='identity')
        return outputs

    def model_prefc(self, inputs, seq_lens):
        pre_lstm_outputs = self.fc(inputs,
                                   input_dim=config.input_feature_dim,
                                   output_dim=config.input_act_dim,
                                   activation='tanh')
        lstm_outputs = self.bidirectional_lstm(
            pre_lstm_outputs,
            seq_lens,
            number_hiddens=config.rnn_hidden_size)
        outputs = self.fc(
            lstm_outputs,
            input_dim=config.rnn_hidden_size[-1] * 2,
            output_dim=config.output_feature_dim,
            activation='identity')
        return outputs

    def mcd_loss(self):
        with tf.name_scope('mcd_loss'):
            loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.sqrt(
                        tf.reduce_sum(
                            tf.square(
                                tf.subtract(self.batch_labels,
                                            self.batch_outputs)
                            ),
                            axis=-1)
                    ),
                    axis=-1)
            )
        return loss

    def mse_loss(self):
        with tf.name_scope('MSE_Loss'):
            mask = tf.expand_dims(
                tf.sequence_mask(self.batch_lengths, dtype=tf.float64), -1)
            mse_loss = tf.losses.mean_squared_error(
                self.batch_labels, self.batch_outputs, weights=mask)
        return mse_loss

    def mse_loss_normalized(self):
        with tf.name_scope('MSE_Loss'):
            mask = tf.sequence_mask(self.batch_lengths, dtype=tf.float64)
            squared_error_sum = tf.reduce_sum(
                tf.squared_difference(self.batch_labels, self.batch_outputs),
                axis=-1)
            masked_squared_error_sum = tf.multiply(squared_error_sum, mask)
            batch_mse_loss = \
                tf.reduce_sum(masked_squared_error_sum,
                              axis=-1) / tf.cast(self.batch_lengths,
                                                 tf.float64)
        return tf.reduce_mean(batch_mse_loss)

    def get_train_step(self):
        if config.optimizer_type == 'adam':
            train_step = \
                tf.train.AdamOptimizer(
                    self.learning_rate_ph).minimize(self.batch_mse_loss)
            print('Optimizer is Adam')
        elif config.optimizer_type == 'adadelta':
            train_step = \
                tf.train.AdadeltaOptimizer(
                    self.learning_rate_ph).minimize(self.batch_mse_loss)
            print('Optimizer is Adadelta')
        else:
            train_step = \
                tf.train.GradientDescentOptimizer(
                    self.learning_rate_ph).minimize(self.batch_mse_loss)
            print('Optimizer is GSD')
        return train_step

    # def saver
