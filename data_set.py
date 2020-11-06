import tensorflow as tf
import config
from load_data import LoadedData


class DataSet:
    def __init__(self, loaded_data,
                 batch_size=config.batch_size,):
        # training data set
        train_dataset_x = tf.data.Dataset.from_generator(
            lambda: loaded_data.train_x,
            tf.float64,
            tf.TensorShape([None, config.input_feature_dim])).repeat()
        train_dataset_y = tf.data.Dataset.from_generator(
            lambda: loaded_data.train_y,
            tf.float64,
            tf.TensorShape([None, config.output_feature_dim])).repeat()
        train_dataset_lens = tf.data.Dataset.from_tensor_slices(
            loaded_data.train_lens).repeat()
        train_dataset_ids = tf.data.Dataset.from_tensor_slices(
            loaded_data.train_uttids).repeat()
        self.train_set = tf.data.Dataset.zip(
            (train_dataset_x, train_dataset_y,
             train_dataset_lens, train_dataset_ids)
        )
        if config.is_shuffle:
            self.train_set = self.train_set.shuffle(
                200, reshuffle_each_iteration=True)
        self.train_set = self.train_set.padded_batch(
            batch_size,
            padded_shapes=(
                [None, config.input_feature_dim],
                [None, config.output_feature_dim],
                [], []))
        self.train_iterator = self.train_set.make_initializable_iterator()

        # validation set
        if loaded_data.vali_x:
            vali_dataset_x = tf.data.Dataset.from_generator(
                lambda: loaded_data.vali_x,
                tf.float64,
                tf.TensorShape([None, config.input_feature_dim])).repeat()
            vali_dataset_y = tf.data.Dataset.from_generator(
                lambda: loaded_data.vali_y,
                tf.float64,
                tf.TensorShape([None, config.output_feature_dim])).repeat()
            vali_dataset_lens = tf.data.Dataset.from_tensor_slices(
                loaded_data.vali_lens).repeat()
            vali_dataset_ids = tf.data.Dataset.from_tensor_slices(
                loaded_data.vali_uttids).repeat()
            self.validation_set = tf.data.Dataset.zip(
                (vali_dataset_x, vali_dataset_y,
                 vali_dataset_lens, vali_dataset_ids))
            if config.vali_batch_size is not None:
                self.vali_batch_size = config.vali_batch_size
            else:
                self.vali_batch_size = loaded_data.vali_set_size
            self.validation_set = self.validation_set.padded_batch(
                batch_size=self.vali_batch_size,
                padded_shapes=(
                    [None, config.input_feature_dim],
                    [None, config.output_feature_dim],
                    [], [])
                )
            self.vali_iterator = self.validation_set.make_initializable_iterator()
        else:
            self.vali_iterator = None

        # test set
        if loaded_data.test_x:
            test_dataset_x = tf.data.Dataset.from_generator(
                lambda: loaded_data.test_x,
                tf.float64,
                tf.TensorShape([None, config.input_feature_dim]))
            test_dataset_y = tf.data.Dataset.from_generator(
                lambda: loaded_data.test_y,
                tf.float64,
                tf.TensorShape([None, config.output_feature_dim]))
            test_dataset_lens = tf.data.Dataset.from_tensor_slices(
                loaded_data.test_lens)
            test_dataset_ids = tf.data.Dataset.from_tensor_slices(
                loaded_data.test_uttids)
            self.test_set = tf.data.Dataset.zip(
                (test_dataset_x, test_dataset_y,
                 test_dataset_lens, test_dataset_ids))
            if config.test_batch_size is not None:
                self.test_batch_size = config.test_batch_size
            else:
                self.test_batch_size = loaded_data.test_set_size
            self.test_set = self.test_set.padded_batch(
                batch_size=self.test_batch_size,
                padded_shapes=([None, config.input_feature_dim],
                               [None, config.output_feature_dim],
                               [], []))
            self.test_iterator = self.test_set.make_initializable_iterator()
        else:
            self.test_iterator = None

    def train_gen(self):
        return self.train_iterator

    def vali_gen(self):
        return self.vali_iterator

    def test_gen(self):
        return self.test_iterator


if __name__ == '__main__':
    ld = LoadedData()
    ld.load_data()
    dataset = DataSet(ld, config.batch_size)
    train_iter = dataset.train_gen()
    test_iter = dataset.test_gen()
    if test_iter is None:
        print('yes')
    # x, y, length, utt_id = train_iter.get_next()
    # loss = tf.reduce_sum(x)
    # with tf.Session() as sess:
    #     sess.run(train_iter.initializer)
    #     for i in range(5):
    #         # ids = sess.run(loss)
    #         # print(ids[0])
    #         print(sess.run(loss))
