import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.externals import joblib
import pickle
import config

__package__ = 'load_data'


class MyError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class LoadedData:
    def __init__(self):
        # training set
        self.train_x = None
        self.train_y = None
        self.train_lens = None
        self.train_uttids = None
        self.train_set_size = None
        # validating set
        self.vali_x = None
        self.vali_y = None
        self.vali_lens = None
        self.vali_uttids = None
        self.vali_set_size = None
        # testing data
        self.test_x = None
        self.test_y = None
        self.test_lens = None
        self.test_uttids = None
        self.test_set_size = None
        # data normalization
        self.feature_scaler = None
        self.feature_scaler_file = None
        self.label_scaler = None
        self.label_scaler_file = None

    # normalization
    def feature_normalize(self):
        if self.feature_scaler is None:
            train_x_np = np.vstack(self.train_x)
            scaler = StandardScaler().fit(train_x_np)
            self.feature_scaler = scaler
            self.feature_scaler_file = config.feature_normalization_scaler_file
            joblib.dump(scaler, self.feature_scaler_file)
            print('feature normalization scaler saved!')
        print('==================Doing feature normalization=================')
        print('Data mean value is %s\n' % self.feature_scaler.mean_)
        print('Data variance is %s\n' % self.feature_scaler.var_)
        self.train_x = [self.feature_scaler.transform(x) for x in self.train_x]
        print('normalized training data')
        if self.vali_x:
            self.vali_x = [self.feature_scaler.transform(x) for x in self.vali_x]
            print('Normalized validation data!')
        if self.test_x:
            self.test_x = [self.feature_scaler.transform(x) for x in self.test_x]
            print('Normalized test data!')

    # label normalization
    def label_normalize(self):
        if self.label_scaler is None:
            train_y = np.vstack(self.train_y)
            scaler = MaxAbsScaler().fit(train_y)
            self.label_scaler = scaler
            self.label_scaler_file = config.label_normalization_scaler_file
            joblib.dump(self.label_scaler, self.label_scaler_file)
            print('label normalization scaler saved!')
        print('==================Doing label normalization=================')
        self.train_y = [self.label_scaler.transform(y) for y in self.train_y]
        print('Scaled training data')
        if self.vali_y:
            self.vali_y = [self.label_scaler.transform(y) for y in self.vali_y]
        print('Scaled validation data')
        if self.test_y:
            self.test_y = [self.label_scaler.transform(y) for y in self.test_y]
        print('Scaled test data')

    def get_feature_scaler_from_file(self,
                                     filename=config.feature_normalization_scaler_file):
        if not tf.gfile.Exists(filename):
            print('No feature scaler file found!')
            raise MyError('Label feature file Not Exists!')
        self.feature_scaler_file = filename
        self.feature_scaler = joblib.load(filename)

    def get_label_scaler_from_file(self,
                                   filename=config.label_normalization_scaler_file):
        if not tf.gfile.Exists(filename):
            print('No label scaler file found!')
            raise MyError('Label scaler file Not Exists!')
        self.label_scaler_file = filename
        self.label_scaler = joblib.load(filename)

    def feature_inverse_normalize(self, data):
        """
        :param data: data should be a list of 2D array
        :return:
        """
        if self.feature_scaler is None:
            self.get_feature_scaler_from_file()
        data_inv_normal = [self.feature_scaler.inverse_transform(d) for d in data]
        return data_inv_normal

    def label_inverse_normalize(self, data):
        """
        :param data: data should be a list of 2D array
        :return:
        """
        if self.label_scaler is None:
            self.get_label_scaler_from_file()
        data_inv_normal = [self.label_scaler.inverse_transform(d) for d in data]
        return data_inv_normal

    # sorting
    def sort_train(self):
        train_list = [(x, y, l, i) for x, y, l, i in
                      zip(self.train_x, self.train_y,
                          self.train_lens, self.train_uttids)]
        train_list = sorted(train_list, key=lambda x: x[2])
        self.train_x = [ele[0] for ele in train_list]
        self.train_y = [ele[1] for ele in train_list]
        self.train_lens = [ele[2] for ele in train_list]
        self.train_uttids = [ele[3] for ele in train_list]
        print('Sorted training set by utterance length')

    def sort_test(self):
        test_list = [(x, y, l, i) for x, y, l, i in
                     zip(self.test_x, self.test_y,
                         self.test_lens, self.test_uttids)]
        test_list = sorted(test_list, key=lambda x: x[2])
        self.test_x = [ele[0] for ele in test_list]
        self.test_y = [ele[1] for ele in test_list]
        self.test_lens = [ele[2] for ele in test_list]
        self.test_uttids = [ele[3] for ele in test_list]
        print('Sorted test set by utterance length')

    def sort_vali(self):
        vali_list = [(x, y, l, i) for x, y, l, i in
                     zip(self.vali_x, self.vali_y,
                         self.vali_lens, self.vali_uttids)]
        vali_list = sorted(vali_list, key=lambda x: x[2])
        self.vali_x = [ele[0] for ele in vali_list]
        self.vali_y = [ele[1] for ele in vali_list]
        self.vali_lens = [ele[2] for ele in vali_list]
        self.vali_uttids = [ele[3] for ele in vali_list]
        print('Sorted validation set by utterance length')

    def sort_data(self):
        self.sort_train()
        self.sort_test()
        self.sort_vali()
        print('Sorted all data')

    def load_data(self, filename=config.tgt_data_dir,
                  test_set_size=config.tgt_test_set_size,
                  vali_set_size=config.tgt_validition_set_size):
        with open(filename + '/mceps.pkl', 'rb') as f:
            mceps = pickle.load(f)
        with open(filename + '/ppgs_dict', 'rb') as f:
            ppgs = pickle.load(f)
        ppgs_lst = list()
        lengths_lst = list()
        mceps_lst = list()
        utt_ids = list()
        for key in mceps.keys():
            ppg_len = np.shape(ppgs[key])[0]
            ppgs_lst.append(ppgs[key])
            mceps_lst.append(mceps[key][:ppg_len, :])
            lengths_lst.append(ppg_len)
            utt_ids.append(key)
        train_data_tuple = (ppgs_lst, mceps_lst, lengths_lst, utt_ids)
        train_data_tuple, test_data_tuple = self.random_subset(
            train_data_tuple, test_set_size
        )
        train_data_tuple, vali_data_tuple = self.random_subset(
            train_data_tuple, vali_set_size
        )
        self.train_x = train_data_tuple[0]
        self.train_y = train_data_tuple[1]
        self.train_lens = train_data_tuple[2]
        self.train_uttids = train_data_tuple[3]
        self.train_set_size = len(self.train_x)
        self.test_x = test_data_tuple[0]
        self.test_y = test_data_tuple[1]
        self.test_lens = test_data_tuple[2]
        self.test_uttids = test_data_tuple[3]
        self.test_set_size = len(self.test_x)
        self.vali_x = vali_data_tuple[0]
        self.vali_y = vali_data_tuple[1]
        self.vali_lens = vali_data_tuple[2]
        self.vali_uttids = vali_data_tuple[3]
        self.vali_set_size = len(self.vali_x)

    def print_info(self):
        print('===============================================================')
        if self.train_x:
            print('Checking training set:')
            print('training set size:', self.train_set_size)
            print('Example data: utterance id: %s, x shape: %s, y shape: %s, length: %s'
                  % (self.train_uttids[0],
                     np.shape(self.train_x[0]),
                     np.shape(self.train_y[0]),
                     self.train_lens[0]))
            if len(self.train_x) != len(self.train_y) \
                    or len(self.train_y) != len(self.train_lens) \
                    or len(self.train_lens) != len(self.train_uttids):
                print('training set: Something wrong! Data set size not matched')
        else:
            print('No training set')
        print('===============================================================')
        if self.test_x:
            print('Checking test set:')
            print('test set size:', self.test_set_size)
            print('Example data: utterance id: %s, x shape: %s, y shape: %s, length: %s'
                  % (self.test_uttids[0],
                     np.shape(self.test_x[0]),
                     np.shape(self.test_y[0]),
                     self.test_lens[0]))
            if len(self.test_x) != len(self.test_y) \
                    or len(self.test_y) != len(self.test_lens) \
                    or len(self.test_lens) != len(self.test_uttids):
                print('test set: Something wrong! Data set size not matched')
        else:
            print('No test set')
        print('===============================================================')
        if self.vali_x:
            print('Checking validation set:')
            print('validation set size:', self.vali_set_size)
            print('Example data: utterance id: %s, x shape: %s, y shape: %s, length: %s'
                  % (self.vali_uttids[0],
                     np.shape(self.vali_x[0]),
                     np.shape(self.vali_y[0]),
                     self.vali_lens[0]))
            if len(self.vali_x) != len(self.vali_y) \
                    or len(self.vali_y) != len(self.vali_lens) \
                    or len(self.vali_lens) != len(self.vali_uttids):
                print('validation set: Something wrong! Data set size not matched')
        else:
            print('No validation set')
        print('===============================================================')
        with open('data_set_info.txt', 'w') as f:
            f.write('training set data:\n')
            for i in range(self.train_set_size):
                f.write('utt_id: %s, utt_len: %s\n' % (self.train_uttids[i],
                                                       self.train_lens[i]))
            f.write('test set data:\n')
            for i in range(self.test_set_size):
                f.write('utt_id: %s, utt_len: %s\n' % (self.test_uttids[i],
                                                       self.test_lens[i]))
            f.write('validation set data:\n')
            for i in range(self.vali_set_size):
                f.write('utt_id: %s, utt_len: %s\n' % (self.vali_uttids[i],
                                                       self.vali_lens[i]))

    @staticmethod
    def random_subset(data_tuple, test_set_size):
        x_lst = list()
        y_lst = list()
        lens_lst = list()
        id_lst = list()
        index = np.arange(len(data_tuple[0]))
        rand_ind = np.random.permutation(index)[: test_set_size]
        for ind in rand_ind:
            x_lst.append(data_tuple[0][ind])
            y_lst.append(data_tuple[1][ind])
            lens_lst.append(data_tuple[2][ind])
            id_lst.append(data_tuple[3][ind])
        # remove these items from the training data set
        # sort the index generated before
        sorted_ind = np.sort(rand_ind)[::-1]
        for ind in sorted_ind:
            del data_tuple[0][ind]
            del data_tuple[1][ind]
            del data_tuple[2][ind]
            del data_tuple[3][ind]
        subset_tuple = (x_lst, y_lst, lens_lst, id_lst)
        return data_tuple, subset_tuple


if __name__ == '__main__':
    data = LoadedData()
    data.load_data()
    data.sort_data()
    data.print_info()
    # data.get_label_scaler_from_file()
    # print(data.label_scaler)
    # data.label_normalize()
    # max_lst = []
    # min_lst = []
    # mean_lst = []
    # data_y = data.test_y
    # for item in np.abs(data_y):
    #     max_lst.append(np.max(item))
    #     min_lst.append(np.min(item))
    #     mean_lst.append(np.average(item))
    # print('maximum is', max(max_lst))
    # print('minimum is', min(min_lst))
    # print('mean is', np.mean(mean_lst))
    #
    # np.savetxt(data.train_uttids[0]+'.txt', data.train_y[0])

    # import pyworld
    # import pysptk
    # import librosa
    # filename = 'C:\\Users\\LUHUI\\Desktop\\HCSI\\paper\\voice conversion\\cmu_arctic\\cmu_us_bdl_arctic\\wav\\arctic_a0207.wav'
    # wav_arr, sr = librosa.load(filename, sr=None, dtype=np.float64)
    # f0, sp, ap = pyworld.wav2world(wav_arr, sr)
    # mcep_ = np.apply_along_axis(pysptk.conversion.sp2mc, 1, sp, 39, 0.42)
    # np.savetxt('mcep.txt', mcep_)
