from dblstm_model import DblstmModel
import synthesis_test
import tensorflow as tf
import config
import os


def train():
    if tf.gfile.Exists(config.log_dir):
        if tf.gfile.Exists(config.log_dir):
            tf.gfile.DeleteRecursively(config.log_dir)
        tf.gfile.MakeDirs(config.log_dir)
    print('checked log dir')
    model = DblstmModel()
    model.train_model()


def test():
    if not tf.gfile.Exists(config.test_results_dir):
        tf.gfile.MakeDirs(config.test_results_dir)
    print('checked test results dir')
    model = DblstmModel()
    model.test_model()
    # model.inference_model()
    synthesis_test.synthesis(True)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # train()
    test()
