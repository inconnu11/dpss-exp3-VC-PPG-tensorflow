import os
# some parameters' configuration

# input and output setting
input_feature_dim = 128
output_feature_dim = 40

# input after activation dim
input_act_dim = 128

# learning rate
initial_lr = 1e-3
final_lr = 1e-3

# Dropout rate
dropout_rate = 0.1

# change learning rate on the step
change_lr_step = 2500

# batch_size
batch_size = 64
vali_batch_size = None
test_batch_size = 128
train_steps = 20000

# evaluating period
evaluate_period = 10
run_meta_period = 10000

# gpu settings
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
gpu_allow_grouth = True

# network setting
rnn_hidden_size = [128, 128]

# training setting
optimizer_type = 'adam'  # can also be 'sgd'

# data dir
# under linux
tgt_data_dir = './data/clb'
# under widows
tgt_scp_dir = './data/clb/wav.scp'

# source speaker data dir
src_data_dir = './data/rms'
src_scp_dir = './data/rms/wav.scp'
src_test_size = 10
src_vali_size = 100

# test result dir
test_results_dir = './test_results'

# log dir
log_dir = './log'

# save ckpt period
save_ckpt_period = 1000

# model dir
model_dir = './model'
# dir only for load model, should change every time test
ckpt_dir = './model_final'

# load model from ckpt or not
is_load_ckpt = False

# test set and validation set setting
tgt_test_set_size = 10  # must be greater than 0
tgt_validition_set_size = 100  # must be greater than 0
is_shuffle = False

# target speaker f0 mean and std
# for slt
# f0_tgt_mean = 173.6955153061961
# f0_tgt_std = 23.01384157628696
# # for bdl
#f0_tgt_mean = 117.34915409597042
#f0_tgt_std = 25.73168782557772
# # for rms
# f0_tgt_mean = 92.60950447063425
# f0_tgt_std = 21.5205740661971
# # for clb
f0_tgt_mean = 179.30753196022624
f0_tgt_std = 24.672621095260492
# data set normalization settings
feature_normalization_scaler_file = 'feature_normalize_scaler.pkl'
label_normalization_scaler_file = 'label_normalize_scaler.pkl'
