import numpy as np
import pyworld as pw
import librosa
import pickle
import config
import pysptk


def load_pkl(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def scp2dict(scp_file=config.src_scp_dir):
    scp_dict = {}
    with open(scp_file, 'r') as f:
        for line in f:
            lst = line.split()
            scp_dict[lst[0]] = lst[1]
    return scp_dict


def mceps2sp(mcep, alpha=0.42, fftlen=1024):
    """
    :param mceps: list of utterance's mcep
    :param alpha: 0.42 for 16k sample rate
    :param fftlen: 1024 for 513 sp
    :return: utterance's list
    """
    sp = np.apply_along_axis(pysptk.conversion.mc2sp, 1, mcep, alpha, fftlen)
    return sp


def data_merge(data):
    data_new = []
    for item in data:
        for sub_item in item:
            data_new.append(sub_item)
    return data_new


def save_pkl(var, save_name):
    with open(save_name, 'wb') as f:
        pickle.dump(var, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved var into {}'.format(save_name))


def read_f0_from_f(f0_f):
    with open(f0_f, 'r') as f:
        f0_lst = list()
        content = f.readlines()
        for i in range(7, len(content)):
            lst = content[i].split()
            if lst[1] == '1':
                f0_lst.append(float(lst[2]))
            else:
                f0_lst.append(0.0)
    return f0_lst


def read_f0_via_id(utt_id, length=None):
    spk = utt_id.split('_')[0]
    if spk not in ['bdl', 'slt', 'rms']:
        raise ValueError('Undefined speaker!')
    f0_f = '/home/lh17/project/cmu_arctic/s5/f0_extraction/'+spk+'/hop_5ms/'+utt_id+'.f0'
    f0_lst = read_f0_from_f(f0_f)
    if length is not None:
        pad_n = length - len(f0_lst)
        f0_lst = np.pad(f0_lst, (0, pad_n), 'constant')
    return f0_lst


def f0_transform(f0_arr,
                 tgt_mean=config.f0_tgt_mean,
                 tgt_std=config.f0_tgt_std):
    voiced_ind = np.where(f0_arr)[0]
    vuv_mask = np.zeros(f0_arr.size, dtype=np.float32)
    vuv_mask[voiced_ind] = 1.0
    voiced = f0_arr[voiced_ind]
    mean = np.mean(voiced)
    std = np.std(voiced)
    f0_normal = (f0_arr-mean) / std
    f0_transformed = f0_normal*tgt_std + tgt_mean
    f0_transformed = f0_transformed * vuv_mask
    return f0_transformed


def mk_data_dict():
    results_dir = config.test_results_dir
    predicted_mceps = load_pkl(results_dir + '/predicted_mceps.pkl')
    predicted_mceps = data_merge(predicted_mceps)
    uttids = load_pkl(results_dir + '/test_uttids.pkl')
    lengths = load_pkl(results_dir + '/test_lengths.pkl')
    mcep_dict = {}
    for i in range(len(uttids)):
        utt_len = lengths[i]
        mcep = predicted_mceps[i][:utt_len, :]
        mcep_dict[uttids[i]] = mcep
    save_pkl(mcep_dict, results_dir + '/mcep_dict.pkl')
    return


def synthesis(resyn=False):
    results_dir = config.test_results_dir
    predicted_mceps = load_pkl(results_dir+'/predicted_mceps.pkl')
    predicted_mceps = data_merge(predicted_mceps)
    uttids = load_pkl(results_dir+'/test_uttids.pkl')
    lengths = load_pkl(results_dir+'/test_lengths.pkl')
    scp_dict = scp2dict()
    data_size = 10  # len(lengths)
    # get f0 range
    if 'bdl' in uttids[0]:
        f0_floor = 30.0
        f0_ceil = 300.0
    elif 'rms' in uttids[0]:
        f0_floor = 30.0
        f0_ceil = 300.0
    elif 'slt' in uttids[0]:
        f0_floor = 70.0
        f0_ceil = 500.0
    else:
        print('Unknown speaker! Check if something Wrong!!!')
        f0_floor = 40.0
        f0_ceil = 600.0
    src_spk = uttids[0].split('_')[0]
    tgt_spk = config.tgt_data_dir.split('/')[-1]
    for i in range(data_size):
        uttid = uttids[i]
        utt_len = lengths[i]
        sp_predict = mceps2sp(predicted_mceps[i][:utt_len, :])
        wav_arr, sr = librosa.load(scp_dict[uttid], sr=None, dtype=np.float64)
        _, t = pw.harvest(wav_arr, sr, f0_floor, f0_ceil)
        f0_raw = read_f0_via_id(uttid, utt_len)
        ap = pw.d4c(wav_arr, f0_raw, t, sr)
        if src_spk != tgt_spk:
            f0_t = f0_transform(f0_raw)
        else:
            f0_t = f0_raw
        y_predict = pw.synthesize(
            f0_t, sp_predict, ap[:utt_len, :], sr, pw.default_frame_period)
        y_predict = y_predict.astype(np.float32)
        librosa.output.write_wav(results_dir+'/'+uttid+'_predict.wav', y_predict, sr)
        if resyn:
            sp = pw.cheaptrick(wav_arr, f0_raw, t, sr)
            y_resyn = pw.synthesize(f0_raw, sp, ap, sr, pw.default_frame_period)
            y_resyn = y_resyn.astype(np.float32)
            librosa.output.write_wav(results_dir+'/'+uttid+'_resyn.wav', y_resyn, sr)
            print('Resynthesized %s groundtruth wav files!' % (i+1))
        print('Synthesized %s wav files!' % (i+1))


if __name__ == "__main__":
    # synthesis(True)
    mk_data_dict()
