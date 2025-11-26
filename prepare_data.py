import os
import math
import shutil
import librosa
import numpy as np
import soundfile as sf

from tqdm import tqdm


# wespeaker_model2 = wespeaker.load_model_local('/media/data1/avdeeva/models/voxblink2_samresnet34')
# wespeaker_model2.set_device('cuda:0')


# def process_wav_wespeaker2(wav_path, **kwargs):
#     embedding = wespeaker_model2.extract_embedding(wav_path)
#
#     return embedding.cpu().numpy()


def get_mixed_dataset():
    whsp_dir = '../custom_datasets/orig_data/vox_whisper_speakers_v1'
    regular_dir = '../custom_datasets/orig_data/vox_regular_speakers_v2'
    out_wav_dir = '../custom_datasets/mixed_data/vox_mix'
    out_rttm_dir = '../custom_datasets/mixed_data/vox_mix'
    sample_rate = 24000

    spk2utt = {}
    for d in [whsp_dir, regular_dir]:
        for f in os.listdir(d):
            f_path = os.path.join(d, f)
            spk = f.split('_')[0]
            if spk not in spk2utt:
                spk2utt[spk] = []
            spk2utt[spk].append(f_path)
    all_spk_lst = list(spk2utt.keys())

    num_utterances = 10000
    for i in range(num_utterances):
        av_spk_lst = all_spk_lst.copy()

        num_spk = np.random.randint(1, 6)
        total_dur = np.random.uniform(25.0, 45.0)
        f_name = '{}_mix'.format(i)

        out_array = []
        out_rttm_lines = []
        out_f_path = os.path.join(out_wav_dir, f_name + '.wav')
        out_rttm_path = os.path.join(out_rttm_dir, f_name + '.rttm')

        curr_st_time = 0.0
        curr_spk_lst = list()
        curr_spk2utt = {}

        skip = False

        while curr_st_time < total_dur:
            slice_len = math.floor(np.random.uniform(3.0, 7.0) * sample_rate)

            create_new_spk = False
            if len(curr_spk_lst) == 0:
                create_new_spk = True
            elif len(curr_spk_lst) < num_spk:
                prob = np.random.uniform(0.0, 1.0)
                if prob > 0.5:
                    create_new_spk = True
                else:
                    create_new_spk = False

            if create_new_spk:
                current_spk = np.random.choice(av_spk_lst)
                av_spk_lst.remove(current_spk)
                curr_spk_lst.append(current_spk)
                curr_spk2utt[current_spk] = spk2utt[current_spk].copy()
            else:
                current_spk = np.random.choice(curr_spk_lst)

            if len(curr_spk2utt[current_spk]) == 0:
                skip = True
                break

            curr_utt = np.random.choice(curr_spk2utt[current_spk])
            curr_spk2utt[current_spk].remove(curr_utt)
            data, _ = sf.read(curr_utt, always_2d=True)
            data = np.squeeze(data[:, 0])

            if len(data) > slice_len:
                rnd_st = np.random.randint(0, len(data) - slice_len)
                data = data[rnd_st: rnd_st + slice_len]

            data_dur = len(data) / sample_rate
            out_array.append(data)
            out_rttm_lines.append('SPEAKER {} 1 {} {} <NA> <NA> {} <NA>\n'.format(f_name,
                                                                                  curr_st_time,
                                                                                  data_dur,
                                                                                  current_spk))
            curr_st_time += data_dur
        if skip:
            continue
        sf.write(out_f_path, np.hstack(out_array), sample_rate)
        with open(out_rttm_path, 'w') as out_file:
            out_file.writelines(out_rttm_lines)


def get_vox_regular_speakers():
    whsp_dir = '../custom_datasets/orig_data/vox_whisper_speakers_v1'
    regular_dir = '../custom_datasets/orig_data/vox_regular_speakers_v2'
    os.makedirs(regular_dir, exist_ok=True)
    whsp_spk_set = set()

    for file in os.listdir(whsp_dir):
        spk = file.split('_')[0]
        whsp_spk_set.add(spk)

    orig_wav_path = '/media/data1/datasets/decoder_datasets/wav/VOXBLINK/wav24k/'

    spk2utt = {}
    for f in os.listdir(orig_wav_path):
        spk_id = f.split('_')[0]
        if spk_id not in spk2utt:
            spk2utt[spk_id] = []
        spk2utt[spk_id].append(f)

    for spk in whsp_spk_set:
        spk2utt.pop(spk)

    spk_lst = list(spk2utt.keys())
    np.random.shuffle(spk_lst)

    t_num = 42
    curr_num = 0
    idx = 0
    while curr_num != t_num:
        if len(spk2utt[spk_lst[idx]]) < 10:
            idx += 1
            continue
        f_names = spk2utt[spk_lst[idx]]
        for f_name in f_names:
            shutil.copy(os.path.join(orig_wav_path, f_name),
                        os.path.join(regular_dir, f_name))
        idx += 1
        curr_num += 1


def get_vox_whisper_speakers():
    in_path = '../temp_wav_dir_new'
    out_path = '../custom_datasets/orig_data/vox_whisper_speakers_v1'
    os.makedirs(out_path, exist_ok=True)

    orig_wav_path = '/media/data1/datasets/decoder_datasets/wav/VOXBLINK/wav24k/'

    spk2utt = {}
    for f in os.listdir(orig_wav_path):
        spk_id = f.split('_')[0]
        if spk_id not in spk2utt:
            spk2utt[spk_id] = []
        spk2utt[spk_id].append(f)

    whsp_spk2num_utt = {}
    for f in os.listdir(in_path):
        spk_id = f.split('_')[0]

        if spk_id not in whsp_spk2num_utt:
            whsp_spk2num_utt[spk_id] = 0

        whsp_spk2num_utt[spk_id] += 1

    num_spk = 0
    for k, v in tqdm(whsp_spk2num_utt.items()):
        if v > 10:
            all_spk_files = spk2utt[k]
            for f in all_spk_files:
                f_path = os.path.join(orig_wav_path, f)
                shutil.copy(f_path, os.path.join(out_path, f))

            num_spk += 1

    print(num_spk)


def get_duration():
    p1 = '../custom_datasets/orig_data/vox_whisper_speakers_v1'

    total_dur = 0.0
    for f in os.listdir(p1):
        f_path = os.path.join(p1, f)
        dur = librosa.get_duration(path=f_path)
        total_dur += dur

    print(total_dur / 60 / 60)


if __name__ == '__main__':
    # get_vox_regular_speakers()
    get_mixed_dataset()
