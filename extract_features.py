import os
import json
import math
import uuid
import wespeaker
import soundfile as sf
import portion as P
import numpy as np

from tqdm import tqdm

wespeaker_model2 = wespeaker.load_model_local('/media/data1/avdeeva/models/voxblink2_samresnet34')
wespeaker_model2.set_device('cuda:0')


def process_wav_wespeaker2(wav_path, **kwargs):
    embedding = wespeaker_model2.extract_embedding(wav_path)

    return embedding.cpu().numpy()


def process_dir(dir_path, spk_mapping, out_feats_path, out_lab_path):
    win_dur = 1.5
    step = 0.75

    for f in tqdm(os.listdir(dir_path)):

        if not f.endswith('.wav'):
            continue

        f_name = f.split('.')[0]

        wav_data, rate = sf.read(os.path.join(dir_path, f))
        rttm_path = os.path.join(dir_path, f.replace('wav', 'rttm'))

        win_len = math.ceil(win_dur * rate)
        step_len = math.ceil(step * rate)

        intervals_lst = []
        wav_labels = []

        chunks_data = []
        chunks_lab = []

        with open(rttm_path) as in_file:
            for line in in_file:
                cc = line.strip().split(' ')
                st, dur = cc[3], cc[4]
                st_frame = math.floor(float(st) * rate)
                end_frame = math.ceil(st_frame + float(dur) * rate)
                spk_id = spk_mapping[cc[7]]
                wav_labels.append(spk_id)
                intervals_lst.append(P.open(st_frame, end_frame))

        num_chunks = len(wav_data) // step_len
        for i in range(num_chunks):
            data_chunk = wav_data[i * step_len: i * step_len + win_len]
            vec_int = P.open(i * step_len, i * step_len + win_len)
            f_path = '{}.wav'.format(str(uuid.uuid4()))
            sf.write(f_path, data_chunk, rate)
            vec = process_wav_wespeaker2(f_path)
            os.remove(f_path)

            chunks_data.append(vec)

            for idx, p in enumerate(intervals_lst):
                if vec_int.overlaps(p):
                    if vec_int in p:
                        spk_lab = wav_labels[idx]
                        break
                    else:
                        curr_intersection = vec_int.intersection(p)
                        intersection_len = curr_intersection.upper - curr_intersection.lower
                        if intersection_len >= step_len:
                            spk_lab = wav_labels[idx]
                            break

            chunks_lab.append(spk_lab)

        assert len(chunks_lab) == len(chunks_data)

        chunks_data = np.stack(chunks_data)
        np.save(os.path.join(out_feats_path, f_name + '.npy'),
                chunks_data)

        np.save(os.path.join(out_lab_path, f_name + '.npy'),
                np.array(chunks_lab))


def make_spk_mapping():
    whsp_dir = '../custom_datasets/orig_data/vox_whisper_speakers_v1'
    regular_dir = '../custom_datasets/orig_data/vox_regular_speakers_v2'
    out_path = '../custom_datasets/mixed_data/meta/spk_mapping.json'
    spk_set = set()

    for d in [whsp_dir, regular_dir]:
        for f in os.listdir(d):
            spk_set.add(f.split('_')[0])

    spk2id = {}
    for idx, spk in enumerate(spk_set):
        spk2id[spk] = idx

    json.dump(spk2id, open(out_path, 'w'))


if __name__ == '__main__':
    _dir_path = '../custom_datasets/mixed_data/vox_mix'
    _out_feats_path = '../custom_datasets/mixed_data/feats'
    _out_lab_path = '../custom_datasets/mixed_data/meta/labels'
    _spk_mapping_path = '../custom_datasets/mixed_data/meta/spk_mapping.json'
    data = json.load(open(_spk_mapping_path))
    make_spk_mapping()
    process_dir(_dir_path, data, _out_feats_path, _out_lab_path)
