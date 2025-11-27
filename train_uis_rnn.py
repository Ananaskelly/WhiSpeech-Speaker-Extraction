import os
import sys
import pickle
import numpy as np
from pyannote.metrics.diarization import DiarizationErrorRate


sys.path.append('../')

from uis_rnn_module import uisrnn


if __name__ == '__main__':
    model_args, training_args, inference_args = uisrnn.parse_arguments()
    model_args.enable_cuda = True
    model_args.observation_dim = 256
    model = uisrnn.UISRNN(model_args)
    print(model.device)
    model.verbose = 3

    features_path = '../custom_datasets/mixed_data/feats'
    labels_path = '../custom_datasets/mixed_data/meta/labels'

    # features = pickle.load(open(features_path, 'rb'))
    # labels = pickle.load(open(labels_path, 'rb'))
    features = []
    labels = []
    for f in os.listdir(features_path):
        f_path = os.path.join(features_path, f)
        l_path = os.path.join(labels_path, f)
        features.append(np.load(f_path))
        labels.append(np.load(l_path))

    features = np.vstack(features).astype(float)
    labels = np.hstack(labels).astype(str)
    print(features.shape, labels.shape)
    # лейблы уникальны на весь датасет
    training_args.enforce_cluster_id_uniqueness = False
    model.fit(train_sequences=features,
              train_cluster_ids=labels,
              args=training_args)

    model.save('./outputs/uis_rnn_v1.pth')
