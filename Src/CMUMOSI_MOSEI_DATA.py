import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def read_data(label_path, feature_root):
    with open(label_path, "rb") as handle:
        video_ids, _, _, _, _, _, _ = pickle.load(handle, encoding="latin1")

    names = [uid for vid in video_ids for uid in video_ids[vid]]
    name_to_feature = {}
    feature_dim = None
    for name in names:
        npy_path = os.path.join(feature_root, name + ".npy")
        dir_path = os.path.join(feature_root, name)
        if os.path.isfile(npy_path):
            feature = np.asarray(np.load(npy_path)).squeeze()
            if feature.ndim == 2:
                feature = feature.mean(axis=0)
        elif os.path.isdir(dir_path):
            frames = [np.load(os.path.join(dir_path, item)) for item in sorted(os.listdir(dir_path))]
            if not frames:
                raise ValueError(f"Empty feature directory: {dir_path}")
            feature = np.asarray(frames).squeeze()
            if feature.ndim == 2:
                feature = feature.mean(axis=0)
        else:
            raise FileNotFoundError(f"Missing feature for {name} under {feature_root}")
        feature = np.asarray(feature, dtype=np.float32).reshape(-1)
        feature_dim = feature.shape[-1] if feature_dim is None else feature_dim
        if feature.shape[-1] != feature_dim:
            raise ValueError(f"Inconsistent feature dimension for {name}: {feature.shape[-1]} != {feature_dim}")
        name_to_feature[name] = feature
    return name_to_feature, int(feature_dim)


class CMUSentimentDataset(Dataset):
    """Conversation-level MOSI/MOSEI features with official train/val/test splits."""

    def __init__(self, label_path, audio_root, text_root, video_root):
        name_to_audio, self.adim = read_data(label_path, audio_root)
        name_to_text, self.tdim = read_data(label_path, text_root)
        name_to_video, self.vdim = read_data(label_path, video_root)
        with open(label_path, "rb") as handle:
            (self.video_ids, self.video_labels, self.video_speakers, _,
             self.train_vids, self.val_vids, self.test_vids) = pickle.load(handle, encoding="latin1")

        self.vids = sorted(self.train_vids) + sorted(self.val_vids) + sorted(self.test_vids)
        self.audio, self.text, self.video, self.labels = {}, {}, {}, {}
        for vid in self.vids:
            uids = self.video_ids[vid]
            self.audio[vid] = np.stack([name_to_audio[uid] for uid in uids])
            self.text[vid] = np.stack([name_to_text[uid] for uid in uids])
            self.video[vid] = np.stack([name_to_video[uid] for uid in uids])
            self.labels[vid] = np.asarray(self.video_labels[vid], dtype=np.float32)

    def __getitem__(self, index):
        vid = self.vids[index]
        length = len(self.labels[vid])
        # MOSI and MOSEI contain one speaker stream; the guest stream is zero-filled.
        return (
            torch.from_numpy(self.audio[vid]),
            torch.from_numpy(self.text[vid]),
            torch.from_numpy(self.video[vid]),
            torch.zeros(length, self.adim),
            torch.zeros(length, self.tdim),
            torch.zeros(length, self.vdim),
            torch.zeros(length),
            torch.ones(length),
            torch.from_numpy(self.labels[vid]),
            vid,
        )

    def __len__(self):
        return len(self.vids)

    def get_featDim(self):
        return self.adim, self.tdim, self.vdim

    def split_indices(self):
        index = {vid: idx for idx, vid in enumerate(self.vids)}
        return (
            [index[vid] for vid in sorted(self.train_vids)],
            [index[vid] for vid in sorted(self.val_vids)],
            [index[vid] for vid in sorted(self.test_vids)],
        )

    @staticmethod
    def collate_fn(data):
        dat = pd.DataFrame(data)
        result = []
        for column in dat:
            if column <= 5:
                result.append(pad_sequence(dat[column]))
            elif column <= 8:
                result.append(pad_sequence(dat[column], batch_first=True))
            else:
                result.append(dat[column].tolist())
        return result
