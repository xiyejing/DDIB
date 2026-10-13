import numpy as np
import torch
from IEMOCAP_DATA import IEMOCAPDataset
from CMUMOSI_MOSEI_DATA import CMUSentimentDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from DDIB_Model import DDIBModel
import config

def get_loaders(audio_root, text_root, video_root, num_folder, dataset, batch_size):
    if dataset in ['CMUMOSI', 'CMUMOSEI']:
        sentiment = CMUSentimentDataset(
            label_path=config.PATH_TO_LABEL[dataset],
            audio_root=audio_root,
            text_root=text_root,
            video_root=video_root,
        )
        train_idxs, val_idxs, test_idxs = sentiment.split_indices()
        train_loader = DataLoader(
            sentiment, batch_size=batch_size, sampler=SubsetRandomSampler(train_idxs),
            collate_fn=sentiment.collate_fn, pin_memory=False,
        )
        val_loader = DataLoader(
            sentiment, batch_size=batch_size, sampler=SubsetRandomSampler(val_idxs),
            collate_fn=sentiment.collate_fn, pin_memory=False,
        )
        test_loader = DataLoader(
            sentiment, batch_size=batch_size, sampler=SubsetRandomSampler(test_idxs),
            collate_fn=sentiment.collate_fn, pin_memory=False,
        )
        adim, tdim, vdim = sentiment.get_featDim()
        # Attach the validation loader without changing the train/test interface.
        train_loader.validation_loader = val_loader
        return [train_loader], [test_loader], adim, tdim, vdim

    if dataset in ['IEMOCAPFOUR', 'IEMOCAPSIX']:
        dataset = IEMOCAPDataset(label_path=config.PATH_TO_LABEL[dataset],
                                 audio_root=audio_root,
                                 text_root=text_root,
                                 video_root=video_root)

        session_to_idx = {}
        for idx, vid in enumerate(dataset.vids):
            session = int(vid[4]) - 1
            if session not in session_to_idx:
                session_to_idx[session] = []
            session_to_idx[session].append(idx)
        assert len(session_to_idx) == num_folder, f'Must split into five folder'

        train_test_idxs = []
        for ii in range(num_folder):
            test_idxs = session_to_idx[ii]
            train_idxs = []
            for jj in range(num_folder):
                if jj != ii:
                    train_idxs.extend(session_to_idx[jj])
            train_test_idxs.append([train_idxs, test_idxs])

        train_loaders = []
        test_loaders = []
        for ii in range(len(train_test_idxs)):
            train_idxs = train_test_idxs[ii][0]
            test_idxs = train_test_idxs[ii][1]
            train_loader = DataLoader(dataset,
                                      batch_size=batch_size,
                                      sampler=SubsetRandomSampler(train_idxs),
                                      collate_fn=dataset.collate_fn,
                                      pin_memory=False)
            test_loader = DataLoader(dataset,
                                     batch_size=batch_size,
                                     sampler=SubsetRandomSampler(test_idxs),
                                     collate_fn=dataset.collate_fn,
                                     pin_memory=False)
            train_loaders.append(train_loader)
            test_loaders.append(test_loader)

        adim, tdim, vdim = dataset.get_featDim()
        return train_loaders, test_loaders, adim, tdim, vdim

    raise ValueError(f'Unsupported dataset: {dataset}')


def build_model(args, adim, tdim, vdim):
    D_e = args.hidden
    model = DDIBModel(args,
                   adim, tdim, vdim, D_e,
                  n_classes=args.n_classes,
                  depth=args.depth, num_heads=args.num_heads, mlp_ratio=1, drop_rate=args.drop_rate,
                  attn_drop_rate=args.attn_drop_rate,
                  no_cuda=args.no_cuda)
    return model


def random_mask(view_num, input_len, missing_rate):
    """Random missing protocol: keep at least one modality."""
    one_rate = 1.0 - float(missing_rate)
    if one_rate <= (1.0 / view_num):
        return np.eye(view_num, dtype=np.int64)[np.random.randint(0, view_num, size=input_len)]
    if one_rate >= 1.0:
        return np.ones((input_len, view_num), dtype=np.int64)

    alldata_len = max(int(input_len), 32)
    error = 1.0
    while error >= 0.005:
        view_preserve = np.eye(view_num, dtype=np.int64)[np.random.randint(0, view_num, size=alldata_len)]
        one_num = view_num * alldata_len * one_rate - alldata_len
        ratio = one_num / (view_num * alldata_len)
        matrix_iter = (np.random.randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int64)
        overlap = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int64))
        one_num_iter = one_num / max(1e-8, (1.0 - overlap / max(one_num, 1e-8)))
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (np.random.randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int64)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int64)
        error = abs(one_rate - np.sum(matrix) / (view_num * alldata_len))
    return matrix[:input_len, :]


def generate_mask(seqlen, batch, test_condition, first_stage, mask_rate=None, mask_first_stage=False):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    """
    use_random_mask = mask_rate is not None and ((not first_stage) or mask_first_stage)
    if use_random_mask:
        assert 0.0 <= float(mask_rate) <= 0.700001, '3-view missing rate must be in [0, 0.7]'
        matrix = random_mask(3, seqlen * batch, float(mask_rate))
        return [matrix[:, 0], matrix[:, 1], matrix[:, 2]]

    if first_stage:
        audio_mask = np.array([1])
        text_mask = np.array([1])
        visual_mask = np.array([1])
    elif test_condition == 'all':
        mask_options = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ])
        sampled = mask_options[np.random.randint(0, len(mask_options), size=seqlen * batch)]
        return [sampled[:, 0], sampled[:, 1], sampled[:, 2]]
    else:
        audio_mask = np.array([1 if 'a' in test_condition else 0])
        text_mask = np.array([1 if 't' in test_condition else 0])
        visual_mask = np.array([1 if 'v' in test_condition else 0])
    audio_mask = audio_mask.repeat(seqlen * batch)
    text_mask = text_mask.repeat(seqlen * batch)
    visual_mask = visual_mask.repeat(seqlen * batch)

    matrix = [audio_mask, text_mask, visual_mask]
    return matrix


def generate_inputs(audio_host, text_host, visual_host, audio_guest, text_guest, visual_guest, qmask):
    input_features = []
    feat1 = torch.cat([audio_host, text_host, visual_host], dim=2) # [seqlen, batch, featdim=adim+tdim+vdim]
    feat2 = torch.cat([audio_guest, text_guest, visual_guest], dim=2)
    featdim = feat1.size(-1)
    tmask = qmask.transpose(0, 1) # [batch, seqlen] -> [seqlen, batch]
    tmask = tmask.unsqueeze(2).repeat(1,1,featdim) # -> [seqlen, batch, featdim]
    select_feat = torch.where(tmask==0, feat1, feat2) # -> [seqlen, batch, featdim]
    input_features.append(select_feat) # 1 * [seqlen, batch, dim]
    return input_features
