import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from training_support import (
    TEST_CONDITIONS, build_masked_batch, map_capacity, map_capacity_multiplier,
    rc_modality_rate_from_stats,
)


def linear_cka(x, y):
    if x.size(0) < 2 or y.size(0) < 2:
        return 0.0
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xy = x.transpose(0, 1).matmul(y)
    xx = x.transpose(0, 1).matmul(x)
    yy = y.transpose(0, 1).matmul(y)
    denom = torch.linalg.norm(xx, ord='fro') * torch.linalg.norm(yy, ord='fro')
    if denom.item() <= 1e-12:
        return 0.0
    return float((torch.linalg.norm(xy, ord='fro').pow(2) / denom).clamp(0.0, 1.0).cpu().item())


def normalized_classification_utility(args, unit_score, full_score):
    chance = 1.0 / float(args.n_classes)
    denom = max(float(full_score) - chance, args.capacity_eps)
    utility = (float(unit_score) - chance) / denom
    return max(0.0, min(1.0, utility))


def linear_probe_ua(args, reps, labels):
    labels = np.asarray(labels).astype(int)
    if torch.is_tensor(reps):
        reps = reps.detach().cpu().numpy()
    reps = np.asarray(reps, dtype=np.float32)
    chance = 1.0 / float(args.n_classes)
    if reps.shape[0] < args.n_classes * 2 or len(np.unique(labels)) < 2:
        return chance

    _, counts = np.unique(labels, return_counts=True)
    stratify = labels if counts.min() >= 2 else None
    indices = np.arange(labels.shape[0])
    try:
        train_idx, eval_idx = train_test_split(
            indices,
            test_size=0.3,
            random_state=args.seed,
            stratify=stratify,
        )
        if len(np.unique(labels[train_idx])) < 2 or len(np.unique(labels[eval_idx])) < 2:
            return chance
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, class_weight='balanced', solver='lbfgs'),
        )
        probe.fit(reps[train_idx], labels[train_idx])
        preds = probe.predict(reps[eval_idx])
        return recall_score(
            labels[eval_idx],
            preds,
            labels=list(range(args.n_classes)),
            average='macro',
            zero_division=0,
        )
    except Exception:
        return chance


def collect_unimodal_calibration(args, model, dataloader, modality):
    if args.dataset not in ['IEMOCAPFOUR', 'IEMOCAPSIX']:
        raise ValueError('Automatic capacity calibration currently supports IEMOCAP classification datasets only.')

    modality_to_idx = {'a': 0, 't': 1, 'v': 2}
    old_condition = getattr(model, 'test_condition', args.test_condition)
    reps_modality, reps_full = [], []
    preds_modality, preds_full, labels = [], [], []

    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            if args.capacity_max_batches is not None and batch_idx >= args.capacity_max_batches:
                break

            input_m, mask_m, umask, label = build_masked_batch(args, data, modality, False)
            model.test_condition = modality
            _, _, _, _, vib_outputs_m, _ = model(input_m, mask_m, umask, False)
            rep_m = getattr(model, 'last_unimodal_reps')[modality]
            out_m = vib_outputs_m[modality_to_idx[modality]]

            input_full, mask_full, umask_full, label_full = build_masked_batch(args, data, 'atv', False)
            model.test_condition = 'atv'
            hidden_full, out_full, *_ = model(input_full, mask_full, umask_full, False)
            rep_full = getattr(model, 'last_pre_fusion_hidden', hidden_full)

            active = umask.reshape(-1).bool()
            reps_modality.append(rep_m.reshape(-1, rep_m.size(-1))[active].detach().cpu())
            reps_full.append(rep_full.reshape(-1, rep_full.size(-1))[active].detach().cpu())
            preds_modality.append(out_m.reshape(-1, out_m.size(-1))[active].argmax(dim=-1).detach().cpu().numpy())
            preds_full.append(out_full.reshape(-1, out_full.size(-1))[active].argmax(dim=-1).detach().cpu().numpy())
            labels.append(label.reshape(-1)[active].detach().cpu().numpy())

    model.test_condition = old_condition
    reps_modality = torch.cat(reps_modality, dim=0)
    reps_full = torch.cat(reps_full, dim=0)
    labels = np.concatenate(labels)
    preds_modality = np.concatenate(preds_modality)
    preds_full = np.concatenate(preds_full)

    cka = linear_cka(reps_modality.float(), reps_full.float())
    ua_modality = linear_probe_ua(args, reps_modality.float(), labels)
    ua_full = linear_probe_ua(args, reps_full.float(), labels)
    return cka, ua_modality, ua_full


def collect_condition_calibration(args, model, dataloader, condition):
    if args.dataset not in ['IEMOCAPFOUR', 'IEMOCAPSIX']:
        raise ValueError('Automatic capacity calibration currently supports IEMOCAP classification datasets only.')

    old_condition = getattr(model, 'test_condition', args.test_condition)
    reps_condition, reps_full = [], []
    preds_condition, preds_full, labels = [], [], []

    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            if args.capacity_max_batches is not None and batch_idx >= args.capacity_max_batches:
                break

            input_c, mask_c, umask, label = build_masked_batch(args, data, condition, False)
            model.test_condition = condition
            hidden_c, out_c, *_ = model(input_c, mask_c, umask, False)
            rep_c = getattr(model, 'last_pre_fusion_hidden', hidden_c)

            input_full, mask_full, umask_full, label_full = build_masked_batch(args, data, 'atv', False)
            model.test_condition = 'atv'
            hidden_full, out_full, *_ = model(input_full, mask_full, umask_full, False)
            rep_full = getattr(model, 'last_pre_fusion_hidden', hidden_full)

            active = umask.reshape(-1).bool()
            reps_condition.append(rep_c.reshape(-1, rep_c.size(-1))[active].detach().cpu())
            reps_full.append(rep_full.reshape(-1, rep_full.size(-1))[active].detach().cpu())
            preds_condition.append(out_c.reshape(-1, out_c.size(-1))[active].argmax(dim=-1).detach().cpu().numpy())
            preds_full.append(out_full.reshape(-1, out_full.size(-1))[active].argmax(dim=-1).detach().cpu().numpy())
            labels.append(label.reshape(-1)[active].detach().cpu().numpy())

    model.test_condition = old_condition
    reps_condition = torch.cat(reps_condition, dim=0)
    reps_full = torch.cat(reps_full, dim=0)
    labels = np.concatenate(labels)
    preds_condition = np.concatenate(preds_condition)
    preds_full = np.concatenate(preds_full)

    cka = linear_cka(reps_condition.float(), reps_full.float())
    ua_condition = linear_probe_ua(args, reps_condition.float(), labels)
    ua_full = linear_probe_ua(args, reps_full.float(), labels)
    return cka, ua_condition, ua_full


def calibrate_unimodal_capacities(args, model, dataloader):
    for modality in ['a', 't', 'v']:
        cka, ua_modality, ua_full = collect_unimodal_calibration(args, model, dataloader, modality)
        utility = (ua_modality + args.capacity_eps) / (ua_full + args.capacity_eps)
        capacity, reliability, base_reliability = map_capacity(
            args, cka, utility, args.uni_cap_min, args.uni_cap_max, args.capacity_gamma_uni)
        setattr(args, f'uni_cap_{modality}', capacity)
        print(
            f"capacity_calibration modality:{modality}; cka:{cka:.6f}; "
            f"utility:{min(1.0, utility):.6f}; capacity:{capacity:.6f}"
        )


def calibrate_rc_unimodal_capacities(args, model, dataloader, warm_stats):
    for modality in ['a', 't', 'v']:
        cka, ua_modality, ua_full = collect_unimodal_calibration(args, model, dataloader, modality)
        utility = normalized_classification_utility(args, ua_modality, ua_full)
        multiplier, reliability, base_reliability = map_capacity_multiplier(
            args, cka, utility, args.rc_alpha_min_uni, args.rc_alpha_max_uni, args.capacity_gamma_uni)
        warm_rate = rc_modality_rate_from_stats(warm_stats, modality) or args.rc_eps
        capacity = max(args.rc_eps, multiplier * warm_rate)
        setattr(args, f'uni_cap_{modality}', capacity)
        print(f"rc_capacity_calibration modality:{modality}; alpha:{multiplier:.6f}; capacity:{capacity:.6f}")


def calibrate_fusion_capacities(args, model, dataloader):
    for condition in TEST_CONDITIONS:
        cka, ua_condition, ua_full = collect_condition_calibration(args, model, dataloader, condition)
        utility = (ua_condition + args.capacity_eps) / (ua_full + args.capacity_eps)
        capacity, reliability, base_reliability = map_capacity(
            args, cka, utility, args.fusion_cap_min, args.fusion_cap_max, args.capacity_gamma_fusion)
        setattr(args, f'fusion_cap_{condition}', capacity)
        print(
            f"capacity_calibration condition:{condition}; cka:{cka:.6f}; "
            f"utility:{min(1.0, utility):.6f}; capacity:{capacity:.6f}"
        )


def calibrate_rc_fusion_capacities(args, model, dataloader, warm_stats_by_condition):
    for condition in TEST_CONDITIONS:
        cka, ua_condition, ua_full = collect_condition_calibration(args, model, dataloader, condition)
        utility = normalized_classification_utility(args, ua_condition, ua_full)
        multiplier, reliability, base_reliability = map_capacity_multiplier(
            args, cka, utility, args.rc_alpha_min_fusion, args.rc_alpha_max_fusion, args.capacity_gamma_fusion)
        warm_stats = warm_stats_by_condition.get(condition, {})
        warm_rate = float(warm_stats.get('fusion_kl', args.rc_eps))
        capacity = max(args.rc_eps, multiplier * warm_rate)
        setattr(args, f'fusion_cap_{condition}', capacity)
        print(f"rc_capacity_calibration condition:{condition}; alpha:{multiplier:.6f}; capacity:{capacity:.6f}")


def compute_class_weights(dataloader, n_classes, device):
    counts = torch.zeros(n_classes, dtype=torch.float)
    for data in dataloader:
        umask = data[7].reshape(-1).bool()
        labels = data[8].reshape(-1)[umask].long()
        counts += torch.bincount(labels.cpu(), minlength=n_classes).float()
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / (n_classes * counts)
    return weights.to(device)
