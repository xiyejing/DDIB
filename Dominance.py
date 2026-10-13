import torch
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).with_name('Src')
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training_support import (
    TEST_CONDITIONS, build_masked_batch, DDIB_uni_condition_scale, masked_cosine_similarity,
)


def compute_DDIB_dominance_weights(args, model, data, active_condition, train, first_stage):
    enabled = (
        train
        and (not first_stage)
        and (getattr(args, 'DDIB_uni', False) or getattr(args, 'DDIB_fusion', False))
    )
    if not enabled:
        return None

    saved_condition = getattr(model, 'test_condition', active_condition)
    was_training = model.training
    sentinel = object()
    saved_attrs = {}
    for attr_name in [
        'last_vib_stats',
        'last_unimodal_reps',
        'last_DDIB_aux_logits',
        'last_DDIB_z_aux_logits',
        'last_second_stage_uni_logits',
        'last_pre_fusion_hidden',
        'last_scrb_aux_hidden',
        'last_scrb_aux_out',
        'last_scrb_rates',
        'last_fusion_kl',
    ]:
        saved_attrs[attr_name] = getattr(model, attr_name, sentinel)
    saved_vib_deterministic_eval = []
    for module in model.modules():
        if hasattr(module, 'deterministic_eval'):
            saved_vib_deterministic_eval.append((module, module.deterministic_eval))
            module.deterministic_eval = True

    weights = {}
    model.eval()
    with torch.no_grad():
        input_full, mask_full, umask_full, _ = build_masked_batch(args, data, 'atv', False)
        model.test_condition = 'atv'
        hidden_full, *_ = model(input_full, mask_full, umask_full, False)
        hidden_full = getattr(model, 'last_pre_fusion_hidden', hidden_full).detach()

        uni_condition_scale = DDIB_uni_condition_scale(args, active_condition)
        if getattr(args, 'DDIB_uni', False) and uni_condition_scale > 0.0:
            uni_scores = []
            for modality in ['a', 't', 'v']:
                input_m, mask_m, umask_m, _ = build_masked_batch(args, data, modality, False)
                model.test_condition = modality
                hidden_m, *_ = model(input_m, mask_m, umask_m, False)
                hidden_m = getattr(model, 'last_pre_fusion_hidden', hidden_m).detach()
                uni_scores.append(masked_cosine_similarity(hidden_m, hidden_full, umask_m))
            uni_scores = torch.stack(uni_scores)
            tau_raw = float(getattr(args, 'DDIB_tau_uni', 0.5))
            tau = max(abs(tau_raw), 1e-6)
            if getattr(args, 'DDIB_uni_weight_direction', 'dominant') == 'weak':
                uni_scores_for_weight = -uni_scores
            else:
                uni_scores_for_weight = uni_scores
            if tau_raw < 0.0:
                uni_scores_for_weight = -uni_scores_for_weight
            uni_weights = len(uni_scores) * torch.softmax(uni_scores_for_weight / tau, dim=0)
            weights['uni'] = {
                modality: uni_weights[idx].detach()
                for idx, modality in enumerate(['a', 't', 'v'])
            }

        if getattr(args, 'DDIB_fusion', False):
            if getattr(args, 'DDIB_fusion_all_conditions', False):
                fusion_scores = []
                for condition in TEST_CONDITIONS:
                    if condition == 'atv':
                        hidden_c = hidden_full
                        umask_c = umask_full
                    else:
                        input_c, mask_c, umask_c, _ = build_masked_batch(args, data, condition, False)
                        model.test_condition = condition
                        hidden_c, *_ = model(input_c, mask_c, umask_c, False)
                        hidden_c = getattr(model, 'last_pre_fusion_hidden', hidden_c).detach()
                    fusion_scores.append(masked_cosine_similarity(hidden_c, hidden_full, umask_c))
                fusion_scores = torch.stack(fusion_scores)
                tau = max(float(getattr(args, 'DDIB_tau_fusion', 0.5)), 1e-6)
                fusion_weights = len(fusion_scores) * torch.softmax(fusion_scores / tau, dim=0)
                condition_idx = TEST_CONDITIONS.index(active_condition)
                weights['fusion'] = fusion_weights[condition_idx].detach()
            elif active_condition == 'atv':
                weights['fusion'] = hidden_full.new_tensor(1.0)
            else:
                input_c, mask_c, umask_c, _ = build_masked_batch(args, data, active_condition, False)
                model.test_condition = active_condition
                hidden_c, *_ = model(input_c, mask_c, umask_c, False)
                hidden_c = getattr(model, 'last_pre_fusion_hidden', hidden_c).detach()
                sim = masked_cosine_similarity(hidden_c, hidden_full, umask_c).detach()
                weights['fusion'] = torch.exp(sim / max(float(getattr(args, 'DDIB_tau_fusion', 0.5)), 1e-6)).clamp(
                    max=float(getattr(args, 'DDIB_fusion_weight_cap', 3.0))
                )

    model.test_condition = saved_condition
    if was_training:
        model.train()
    else:
        model.eval()
    for attr_name, attr_value in saved_attrs.items():
        if attr_value is sentinel:
            if hasattr(model, attr_name):
                delattr(model, attr_name)
        else:
            setattr(model, attr_name, attr_value)
    for module, deterministic_eval in saved_vib_deterministic_eval:
        module.deterministic_eval = deterministic_eval
    return weights
