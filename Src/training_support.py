import numpy as np
import torch
import torch.nn.functional as F
from data_pipeline import generate_mask, generate_inputs

TEST_CONDITIONS = ['a', 't', 'v', 'at', 'av', 'tv', 'atv']


def format_aib_stats(stats, prefix):
    if not stats:
        return 'none'
    prefixes = list(prefix) if set(prefix).issubset({'a', 't', 'v'}) and len(prefix) > 1 else [prefix]
    keys = []
    for stat_prefix in prefixes:
        keys.extend([
            f'{stat_prefix}_kl',
            f'{stat_prefix}_post_var',
            f'{stat_prefix}_mu_abs',
            f'{stat_prefix}_common_kl',
            f'{stat_prefix}_common_post_var',
            f'{stat_prefix}_common_mu_abs',
        ])
    keys.extend(['fusion_kl', 'fusion_post_var', 'fusion_mu_abs'])
    for stat_prefix in prefixes:
        keys.append(f'scrb_{stat_prefix}_kl')
    for stat_prefix in ['a', 't', 'v']:
        keys.append(f'DDIB_u_{stat_prefix}')
    keys.append('DDIB_fusion')
    return ' '.join(f'--{key} {stats[key]:.6f}' for key in keys if key in stats)


def get_condition_arg(args, prefix, condition):
    return getattr(args, f'{prefix}_{condition}', None)


def ib_penalty(rate, beta, cap=None, kappa=0.0):
    penalty = rate * beta
    if cap is not None and kappa > 0:
        cap_tensor = rate.new_tensor(cap)
        penalty = penalty + 0.5 * kappa * torch.relu(rate - cap_tensor).pow(2)
    return penalty


def normalized_rate_penalty(rate, cap, multiplier, kappa=0.0, eps=1e-6):
    if cap is None:
        return rate.new_tensor(0.0)
    cap_tensor = rate.new_tensor(max(float(cap), eps))
    violation = torch.relu(rate / (cap_tensor + eps) - 1.0)
    return multiplier * violation + 0.5 * kappa * violation.pow(2)


def masked_token_kl(student_logits, teacher_logits, umask, temperature=1.0, eps=1e-8):
    temperature = max(float(temperature), eps)
    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
    token_kl = F.kl_div(student_log_prob, teacher_prob, reduction='none').sum(dim=-1) * (temperature ** 2)
    return (token_kl * umask).sum() / (umask.sum() + eps)


def masked_token_jsd(reference_logits, compressed_logits, umask, eps=1e-8):
    reference_prob = F.softmax(reference_logits.detach(), dim=-1)
    compressed_prob = F.softmax(compressed_logits, dim=-1)
    mixture = 0.5 * (reference_prob + compressed_prob)
    ref_kl = (reference_prob * ((reference_prob + eps).log() - (mixture + eps).log())).sum(dim=-1)
    comp_kl = (compressed_prob * ((compressed_prob + eps).log() - (mixture + eps).log())).sum(dim=-1)
    token_jsd = 0.5 * (ref_kl + comp_kl)
    return (token_jsd * umask).sum() / (umask.sum() + eps)


def masked_hidden_mse(student_hidden, teacher_hidden, umask, eps=1e-8):
    token_mse = (student_hidden - teacher_hidden).pow(2).mean(dim=-1)
    return (token_mse * umask).sum() / (umask.sum() + eps)


def masked_hidden_margin_mse(student_hidden, teacher_hidden, umask, margin=0.0, eps=1e-8):
    token_mse = (student_hidden - teacher_hidden).pow(2).mean(dim=-1)
    token_margin = torch.relu(token_mse - float(margin))
    return (token_margin * umask).sum() / (umask.sum() + eps)


def masked_cosine_similarity(student_hidden, teacher_hidden, umask, eps=1e-8):
    valid = umask.reshape(-1).bool()
    student = student_hidden.reshape(-1, student_hidden.size(-1))[valid]
    teacher = teacher_hidden.reshape(-1, teacher_hidden.size(-1))[valid]
    if student.numel() == 0:
        return student_hidden.new_tensor(0.0)
    return F.cosine_similarity(student, teacher, dim=-1, eps=eps).mean()


def second_stage_ib_scale(args, first_stage):
    if first_stage:
        return 1.0
    warmup = getattr(args, 'second_stage_ib_warmup', 0.0)
    if warmup <= 0:
        return 1.0
    current_epoch = getattr(args, 'current_epoch', args.stage_epoch)
    elapsed = max(0.0, float(current_epoch) - float(args.stage_epoch) + 1.0)
    return min(1.0, elapsed / warmup)


def first_stage_ib_scale(args, first_stage):
    if not first_stage:
        return 1.0
    warmup = float(getattr(args, 'first_stage_ib_warmup', 0.0))
    if warmup <= 0:
        return 1.0
    current_epoch = max(0.0, float(getattr(args, 'current_epoch', 0.0)))
    return min(1.0, (current_epoch + 1.0) / warmup)


def DDIB_fusion_alpha_scale(args, first_stage):
    if first_stage or not getattr(args, 'DDIB_fusion', False):
        return 0.0 if getattr(args, 'DDIB_fusion', False) else 1.0
    elapsed = max(0.0, float(getattr(args, 'current_epoch', args.stage_epoch)) - float(args.stage_epoch) + 1.0)
    start_epoch = max(0.0, float(getattr(args, 'DDIB_fusion_start_epoch', 0.0)))
    if elapsed <= start_epoch:
        return 0.0
    warmup = max(0.0, float(getattr(args, 'DDIB_fusion_alpha_warmup', 0.0)))
    if warmup <= 0:
        return 1.0
    return min(1.0, (elapsed - start_epoch) / warmup)


def update_dual_beta(beta, rate_ema, rate, cap, lr, ema_tau):
    if cap is None:
        return beta, rate_ema
    next_ema = rate if rate_ema is None else ema_tau * rate_ema + (1.0 - ema_tau) * rate
    next_beta = max(0.0, beta + lr * (next_ema - cap))
    return next_beta, next_ema


def update_bounded_dual_lambda(lam, rate_ema, rate, cap, lr, ema_tau, lambda_max, eps=1e-6):
    if cap is None or rate is None:
        return lam, rate_ema
    next_ema = rate if rate_ema is None else ema_tau * rate_ema + (1.0 - ema_tau) * rate
    violation = next_ema / (float(cap) + eps) - 1.0
    next_lam = min(float(lambda_max), max(0.0, lam + lr * violation))
    return next_lam, next_ema


def condition_modalities(condition):
    if condition == 'all':
        return ['a', 't', 'v']
    return [modality for modality in ['a', 't', 'v'] if modality in condition]


def parse_condition_subset(value):
    if value is None:
        return None
    conditions = []
    for condition in str(value).replace(';', ',').split(','):
        condition = condition.strip()
        if not condition:
            continue
        if condition not in TEST_CONDITIONS:
            raise ValueError(f'Unknown condition: {condition}')
        if condition not in conditions:
            conditions.append(condition)
    return conditions if conditions else None


def DDIB_uni_condition_scale(args, condition):
    selected = getattr(args, 'DDIB_uni_condition_set', None)
    if selected is not None and condition not in selected:
        return 0.0
    if getattr(args, 'DDIB_uni_single_only', False) and len(condition_modalities(condition)) != 1:
        return 0.0
    return float(getattr(args, 'DDIB_uni_condition_weight', 1.0))


def generic_condition_probabilities(args):
    modality_count = 3
    uniform = np.ones(len(TEST_CONDITIONS), dtype=np.float64) / len(TEST_CONDITIONS)
    beta = float(getattr(args, 'schedule_cardinality_beta', 1.0))
    omega = float(getattr(args, 'schedule_cardinality_omega', 0.5))
    omega = min(1.0, max(0.0, omega))
    card_scores = np.array([
        np.exp(beta * len(condition_modalities(condition)) / modality_count)
        for condition in TEST_CONDITIONS
    ], dtype=np.float64)
    card_probs = card_scores / card_scores.sum()
    probs = (1.0 - omega) * uniform + omega * card_probs
    return probs / probs.sum()


def expand_condition_schedule_from_probs(probs, length=100):
    counts = np.maximum(1, np.rint(probs * int(length)).astype(int))
    # Allocate the remaining slots after rounding.
    while counts.sum() > length and counts.max() > 1:
        counts[counts.argmax()] -= 1
    while counts.sum() < length:
        counts[probs.argmax()] += 1
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum()
    used = np.zeros(len(TEST_CONDITIONS), dtype=np.int64)
    schedule = []
    for step in range(int(length)):
        deficit = (step + 1) * probs - used
        deficit = np.where(used < counts, deficit, -np.inf)
        idx = int(deficit.argmax())
        schedule.append(TEST_CONDITIONS[idx])
        used[idx] += 1
    return schedule


def map_reliability(args, similarity, utility):
    similarity = max(0.0, min(1.0, similarity))
    utility = max(0.0, min(1.0, utility))
    base_reliability = (similarity ** args.capacity_eta) * (utility ** (1.0 - args.capacity_eta))

    if args.capacity_source == 'cka-only':
        reliability = similarity
    elif args.capacity_source == 'ua-only':
        reliability = utility
    elif args.capacity_source == 'equal':
        reliability = 0.5
    elif args.capacity_source == 'random':
        reliability = float(np.random.uniform(0.0, 1.0))
    elif args.capacity_source == 'reversed':
        reliability = 1.0 - base_reliability
    else:
        reliability = base_reliability

    return reliability, base_reliability


def map_capacity(args, similarity, utility, cap_min, cap_max, gamma):
    reliability, base_reliability = map_reliability(args, similarity, utility)
    capacity = cap_min + (cap_max - cap_min) * (reliability ** gamma)
    return capacity, reliability, base_reliability


def map_capacity_multiplier(args, similarity, utility, alpha_min, alpha_max, gamma):
    reliability, base_reliability = map_reliability(args, similarity, utility)
    multiplier = alpha_min + (alpha_max - alpha_min) * (reliability ** gamma)
    return multiplier, reliability, base_reliability


def rc_modality_rate_from_stats(stats, modality):
    values = []
    for key in [f'scrb_{modality}_kl', f'{modality}_kl', f'{modality}_common_kl']:
        if key in stats:
            values.append(float(stats[key]))
    if not values:
        return None
    return max(values)


def build_masked_batch(args, data, test_condition, first_stage):
    audio_host, text_host, visual_host = data[0], data[1], data[2]
    audio_guest, text_guest, visual_guest = data[3], data[4], data[5]
    qmask, umask, label = data[6], data[7], data[8]
    seqlen = audio_host.size(0)
    batch = audio_host.size(1)

    matrix = generate_mask(
        seqlen,
        batch,
        test_condition,
        first_stage,
        getattr(args, 'mask_rate', None),
        getattr(args, 'mask_first_stage', False),
    )
    audio_host_mask = torch.LongTensor(np.reshape(matrix[0], (batch, seqlen, 1)).transpose(1, 0, 2))
    text_host_mask = torch.LongTensor(np.reshape(matrix[1], (batch, seqlen, 1)).transpose(1, 0, 2))
    visual_host_mask = torch.LongTensor(np.reshape(matrix[2], (batch, seqlen, 1)).transpose(1, 0, 2))

    matrix = generate_mask(
        seqlen,
        batch,
        test_condition,
        first_stage,
        getattr(args, 'mask_rate', None),
        getattr(args, 'mask_first_stage', False),
    )
    audio_guest_mask = torch.LongTensor(np.reshape(matrix[0], (batch, seqlen, 1)).transpose(1, 0, 2))
    text_guest_mask = torch.LongTensor(np.reshape(matrix[1], (batch, seqlen, 1)).transpose(1, 0, 2))
    visual_guest_mask = torch.LongTensor(np.reshape(matrix[2], (batch, seqlen, 1)).transpose(1, 0, 2))

    masked_audio_host = audio_host * audio_host_mask
    masked_audio_guest = audio_guest * audio_guest_mask
    masked_text_host = text_host * text_host_mask
    masked_text_guest = text_guest * text_guest_mask
    masked_visual_host = visual_host * visual_host_mask
    masked_visual_guest = visual_guest * visual_guest_mask

    cuda = torch.cuda.is_available() and not args.no_cuda
    if cuda:
        device = args.device
        masked_audio_host, audio_host_mask = masked_audio_host.to(device), audio_host_mask.to(device)
        masked_text_host, text_host_mask = masked_text_host.to(device), text_host_mask.to(device)
        masked_visual_host, visual_host_mask = masked_visual_host.to(device), visual_host_mask.to(device)
        masked_audio_guest, audio_guest_mask = masked_audio_guest.to(device), audio_guest_mask.to(device)
        masked_text_guest, text_guest_mask = masked_text_guest.to(device), text_guest_mask.to(device)
        masked_visual_guest, visual_guest_mask = masked_visual_guest.to(device), visual_guest_mask.to(device)
        qmask = qmask.to(device)
        umask = umask.to(device)
        label = label.to(device)

    masked_input_features = generate_inputs(masked_audio_host, masked_text_host, masked_visual_host,
                                            masked_audio_guest, masked_text_guest, masked_visual_guest, qmask)
    input_features_mask = generate_inputs(audio_host_mask, text_host_mask, visual_host_mask,
                                          audio_guest_mask, text_guest_mask, visual_guest_mask, qmask)
    return masked_input_features[0], input_features_mask[0], umask, label
