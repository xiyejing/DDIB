CHECKPOINT_MODES = [
    'final',
    'best_acc',
    'best_ua',
    'best_f1',
    'best_acc_ua',
    'best_hmean_acc_ua',
    'best_acc_ua_f1',
]

CHECKPOINT_SELECTION = 'best_acc_ua'
SELECTION_START = 'all'
CHECKPOINT_EVAL_VARIANTS = 'both'
CHECKPOINT_TIE_BREAK = 'acc_ua'


def metric_score(acc, ua, f1, mode):
    if mode == 'best_acc':
        return float(acc)
    if mode == 'best_ua':
        return float(ua)
    if mode == 'best_f1':
        return float(f1)
    if mode == 'best_acc_ua':
        return 0.5 * (float(acc) + float(ua))
    if mode == 'best_hmean_acc_ua':
        denom = max(float(acc) + float(ua), 1e-12)
        return 2.0 * float(acc) * float(ua) / denom
    if mode == 'best_acc_ua_f1':
        return (float(acc) + float(ua) + float(f1)) / 3.0
    if mode == 'final':
        return 0.0
    raise ValueError(f'Unknown checkpoint selection mode: {mode}')


def result_score(result, mode):
    _, ua, acc, f1 = result[:4]
    return metric_score(acc, ua, f1, mode)


def tie_score(acc, ua, f1, tie_break):
    if tie_break == 'acc':
        return float(acc)
    if tie_break == 'ua':
        return float(ua)
    if tie_break == 'f1':
        return float(f1)
    if tie_break == 'acc_ua':
        return 0.5 * (float(acc) + float(ua))
    if tie_break == 'acc_ua_f1':
        return (float(acc) + float(ua) + float(f1)) / 3.0
    return 0.0


def choose_result_variant(variant_results, mode, tie_break='acc_ua'):
    if len(variant_results) == 1:
        return variant_results[0]
    effective_mode = mode if mode != 'final' else 'best_acc_ua'
    best = None
    best_key = None
    for variant, result in variant_results:
        _, ua, acc, f1 = result[:4]
        key = (
            result_score(result, effective_mode),
            tie_score(acc, ua, f1, tie_break),
            float(acc),
            float(ua),
            float(f1),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = (variant, result)
    return best


def resolve_selection_start(total_candidates, stage_epoch=0, epochs=0, mask_rate=None):
    if total_candidates <= 0:
        return 0
    raw = SELECTION_START
    if raw == 'all':
        start = 0
    elif raw == 'stage2':
        start = int(max(0, stage_epoch))
    elif raw == 'auto':
        if mask_rate is not None and stage_epoch < epochs:
            start = int(max(0, stage_epoch))
        else:
            start = 0
    else:
        start = int(raw)
    return min(max(start, 0), total_candidates - 1)


def choose_best_index(accs, uas, fscores, mode, start=0, tie_break='acc_ua'):
    if len(accs) == 0:
        raise ValueError('No checkpoint candidates available.')
    if mode == 'final':
        return len(accs) - 1
    best_idx = start
    best_key = None
    for idx in range(start, len(accs)):
        key = (
            metric_score(accs[idx], uas[idx], fscores[idx], mode),
            tie_score(accs[idx], uas[idx], fscores[idx], tie_break),
            float(accs[idx]),
            float(uas[idx]),
            float(fscores[idx]),
            idx,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_idx = idx
    return int(best_idx)
