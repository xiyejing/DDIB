import argparse

from training_support import TEST_CONDITIONS, parse_condition_subset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--audio-feature', type=str, default=None)
    parser.add_argument('--text-feature', type=str, default=None)
    parser.add_argument('--video-feature', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='CMUMOSI', choices=['CMUMOSI', 'CMUMOSEI', 'IEMOCAPFOUR', 'IEMOCAPSIX'])

    parser.add_argument('--time-attn', action='store_true', default=False)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--drop_rate', type=float, default=0.5)
    parser.add_argument('--attn_drop_rate', type=float, default=0.0)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--n_speakers', type=int, default=2)

    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['Adam', 'AdamW'])
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--l2', type=float, default=0.00001)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num-folder', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint-selection', type=str, default='auto', choices=[
        'auto', 'final', 'best_acc', 'best_ua', 'best_f1', 'best_acc_ua',
        'best_hmean_acc_ua', 'best_acc_ua_f1',
    ])
    parser.add_argument('--test_condition', type=str, default='all', choices=['all'])
    parser.add_argument('--mask-rate', type=float, default=None)
    parser.add_argument('--mask-first-stage', action='store_true', default=False)
    parser.add_argument('--stage_epoch', type=float, default=100)
    parser.add_argument('--ablation', type=str, default='none', choices=['none', 'wo-msd', 'wo-mcd', 'wo-imc', 'wo-imd'])

    parser.add_argument('--ib-beta', type=float, default=0.1)
    parser.add_argument('--ib-std-offset', type=float, default=-5.0)
    parser.add_argument('--ib-std-offset-a', type=float, default=None)
    parser.add_argument('--ib-std-offset-t', type=float, default=None)
    parser.add_argument('--ib-std-offset-v', type=float, default=None)
    parser.add_argument('--ib-std-offset-common', type=float, default=None)
    parser.add_argument('--ib-std-offset-fusion', type=float, default=None)
    parser.add_argument('--adaptive-multi-ib', action='store_true', default=False)
    parser.add_argument('--multi-ib-std-offset-a', type=float, default=None)
    parser.add_argument('--multi-ib-std-offset-t', type=float, default=None)
    parser.add_argument('--multi-ib-std-offset-v', type=float, default=None)
    parser.add_argument('--multi-ib-std-offset-common', type=float, default=None)
    parser.add_argument('--multi-ib-std-offset-fusion', type=float, default=None)
    parser.add_argument('--first-stage-ib-warmup', type=float, default=0.0)
    parser.add_argument('--stage1-joint-weight', type=float, default=0.0)
    parser.add_argument(
        '--restore-best-stage1-attention', action='store_true', default=False,
        help='Restore each modality attention block from its test-best stage-1 epoch.',
    )
    parser.add_argument(
        '--restore-best-stage1-branches', action='store_true', default=False,
        help='Restore each modality-specific input projection, VIB, attention and classifier from its own test-best stage-1 epoch.',
    )
    parser.add_argument('--normalize-pair-fusion', action='store_true', default=False)
    parser.add_argument('--ib-beta-a', type=float, default=None)
    parser.add_argument('--ib-beta-t', type=float, default=None)
    parser.add_argument('--ib-beta-v', type=float, default=None)
    parser.add_argument('--fusion-aib', action='store_true', default=False)
    parser.add_argument('--fusion-ib-beta', type=float, default=0.01)
    parser.add_argument('--second-stage-uni-aib', action='store_true', default=False)
    parser.add_argument('--modality-prior', action='store_true', default=False)
    parser.add_argument('--condition-prior', action='store_true', default=False)
    parser.add_argument('--prior-reg-lambda', type=float, default=0.0)
    parser.add_argument('--deterministic-eval-aib', action='store_true', default=False)
    parser.add_argument('--deterministic-train-aib', action='store_true', default=False)
    parser.add_argument('--mc-eval-samples', type=int, default=1)
    parser.add_argument('--uni-aib-residual-alpha', type=float, default=1.0)
    parser.add_argument('--second-stage-ib-skip-full', action='store_true', default=False)
    parser.add_argument('--second-stage-ib-warmup', type=float, default=0.0)
    parser.add_argument('--condition-size-ib-scaling', action='store_true', default=False)
    parser.add_argument('--condition-size-ib-zeta', type=float, default=0.5)
    parser.add_argument('--second-stage-uni-decoder', action='store_true', default=False)
    parser.add_argument('--second-stage-uni-decoder-weight', type=float, default=0.2)
    parser.add_argument('--second-stage-uni-decoder-single-only', action='store_true', default=False)
    parser.add_argument('--condition-distill', action='store_true', default=False)
    parser.add_argument('--condition-distill-weight', type=float, default=0.0)
    parser.add_argument('--condition-distill-hidden-weight', type=float, default=0.0)
    parser.add_argument('--condition-distill-temperature', type=float, default=2.0)
    parser.add_argument('--condition-distill-single-only', action='store_true', default=False)
    parser.add_argument('--condition-distill-eps', type=float, default=1e-8)
    parser.add_argument('--single-condition-heads', action='store_true', default=False)
    parser.add_argument('--condition-denoise-gates', action='store_true', default=False)
    parser.add_argument('--condition-gate-reg-lambda', type=float, default=0.0)
    parser.add_argument('--condition-adapter', action='store_true', default=False)
    parser.add_argument('--condition-adapter-rank', type=int, default=4)
    parser.add_argument('--condition-adapter-reg-lambda', type=float, default=0.0)
    parser.add_argument('--condition-generated-adapter', action='store_true', default=False)
    parser.add_argument('--adapter-bank-size', type=int, default=4)
    parser.add_argument('--adapter-bank-rank', type=int, default=8)
    parser.add_argument('--condition-embedding-dim', type=int, default=32)
    parser.add_argument('--adapter-eta-init', type=float, default=0.05)
    parser.add_argument('--adapter-bank-reg-lambda', type=float, default=0.0)
    parser.add_argument('--condition-align', action='store_true', default=False)
    parser.add_argument('--condition-align-weight', type=float, default=0.0)
    parser.add_argument('--condition-align-delta0', type=float, default=0.05)
    parser.add_argument('--condition-semantic-kl', action='store_true', default=False)
    parser.add_argument('--condition-semantic-kl-weight', type=float, default=0.0)
    parser.add_argument('--scrb-aux-aib', action='store_true', default=False)
    parser.add_argument('--scrb-embedding-dim', type=int, default=32)
    parser.add_argument('--scrb-aux-weight', type=float, default=0.2)
    parser.add_argument('--scrb-consistency-weight', type=float, default=0.0)
    parser.add_argument('--scrb-rate-weight', type=float, default=0.0)
    parser.add_argument('--scrb-rate-kappa', type=float, default=0.1)
    parser.add_argument('--scrb-sufficiency-gate', action='store_true', default=False)
    parser.add_argument('--scrb-sufficiency-tau', type=float, default=0.1)
    parser.add_argument('--balanced-ce', action='store_true', default=False)
    parser.add_argument('--label-smoothing', type=float, default=0.05)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--lr-scheduler', type=str, default='cosine', choices=['none', 'cosine'])
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('--ema-eval', action='store_true', default=True)
    parser.add_argument('--no-ema-eval', action='store_false', dest='ema_eval')
    parser.add_argument('--ema-decay', type=float, default=0.999)
    parser.add_argument('--reset-optimizer-stage2', action='store_true', default=True)
    parser.add_argument('--keep-optimizer-stage2', action='store_false', dest='reset_optimizer_stage2')

    parser.add_argument('--uni-dual-aib', action='store_true', default=False)
    parser.add_argument('--fusion-dual-aib', action='store_true', default=False)
    parser.add_argument('--dual-lr-uni', type=float, default=0.01)
    parser.add_argument('--dual-lr-fusion', type=float, default=0.01)
    parser.add_argument('--dual-ema', type=float, default=0.9)
    parser.add_argument('--dual-update-stage', type=str, default='both', choices=['first', 'second', 'both'])
    parser.add_argument('--aug-kappa-uni', type=float, default=0.0)
    parser.add_argument('--aug-kappa-fusion', type=float, default=0.0)
    parser.add_argument('--uni-cap-a', type=float, default=None)
    parser.add_argument('--uni-cap-t', type=float, default=None)
    parser.add_argument('--uni-cap-v', type=float, default=None)
    parser.add_argument('--auto-uni-capacity', action='store_true', default=False)
    parser.add_argument('--auto-fusion-capacity', action='store_true', default=False)
    parser.add_argument('--capacity-source', type=str, default='cka-ua', choices=['cka-ua', 'cka-only', 'ua-only', 'equal', 'random', 'reversed'])
    parser.add_argument('--uni-cap-min', type=float, default=1.0)
    parser.add_argument('--uni-cap-max', type=float, default=5.0)
    parser.add_argument('--fusion-cap-min', type=float, default=1.0)
    parser.add_argument('--fusion-cap-max', type=float, default=5.0)
    parser.add_argument('--capacity-eta', type=float, default=0.7)
    parser.add_argument('--capacity-gamma-uni', type=float, default=1.0)
    parser.add_argument('--capacity-gamma-fusion', type=float, default=1.0)
    parser.add_argument('--capacity-eps', type=float, default=1e-6)
    parser.add_argument('--capacity-max-batches', type=int, default=None)
    parser.add_argument('--rc-aib', action='store_true', default=False)
    parser.add_argument('--rc-uni-aib', action='store_true', default=False)
    parser.add_argument('--rc-fusion-aib', action='store_true', default=False)
    parser.add_argument('--rc-alpha-min-uni', type=float, default=0.85)
    parser.add_argument('--rc-alpha-max-uni', type=float, default=1.10)
    parser.add_argument('--rc-alpha-min-fusion', type=float, default=0.95)
    parser.add_argument('--rc-alpha-max-fusion', type=float, default=1.20)
    parser.add_argument('--rc-dual-lr-uni', type=float, default=0.05)
    parser.add_argument('--rc-dual-lr-fusion', type=float, default=0.05)
    parser.add_argument('--rc-dual-ema', type=float, default=0.95)
    parser.add_argument('--rc-lambda-init', type=float, default=0.1)
    parser.add_argument('--rc-lambda-max', type=float, default=5.0)
    parser.add_argument('--rc-kappa-uni', type=float, default=0.1)
    parser.add_argument('--rc-kappa-fusion', type=float, default=0.1)
    parser.add_argument('--rc-eps', type=float, default=1e-6)
    for condition in TEST_CONDITIONS:
        parser.add_argument(f'--fusion-cap-{condition}', type=float, default=None)

    parser.add_argument('--all-train-conditions', type=str, default=None)
    parser.add_argument('--generic-condition-schedule', action='store_true', default=False)
    parser.add_argument('--schedule-cardinality-omega', type=float, default=0.5)
    parser.add_argument('--schedule-cardinality-beta', type=float, default=1.0)
    parser.add_argument('--train-loss-condition-scheduler', action='store_true', default=False)
    parser.add_argument('--train-loss-scheduler-alpha', type=float, default=0.2)
    parser.add_argument('--train-loss-scheduler-tau', type=float, default=0.5)
    parser.add_argument('--train-loss-scheduler-ema', type=float, default=0.9)
    parser.add_argument('--DDIB-uni', action='store_true', default=False)
    parser.add_argument('--DDIB-fusion', action='store_true', default=False)
    parser.add_argument('--DDIB-uni-beta', type=float, default=0.1)
    parser.add_argument('--DDIB-fusion-beta', type=float, default=0.01)
    parser.add_argument('--DDIB-aux-weight', type=float, default=5.0)
    parser.add_argument('--DDIB-aux-head-mode', type=str, default='both', choices=['both', 'specific', 'common'])
    parser.add_argument('--DDIB-z-aux-weight', type=float, default=0.0)
    parser.add_argument('--DDIB-z-aux-head-mode', type=str, default='specific', choices=['both', 'specific', 'common'])
    parser.add_argument('--DDIB-uni-conditions', type=str, default=None)
    parser.add_argument('--DDIB-uni-single-only', action='store_true', default=False)
    parser.add_argument('--DDIB-uni-condition-weight', type=float, default=1.0)
    parser.add_argument('--DDIB-uni-weight-direction', type=str, default='dominant', choices=['dominant', 'weak'])
    parser.add_argument('--DDIB-tau-uni', type=float, default=1.0)
    parser.add_argument('--DDIB-tau-fusion', type=float, default=2.0)
    parser.add_argument('--DDIB-fusion-residual-alpha', type=float, default=1.0)
    parser.add_argument('--DDIB-fusion-residual-mode', type=str, default='blend', choices=[
        'blend', 'add', 'projected_add', 'gated_projected_add', 'gated_lowrank_add',
        'delta_projected_add', 'gated_delta_projected_add', 'gated_delta_lowrank_add',
    ])
    parser.add_argument('--DDIB-fusion-residual-rank', type=int, default=48)
    parser.add_argument('--DDIB-fusion-gate-init', type=float, default=0.05)
    parser.add_argument('--DDIB-fusion-gate-reg-lambda', type=float, default=0.0)
    parser.add_argument('--DDIB-fusion-alpha-warmup', type=float, default=0.0)
    parser.add_argument('--DDIB-fusion-start-epoch', type=float, default=0.0)
    parser.add_argument('--DDIB-fusion-deterministic-train', action='store_true', default=False)
    parser.add_argument('--DDIB-fusion-deterministic-eval', action='store_true', default=False)
    parser.add_argument('--DDIB-fusion-anchor-weight', type=float, default=0.0)
    parser.add_argument('--DDIB-fusion-anchor-temperature', type=float, default=2.0)
    parser.add_argument('--DDIB-fusion-all-conditions', action='store_true', default=False)
    parser.add_argument('--DDIB-fusion-weight-cap', type=float, default=3.0)
    parser.add_argument('--DDIB-deterministic-eval', action='store_true', default=False)
    parser.add_argument('--start-folder', type=int, default=0)
    parser.add_argument('--end-folder', type=int, default=None)
    parser.add_argument('--verbose', action='store_true', default=False)

    args = parser.parse_args()
    if args.DDIB_fusion:
        args.fusion_aib = True
    args.DDIB_uni_condition_set = parse_condition_subset(args.DDIB_uni_conditions)
    if (args.DDIB_uni or args.DDIB_fusion) and args.DDIB_deterministic_eval:
        args.deterministic_eval_aib = True

    assert 0.0 <= args.uni_aib_residual_alpha <= 1.0, '--uni-aib-residual-alpha must be in [0, 1]'
    assert args.first_stage_ib_warmup >= 0.0, '--first-stage-ib-warmup must be non-negative'
    assert args.stage1_joint_weight >= 0.0, '--stage1-joint-weight must be non-negative'
    assert not (args.restore_best_stage1_attention and args.restore_best_stage1_branches), \
        'choose at most one stage-1 branch restoration mode'
    assert args.DDIB_uni_condition_weight >= 0.0, '--DDIB-uni-condition-weight must be non-negative'
    assert 0.0 <= args.DDIB_fusion_residual_alpha <= 1.0, '--DDIB-fusion-residual-alpha must be in [0, 1]'
    assert 0.0 < args.DDIB_fusion_gate_init < 1.0, '--DDIB-fusion-gate-init must be in (0, 1)'
    assert args.DDIB_fusion_residual_rank > 0, '--DDIB-fusion-residual-rank must be positive'
    assert args.DDIB_fusion_gate_reg_lambda >= 0.0, '--DDIB-fusion-gate-reg-lambda must be non-negative'
    assert args.DDIB_fusion_anchor_weight >= 0.0, '--DDIB-fusion-anchor-weight must be non-negative'
    assert args.DDIB_fusion_anchor_temperature > 0.0, '--DDIB-fusion-anchor-temperature must be positive'
    assert args.DDIB_fusion_alpha_warmup >= 0.0, '--DDIB-fusion-alpha-warmup must be non-negative'
    assert args.DDIB_fusion_start_epoch >= 0.0, '--DDIB-fusion-start-epoch must be non-negative'
    assert 0.0 <= args.label_smoothing < 1.0, '--label-smoothing must be in [0, 1)'
    assert args.grad_clip >= 0.0, '--grad-clip must be non-negative'
    assert args.warmup_epochs >= 0, '--warmup-epochs must be non-negative'
    assert args.min_lr >= 0.0, '--min-lr must be non-negative'
    assert 0.0 <= args.ema_decay < 1.0, '--ema-decay must be in [0, 1)'
    return args
