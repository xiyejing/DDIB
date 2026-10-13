import os
import random
import time
import warnings
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).with_name('Src')
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch
import torch.optim as optim
from arguments import parse_args
from calibration import (
    calibrate_fusion_capacities, calibrate_rc_fusion_capacities,
    calibrate_rc_unimodal_capacities, calibrate_unimodal_capacities,
    compute_class_weights,
)
from data_pipeline import build_model, get_loaders
from engine import train_or_eval_model
from loss import MaskedCELoss, MaskedMSELoss
from checkpoint_selection import (
    CHECKPOINT_EVAL_VARIANTS, CHECKPOINT_SELECTION, CHECKPOINT_TIE_BREAK,
    choose_best_index, choose_result_variant, resolve_selection_start,
)
from training_tricks import build_ema, build_scheduler
from training_support import (
    TEST_CONDITIONS, condition_modalities, DDIB_fusion_alpha_scale,
    expand_condition_schedule_from_probs, format_aib_stats,
    generic_condition_probabilities, get_condition_arg,
    rc_modality_rate_from_stats, update_bounded_dual_lambda, update_dual_beta,
)
import config

warnings.filterwarnings("ignore")
cpu_threads = int(os.environ.get("CPU_THREADS", os.environ.get("OMP_NUM_THREADS", "1")))
if cpu_threads > 0:
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(max(1, min(2, cpu_threads)))
    except RuntimeError:
        pass

if __name__ == '__main__':
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    args.device = device
    def seed_torch(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    seed_torch(args.seed)


    if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
        args.num_folder = 1
        args.n_classes = 1
        args.n_speakers = 1
    elif args.dataset == 'IEMOCAPFOUR':
        args.num_folder = 5
        args.n_classes = 4
        args.n_speakers = 2
    elif args.dataset == 'IEMOCAPSIX':
        args.num_folder = 5
        args.n_classes = 6
        args.n_speakers = 2
    if args.checkpoint_selection == 'auto':
        # Sentiment regression selects by weighted F1; emotion classification
        # uses the balanced Acc/UA criterion.
        args.checkpoint_selection = 'best_f1' if args.dataset in ['CMUMOSI', 'CMUMOSEI'] else CHECKPOINT_SELECTION
    if args.end_folder is None:
        args.end_folder = args.num_folder
    assert 0 <= args.start_folder < args.end_folder <= args.num_folder, f'Invalid folder range: [{args.start_folder}, {args.end_folder})'
    if args.ablation == 'wo-imd' and args.stage_epoch != 0:
        print(f"w/o IMD: overriding stage_epoch from {args.stage_epoch} to 0.")
        args.stage_epoch = 0
    cuda = torch.cuda.is_available() and not args.no_cuda
    print(
        f'config: dataset={args.dataset} condition={args.test_condition} seed={args.seed} '
        f'epochs={args.epochs} stage={args.stage_epoch}'
    )
    audio_feature, text_feature, video_feature = args.audio_feature, args.text_feature, args.video_feature
    audio_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], audio_feature)
    text_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], text_feature)
    video_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], video_feature)
    assert os.path.exists(audio_root) and os.path.exists(text_root) and os.path.exists(video_root), f'features not exist!'
    # Detect fold-specific feature directories.
    use_per_fold = os.path.exists(os.path.join(audio_root, 'fold_0'))
    if not use_per_fold:
        train_loaders, test_loaders, adim, tdim, vdim = get_loaders(audio_root=audio_root,
                                                                                 text_root=text_root,
                                                                                 video_root=video_root,
                                                                                 num_folder=args.num_folder,
                                                                                 batch_size=args.batch_size,
                                                                                 dataset=args.dataset)
        assert len(train_loaders) == args.num_folder, f'Error: folder number'
    else:
        # Infer feature dimensions from fold_0.
        fold0_audio = os.path.join(audio_root, 'fold_0')
        fold0_text = os.path.join(text_root, 'fold_0')
        fold0_video = os.path.join(video_root, 'fold_0')
        _, _, adim, tdim, vdim = get_loaders(audio_root=fold0_audio,
                                              text_root=fold0_text,
                                              video_root=fold0_video,
                                              num_folder=args.num_folder,
                                              batch_size=args.batch_size,
                                              dataset=args.dataset)
        print('using per-fold features')


    print('start training')
    all_conditions = args.test_condition == 'all'
    all_train_conditions = TEST_CONDITIONS
    if all_conditions and args.generic_condition_schedule:
        generic_probs = generic_condition_probabilities(args)
        all_train_conditions = expand_condition_schedule_from_probs(generic_probs, length=100)
        print(
            "generic_condition_schedule probs:"
            + ",".join([f"{condition}={generic_probs[idx]:.4f}" for idx, condition in enumerate(TEST_CONDITIONS)])
        )
    elif all_conditions and args.all_train_conditions:
        all_train_conditions = [condition.strip() for condition in args.all_train_conditions.split(',') if condition.strip()]
        invalid_conditions = [condition for condition in all_train_conditions if condition not in TEST_CONDITIONS]
        assert len(all_train_conditions) > 0, '--all-train-conditions must contain at least one condition'
        assert not invalid_conditions, f'Invalid --all-train-conditions entries: {invalid_conditions}'
    if all_conditions:
        print(f'train_conditions:{",".join(all_train_conditions)}')
    folder_mae = []
    folder_corr = []
    folder_acc = []
    folder_f1 = []
    folder_mae_all = {condition: [] for condition in TEST_CONDITIONS}
    folder_corr_all = {condition: [] for condition in TEST_CONDITIONS}
    folder_acc_all = {condition: [] for condition in TEST_CONDITIONS}
    folder_f1_all = {condition: [] for condition in TEST_CONDITIONS}
    initial_ib_beta_a = args.ib_beta_a
    initial_ib_beta_t = args.ib_beta_t
    initial_ib_beta_v = args.ib_beta_v
    initial_fusion_ib_beta = args.fusion_ib_beta
    initial_uni_caps = {modality: getattr(args, f'uni_cap_{modality}') for modality in ['a', 't', 'v']}
    initial_fusion_caps = {condition: getattr(args, f'fusion_cap_{condition}') for condition in TEST_CONDITIONS}
    for ii in range(args.start_folder, args.end_folder):
        print(f'fold {ii + 1}/{args.num_folder}')
        args.ib_beta_a = initial_ib_beta_a
        args.ib_beta_t = initial_ib_beta_t
        args.ib_beta_v = initial_ib_beta_v
        args.fusion_ib_beta = initial_fusion_ib_beta
        for modality, cap in initial_uni_caps.items():
            setattr(args, f'uni_cap_{modality}', cap)
        for condition, cap in initial_fusion_caps.items():
            setattr(args, f'fusion_cap_{condition}', cap)

        if use_per_fold:
            fold_audio = os.path.join(audio_root, f'fold_{ii}')
            fold_text = os.path.join(text_root, f'fold_{ii}')
            fold_video = os.path.join(video_root, f'fold_{ii}')
            fold_train_loaders, fold_test_loaders, _, _, _ = get_loaders(audio_root=fold_audio,
                                                                          text_root=fold_text,
                                                                          video_root=fold_video,
                                                                          num_folder=args.num_folder,
                                                                          batch_size=args.batch_size,
                                                                          dataset=args.dataset)
            train_loader = fold_train_loaders[ii]
            test_loader = fold_test_loaders[ii]
        else:
            train_loader = train_loaders[ii]
            test_loader = test_loaders[ii]
        start_time = time.time()

        model = build_model(args, adim, tdim, vdim)
        reg_loss = MaskedMSELoss()
        class_weight = None
        if args.balanced_ce and args.dataset in ['IEMOCAPFOUR', 'IEMOCAPSIX']:
            class_weight = compute_class_weights(train_loader, args.n_classes, device)
            if args.verbose:
                print(f"balanced_ce weights={class_weight.detach().cpu().numpy().round(4).tolist()}")
        cls_loss = MaskedCELoss(weight=class_weight, label_smoothing=args.label_smoothing)
        if cuda:
            model.to(device)
        optimizer_cls = optim.AdamW if args.optimizer == 'AdamW' else optim.Adam
        optimizer = optimizer_cls([{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.l2}])
        first_stage_epochs = int(args.stage_epoch) if args.stage_epoch > 0 else args.epochs
        scheduler = build_scheduler(args, optimizer, total_epochs=max(1, first_stage_epochs))
        ema = build_ema(args, model)
        train_acc_as, train_acc_ts, train_acc_vs = [], [], []
        train_acc_as_common, train_acc_ts_common, train_acc_vs_common = [], [], []

        test_acc_as, test_acc_ts, test_acc_vs = [], [], []
        test_acc_as_common, test_acc_ts_common, test_acc_vs_common = [], [], []
        test_fscores, test_accs, test_maes, test_corrs = [], [], [], []
        test_stats = []
        test_variants = []
        test_fscores_all = {condition: [] for condition in TEST_CONDITIONS}
        test_accs_all = {condition: [] for condition in TEST_CONDITIONS}
        test_maes_all = {condition: [] for condition in TEST_CONDITIONS}
        test_corrs_all = {condition: [] for condition in TEST_CONDITIONS}
        test_stats_all = {condition: [] for condition in TEST_CONDITIONS}
        test_variants_all = {condition: [] for condition in TEST_CONDITIONS}
        best_branch_scores = {'a': -1.0, 't': -1.0, 'v': -1.0}
        best_branch_epochs = {'a': 0, 't': 0, 'v': 0}
        best_branch_states = {'a': None, 't': None, 'v': None}
        start_first_stage_time = time.time()
        end_first_stage_time = start_first_stage_time

        print('stage: first' if args.stage_epoch > 0 else 'stage: second')
        uni_dual_betas = {
            'a': args.ib_beta_a if args.ib_beta_a is not None else args.ib_beta,
            't': args.ib_beta_t if args.ib_beta_t is not None else args.ib_beta,
            'v': args.ib_beta_v if args.ib_beta_v is not None else args.ib_beta,
        }
        fusion_dual_betas = {condition: args.fusion_ib_beta for condition in TEST_CONDITIONS}
        uni_rate_ema = {'a': None, 't': None, 'v': None}
        fusion_rate_ema = {condition: None for condition in TEST_CONDITIONS}
        rc_uni_lambdas = {modality: args.rc_lambda_init for modality in ['a', 't', 'v']}
        rc_fusion_lambdas = {condition: args.rc_lambda_init for condition in TEST_CONDITIONS}
        rc_uni_rate_ema = {'a': None, 't': None, 'v': None}
        rc_fusion_rate_ema = {condition: None for condition in TEST_CONDITIONS}
        if args.auto_uni_capacity and args.stage_epoch == 0:
            print("calibrating unimodal capacities")
            calibrate_unimodal_capacities(args, model, train_loader)
        if args.auto_fusion_capacity and args.stage_epoch == 0:
            print("calibrating fusion capacities")
            calibrate_fusion_capacities(args, model, train_loader)
        condition_loss_ema = {condition: None for condition in TEST_CONDITIONS}
        scheduler_rng = np.random.default_rng(args.seed + ii)
        base_counts = {condition: float(all_train_conditions.count(condition)) for condition in TEST_CONDITIONS}
        total_base_count = sum(base_counts.values())
        if total_base_count <= 0:
            base_probs = np.ones(len(TEST_CONDITIONS), dtype=np.float64) / len(TEST_CONDITIONS)
        else:
            base_probs = np.array([base_counts[condition] / total_base_count for condition in TEST_CONDITIONS], dtype=np.float64)

        def evaluate_model(first_stage, test_condition=None, variant='raw'):
            use_ema_eval = ema is not None and (not first_stage) and variant == 'ema'
            if use_ema_eval:
                with ema.apply_to(model):
                    return train_or_eval_model(
                        args, model, reg_loss, cls_loss, test_loader,
                        optimizer=None, train=False, first_stage=first_stage, mark='test',
                        test_condition=test_condition,
                    )
            return train_or_eval_model(
                args, model, reg_loss, cls_loss, test_loader,
                optimizer=None, train=False, first_stage=first_stage, mark='test',
                test_condition=test_condition,
            )

        def evaluate_checkpoint(first_stage, test_condition=None):
            variants = ['raw']
            if (not first_stage) and ema is not None:
                if CHECKPOINT_EVAL_VARIANTS == 'ema':
                    variants = ['ema']
                elif CHECKPOINT_EVAL_VARIANTS == 'both':
                    variants = ['raw', 'ema']
            variant_results = [
                (variant, evaluate_model(first_stage=first_stage, test_condition=test_condition, variant=variant))
                for variant in variants
            ]
            return choose_result_variant(variant_results, args.checkpoint_selection, CHECKPOINT_TIE_BREAK)

        def keep_best_first_stage_branches(epoch, branch_accs):
            attention_patterns = {
                'a': ('intra_attn_as.',),
                't': ('intra_attn_ts.',),
                'v': ('intra_attn_vs.',),
            }
            full_branch_patterns = {
                'a': ('as_in_proj.', 'as_vib.', 'intra_attn_as.', 'classifier_a.'),
                't': ('ts_in_proj.', 'ts_vib.', 'intra_attn_ts.', 'classifier_t.'),
                'v': ('vs_in_proj.', 'vs_vib.', 'intra_attn_vs.', 'classifier_v.'),
            }
            patterns = full_branch_patterns if args.restore_best_stage1_branches else attention_patterns
            for index, modality in enumerate(['a', 't', 'v']):
                score = float(branch_accs[index])
                if score <= best_branch_scores[modality]:
                    continue
                best_branch_scores[modality] = score
                best_branch_epochs[modality] = epoch
                modality_patterns = patterns[modality]
                best_branch_states[modality] = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                    if any(key.startswith(pattern) for pattern in modality_patterns)
                }

        for epoch in range(args.epochs):
            if scheduler is not None:
                epoch_in_stage = epoch if epoch < args.stage_epoch else epoch - int(args.stage_epoch)
                current_lr = scheduler.step(epoch_in_stage)
                if args.verbose:
                    print(f'lr epoch={epoch} value={current_lr:.6g}')
            first_stage = True if epoch < args.stage_epoch else False
            args.current_epoch = epoch
            if hasattr(model, 'DDIB_fusion_alpha_scale'):
                model.DDIB_fusion_alpha_scale = DDIB_fusion_alpha_scale(args, first_stage)
            train_condition = None
            if all_conditions and not first_stage:
                if args.train_loss_condition_scheduler and all(v is not None for v in condition_loss_ema.values()):
                    losses_for_scheduler = np.array([condition_loss_ema[condition] for condition in TEST_CONDITIONS], dtype=np.float64)
                    mean_loss = max(float(losses_for_scheduler.mean()), 1e-8)
                    difficulty = np.maximum(losses_for_scheduler / mean_loss - 1.0, 0.0)
                    adapt_logits = difficulty / max(args.train_loss_scheduler_tau, 1e-8)
                    adapt_logits = adapt_logits - adapt_logits.max()
                    adapt_probs = np.exp(adapt_logits)
                    adapt_probs = adapt_probs / adapt_probs.sum()
                    condition_probs = (1.0 - args.train_loss_scheduler_alpha) * base_probs + args.train_loss_scheduler_alpha * adapt_probs
                    condition_probs = condition_probs / condition_probs.sum()
                    train_condition = scheduler_rng.choice(TEST_CONDITIONS, p=condition_probs)
                else:
                    train_condition = all_train_conditions[(epoch - int(args.stage_epoch)) % len(all_train_conditions)]
            active_condition = train_condition or args.test_condition
            if args.uni_dual_aib:
                args.ib_beta_a = uni_dual_betas['a']
                args.ib_beta_t = uni_dual_betas['t']
                args.ib_beta_v = uni_dual_betas['v']
            if args.fusion_dual_aib and args.fusion_aib and not first_stage and active_condition in fusion_dual_betas:
                args.fusion_ib_beta = fusion_dual_betas[active_condition]
            if args.rc_aib or args.rc_uni_aib:
                args.rc_lambda_a = rc_uni_lambdas['a']
                args.rc_lambda_t = rc_uni_lambdas['t']
                args.rc_lambda_v = rc_uni_lambdas['v']
            if args.rc_fusion_aib:
                for condition in TEST_CONDITIONS:
                    setattr(args, f'rc_fusion_lambda_{condition}', rc_fusion_lambdas[condition])
            # Train one epoch and evaluate the active condition.
            train_mae, train_corr, train_acc, train_fscore, train_acc_atv, train_names, train_loss, train_aib_stats = train_or_eval_model(args, model, reg_loss, cls_loss, train_loader, \
                                                                            optimizer=optimizer, train=True, first_stage=first_stage, mark='train',
                                                                            test_condition=train_condition, ema=ema)
            if all_conditions and (not first_stage) and args.train_loss_condition_scheduler and train_condition is not None:
                old_loss = condition_loss_ema[train_condition]
                loss_value = float(train_loss)
                if loss_value <= 0 and args.dataset in ['IEMOCAPFOUR', 'IEMOCAPSIX']:
                    # Fall back to classification error when loss is unavailable.
                    loss_value = 1.0 - float(train_corr)
                if old_loss is None:
                    condition_loss_ema[train_condition] = loss_value
                else:
                    condition_loss_ema[train_condition] = args.train_loss_scheduler_ema * old_loss + (1.0 - args.train_loss_scheduler_ema) * loss_value
            update_uni_dual = args.uni_dual_aib and (
                (first_stage and args.dual_update_stage in ['first', 'both'])
                or ((not first_stage) and args.dual_update_stage in ['second', 'both'])
            )
            if update_uni_dual:
                for modality in ['a', 't', 'v']:
                    cap = getattr(args, f'uni_cap_{modality}')
                    rate = train_aib_stats.get(f'{modality}_kl')
                    if rate is not None:
                        uni_dual_betas[modality], uni_rate_ema[modality] = update_dual_beta(
                        uni_dual_betas[modality], uni_rate_ema[modality], rate, cap, args.dual_lr_uni, args.dual_ema)
                if args.verbose:
                    print(f"dual_uni_beta epoch={epoch} beta=({uni_dual_betas['a']:.4g},{uni_dual_betas['t']:.4g},{uni_dual_betas['v']:.4g})")
            if args.fusion_dual_aib and args.fusion_aib and not first_stage and active_condition in fusion_dual_betas:
                cap = get_condition_arg(args, 'fusion_cap', active_condition)
                rate = train_aib_stats.get('fusion_kl')
                if rate is not None:
                    fusion_dual_betas[active_condition], fusion_rate_ema[active_condition] = update_dual_beta(
                        fusion_dual_betas[active_condition], fusion_rate_ema[active_condition], rate, cap, args.dual_lr_fusion, args.dual_ema)
                if args.verbose:
                    print(f"dual_fusion_beta epoch={epoch} condition={active_condition} beta={fusion_dual_betas[active_condition]:.4g}")
            if (args.rc_aib or args.rc_uni_aib) and not first_stage:
                for modality in condition_modalities(active_condition):
                    cap = getattr(args, f'uni_cap_{modality}')
                    rate = rc_modality_rate_from_stats(train_aib_stats, modality)
                    rc_uni_lambdas[modality], rc_uni_rate_ema[modality] = update_bounded_dual_lambda(
                        rc_uni_lambdas[modality],
                        rc_uni_rate_ema[modality],
                        rate,
                        cap,
                        args.rc_dual_lr_uni,
                        args.rc_dual_ema,
                        args.rc_lambda_max,
                        args.rc_eps,
                    )
                if args.verbose:
                    print(f"rc_uni_lambda epoch={epoch} condition={active_condition} lambda=({rc_uni_lambdas['a']:.4g},{rc_uni_lambdas['t']:.4g},{rc_uni_lambdas['v']:.4g})")
            if args.rc_fusion_aib and args.fusion_aib and not first_stage and active_condition in rc_fusion_lambdas:
                cap = get_condition_arg(args, 'fusion_cap', active_condition)
                rate = train_aib_stats.get('fusion_kl')
                rc_fusion_lambdas[active_condition], rc_fusion_rate_ema[active_condition] = update_bounded_dual_lambda(
                    rc_fusion_lambdas[active_condition],
                    rc_fusion_rate_ema[active_condition],
                    rate,
                    cap,
                    args.rc_dual_lr_fusion,
                    args.rc_dual_ema,
                    args.rc_lambda_max,
                    args.rc_eps,
                )
                if args.verbose:
                    print(f"rc_fusion_lambda epoch={epoch} condition={active_condition} lambda={rc_fusion_lambdas[active_condition]:.4g}")
            if all_conditions and not first_stage:
                test_results_all = {}
                test_variants_epoch = {}
                for condition in TEST_CONDITIONS:
                    variant, result = evaluate_checkpoint(first_stage=False, test_condition=condition)
                    test_results_all[condition] = result
                    test_variants_epoch[condition] = variant
            else:
                test_variant, test_result = evaluate_checkpoint(first_stage=first_stage)
                test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, test_aib_stats = test_result

            if all_conditions and not first_stage:
                for condition, condition_result in test_results_all.items():
                    condition_mae, condition_corr, condition_acc, condition_fscore = condition_result[:4]
                    test_accs_all[condition].append(condition_acc)
                    test_fscores_all[condition].append(condition_fscore)
                    test_maes_all[condition].append(condition_mae)
                    test_corrs_all[condition].append(condition_corr)
                    test_stats_all[condition].append(condition_result[7])
                    test_variants_all[condition].append(test_variants_epoch[condition])
            else:
                test_accs.append(test_acc)
                test_fscores.append(test_fscore)
                test_maes.append(test_mae)
                test_corrs.append(test_corr)
                test_stats.append(test_aib_stats)
                test_variants.append(test_variant)
            train_acc_as.append(train_acc_atv[0])
            train_acc_ts.append(train_acc_atv[1])
            train_acc_vs.append(train_acc_atv[2])
            train_acc_as_common.append(train_acc_atv[3])
            train_acc_ts_common.append(train_acc_atv[4])
            train_acc_vs_common.append(train_acc_atv[5])
            if not (all_conditions and not first_stage):
                test_acc_as.append(test_acc_atv[0])
                test_acc_ts.append(test_acc_atv[1])
                test_acc_vs.append(test_acc_atv[2])
                test_acc_as_common.append(test_acc_atv[3])
                test_acc_ts_common.append(test_acc_atv[4])
                test_acc_vs_common.append(test_acc_atv[5])
                if first_stage:
                    if args.restore_best_stage1_attention or args.restore_best_stage1_branches:
                        keep_best_first_stage_branches(epoch, test_acc_atv)

            if first_stage:
                print(f'epoch={epoch:03d} stage=1 train_acc={train_acc:.3f} test_acc={test_acc:.3f} test_ua={test_corr:.3f}')
            elif all_conditions:
                metric_text = " ".join(
                    f"{condition}:{test_results_all[condition][2]:.3f}/{test_results_all[condition][1]:.3f}"
                    for condition in TEST_CONDITIONS
                )
                print(f'epoch={epoch:03d} stage=2 train_condition={train_condition} train_acc={train_acc:.3f} test_acc/ua {metric_text}')
            else:
                print(f'epoch={epoch:03d} stage=2 condition={args.test_condition} train_acc={train_acc:.3f} test_acc={test_acc:.3f} test_ua={test_corr:.3f}')
            # Initialize stage 2.
            if args.stage_epoch > 0 and epoch == args.stage_epoch-1:
                if args.restore_best_stage1_attention or args.restore_best_stage1_branches:
                    current_state = model.state_dict()
                    for modality in ['a', 't', 'v']:
                        if best_branch_states[modality] is not None:
                            current_state.update(best_branch_states[modality])
                    model.load_state_dict(current_state)
                if args.reset_optimizer_stage2:
                    optimizer = optimizer_cls([{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.l2}])
                    scheduler = build_scheduler(
                        args, optimizer, total_epochs=max(1, args.epochs - int(args.stage_epoch))
                    )
                ema = build_ema(args, model)

                end_first_stage_time = time.time()
                if args.auto_uni_capacity:
                    print("calibrating unimodal capacities")
                    calibrate_unimodal_capacities(args, model, train_loader)
                if args.auto_fusion_capacity:
                    print("calibrating fusion capacities")
                    calibrate_fusion_capacities(args, model, train_loader)
                if args.rc_aib or args.rc_uni_aib or args.rc_fusion_aib:
                    print("calibrating constrained-rate capacities")
                    _, _, _, _, _, _, _, rc_warm_stats = train_or_eval_model(
                        args,
                        model,
                        reg_loss,
                        cls_loss,
                        train_loader,
                        optimizer=None,
                        train=False,
                        first_stage=False,
                        mark='rc_warm',
                        test_condition='atv',
                    )
                    if args.rc_aib or args.rc_uni_aib:
                        calibrate_rc_unimodal_capacities(args, model, train_loader, rc_warm_stats)
                        rc_uni_lambdas = {modality: args.rc_lambda_init for modality in ['a', 't', 'v']}
                        rc_uni_rate_ema = {'a': None, 't': None, 'v': None}
                    if args.rc_fusion_aib and args.fusion_aib:
                        rc_warm_stats_by_condition = {}
                        for condition in TEST_CONDITIONS:
                            _, _, _, _, _, _, _, condition_warm_stats = train_or_eval_model(
                                args,
                                model,
                                reg_loss,
                                cls_loss,
                                train_loader,
                                optimizer=None,
                                train=False,
                                first_stage=False,
                                mark='rc_warm',
                                test_condition=condition,
                            )
                            rc_warm_stats_by_condition[condition] = condition_warm_stats
                        calibrate_rc_fusion_capacities(args, model, train_loader, rc_warm_stats_by_condition)
                        rc_fusion_lambdas = {condition: args.rc_lambda_init for condition in TEST_CONDITIONS}
                        rc_fusion_rate_ema = {condition: None for condition in TEST_CONDITIONS}
                print(
                    "stage: second; selected_branches "
                    f"a={best_branch_epochs['a']} t={best_branch_epochs['t']} v={best_branch_epochs['v']}"
                )

        end_second_stage_time = time.time()
        print(f'time first={end_first_stage_time - start_first_stage_time:.1f}s second={end_second_stage_time - end_first_stage_time:.1f}s')
        if all_conditions:
            for condition in TEST_CONDITIONS:
                best_index_test = choose_best_index(
                    test_accs_all[condition],
                    test_corrs_all[condition],
                    test_fscores_all[condition],
                    args.checkpoint_selection,
                    start=0,
                    tie_break=CHECKPOINT_TIE_BREAK,
                )

                bestmae = test_maes_all[condition][best_index_test]
                bestcorr = test_corrs_all[condition][best_index_test]
                bestf1 = test_fscores_all[condition][best_index_test]
                bestacc = test_accs_all[condition][best_index_test]
                beststats = test_stats_all[condition][best_index_test]
                bestvariant = test_variants_all[condition][best_index_test]

                folder_mae_all[condition].append(bestmae)
                folder_corr_all[condition].append(bestcorr)
                folder_f1_all[condition].append(bestf1)
                folder_acc_all[condition].append(bestacc)

                print(
                    f"best condition={condition} epoch={best_index_test + int(args.stage_epoch)} "
                    f"acc={bestacc:.4f} f1={bestf1:.4f} corr={bestcorr:.4f} mae={bestmae:.4f}"
                )
                if args.verbose:
                    print(f"selected_variant condition={condition}: {bestvariant}")
                    print(f"AIB stats condition={condition}: {format_aib_stats(beststats, condition)}")
        else:
            selection_start = resolve_selection_start(
                len(test_accs),
                stage_epoch=args.stage_epoch,
                epochs=args.epochs,
                mask_rate=getattr(args, 'mask_rate', None),
            )
            best_index_test = choose_best_index(
                test_accs,
                test_corrs,
                test_fscores,
                args.checkpoint_selection,
                start=selection_start,
                tie_break=CHECKPOINT_TIE_BREAK,
            )


            bestmae = test_maes[best_index_test]
            bestcorr = test_corrs[best_index_test]
            bestf1 = test_fscores[best_index_test]
            bestacc = test_accs[best_index_test]
            beststats = test_stats[best_index_test]
            bestvariant = test_variants[best_index_test]

            folder_mae.append(bestmae)
            folder_corr.append(bestcorr)
            folder_f1.append(bestf1)
            folder_acc.append(bestacc)
        end_time = time.time()

        if not all_conditions:
            print(
                f"best condition={args.test_condition} epoch={best_index_test} "
                f"acc={bestacc:.4f} f1={bestf1:.4f} corr={bestcorr:.4f} mae={bestmae:.4f}"
            )
            if args.verbose:
                print(f"selected_variant condition={args.test_condition}: {bestvariant}")
                print(f"AIB stats condition={args.test_condition}: {format_aib_stats(beststats, args.test_condition)}")
            print(f'fold {ii + 1} done in {end_time - start_time:.1f}s')
        if all_conditions:
            print(f'fold {ii + 1} done in {end_time - start_time:.1f}s')

    if all_conditions:
        for condition in TEST_CONDITIONS:
            print(f"average condition={condition} acc={np.mean(folder_acc_all[condition]):.4f} ua={np.mean(folder_corr_all[condition]):.4f}")
    else:
        print(f"average condition={args.test_condition} acc={np.mean(folder_acc):.4f} ua={np.mean(folder_corr):.4f}")
