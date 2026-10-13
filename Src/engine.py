import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, recall_score
from Dominance import compute_DDIB_dominance_weights
from training_support import (
    build_masked_batch, condition_modalities, DDIB_uni_condition_scale,
    first_stage_ib_scale, get_condition_arg, ib_penalty, masked_hidden_margin_mse,
    masked_hidden_mse, masked_token_jsd, masked_token_kl, normalized_rate_penalty,
    second_stage_ib_scale,
)

def train_or_eval_model(args, model, reg_loss, cls_loss, dataloader, optimizer=None, train=False, first_stage=True, mark='train', test_condition=None, ema=None):
    preds, preds_a, preds_t, preds_v, masks, labels = [], [], [], [], [], []
    preds_a_common, preds_t_common, preds_v_common, = [], [], []
    preds_intra_a, preds_intra_t, preds_intra_v, preds_inter_a, preds_inter_t, preds_inter_v,  = [], [], [], [], [], []
    losses = []
    stats_sums = {}
    stats_weight = 0.0
    dataset = args.dataset
    if dataset not in ['CMUMOSI', 'CMUMOSEI', 'IEMOCAPFOUR', 'IEMOCAPSIX']:
        raise ValueError(f'Unsupported dataset: {dataset}')
    test_condition = test_condition or args.test_condition
    if hasattr(model, 'test_condition'):
        model.test_condition = test_condition

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    use_msd = args.ablation not in ['wo-msd', 'wo-imd']
    use_mcd = args.ablation not in ['wo-mcd', 'wo-imd']
    ib_beta_a = args.ib_beta_a if args.ib_beta_a is not None else args.ib_beta
    ib_beta_t = args.ib_beta_t if args.ib_beta_t is not None else args.ib_beta
    ib_beta_v = args.ib_beta_v if args.ib_beta_v is not None else args.ib_beta
    uni_cap_a = args.uni_cap_a
    uni_cap_t = args.uni_cap_t
    uni_cap_v = args.uni_cap_v
    fusion_cap = get_condition_arg(args, 'fusion_cap', test_condition)

    def add_loss(total, term):
        return term if total is None else total + term

    def optimizer_step():
        grad_clip = float(getattr(args, 'grad_clip', 0.0))
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)

    for data in dataloader:
        vidnames = []
        if train: optimizer.zero_grad()
        
        vidnames += data[-1]
        masked_input, input_mask, umask, label = build_masked_batch(args, data, test_condition, first_stage)
        hidden, out, outputs_intras, outputs_inters, vib_outputs, vib_kls = model(masked_input, input_mask, umask, first_stage)
        mc_eval_samples = max(1, int(getattr(args, 'mc_eval_samples', 1)))
        if (not train) and (not first_stage) and mc_eval_samples > 1:
            sampled_outs = [out]
            with torch.no_grad():
                for _ in range(mc_eval_samples - 1):
                    _, sampled_out, *_ = model(masked_input, input_mask, umask, first_stage)
                    sampled_outs.append(sampled_out)
            if dataset in ['IEMOCAPFOUR', 'IEMOCAPSIX']:
                out = torch.stack([F.softmax(sampled_out, dim=-1) for sampled_out in sampled_outs], dim=0).mean(dim=0)
            else:
                out = torch.stack(sampled_outs, dim=0).mean(dim=0)
        batch_weight = float(umask.sum().detach().cpu().item())
        if hasattr(model, 'last_vib_stats'):
            for stat_name, stat_value in model.last_vib_stats.items():
                stats_sums[stat_name] = stats_sums.get(stat_name, 0.0) + float(stat_value.detach().cpu().item()) * batch_weight
            stats_weight += batch_weight
        outputs_intra_a, outputs_intra_t, outputs_intra_v = outputs_intras[0], outputs_intras[1], outputs_intras[2]
        outputs_inter_a, outputs_inter_t, outputs_inter_v = outputs_inters[0], outputs_inters[1], outputs_inters[2]
        out_a, out_t, out_v, out_a_common, out_t_common, out_v_common = vib_outputs[0], vib_outputs[1], vib_outputs[2], vib_outputs[3], vib_outputs[4], vib_outputs[5]
        a_kls, t_kls, v_kls, a_kls_common, t_kls_common, v_kls_common = vib_kls[0], vib_kls[1], vib_kls[2], vib_kls[3], vib_kls[4], vib_kls[5]
        uni_kls = {
            'a': (a_kls, a_kls_common),
            't': (t_kls, t_kls_common),
            'v': (v_kls, v_kls_common),
        }
        uni_logits = {
            'a': (out_a, out_a_common),
            't': (out_t, out_t_common),
            'v': (out_v, out_v_common),
        }
        uni_betas = {
            'a': ib_beta_a,
            't': ib_beta_t,
            'v': ib_beta_v,
        }
        uni_caps = {
            'a': uni_cap_a,
            't': uni_cap_t,
            'v': uni_cap_v,
        }
        current_fusion_kl = getattr(model, 'last_fusion_kl', out.new_tensor(0.0))
        DDIB_weights = compute_DDIB_dominance_weights(args, model, data, test_condition, train, first_stage)
        if DDIB_weights is not None:
            if 'uni' in DDIB_weights:
                for modality, weight_value in DDIB_weights['uni'].items():
                    stats_sums[f'DDIB_u_{modality}'] = stats_sums.get(f'DDIB_u_{modality}', 0.0) + float(weight_value.detach().cpu().item()) * batch_weight
            if 'fusion' in DDIB_weights:
                stats_sums['DDIB_fusion'] = stats_sums.get('DDIB_fusion', 0.0) + float(DDIB_weights['fusion'].detach().cpu().item()) * batch_weight
        ib_scale = second_stage_ib_scale(args, first_stage)
        first_ib_scale = first_stage_ib_scale(args, first_stage)
        skip_condition_ib = (
            (not first_stage)
            and getattr(args, 'second_stage_ib_skip_full', False)
            and test_condition == 'atv'
        )
        rc_uni_enabled = getattr(args, 'rc_aib', False) or getattr(args, 'rc_uni_aib', False)
        rc_fusion_enabled = getattr(args, 'rc_fusion_aib', False)
        rc_uni_lambdas = {
            'a': getattr(args, 'rc_lambda_a', getattr(args, 'rc_lambda_init', 0.0)),
            't': getattr(args, 'rc_lambda_t', getattr(args, 'rc_lambda_init', 0.0)),
            'v': getattr(args, 'rc_lambda_v', getattr(args, 'rc_lambda_init', 0.0)),
        }
        rc_fusion_lambda = get_condition_arg(args, 'rc_fusion_lambda', test_condition)
        if rc_fusion_lambda is None:
            rc_fusion_lambda = getattr(args, 'rc_lambda_init', 0.0)
        condition_ib_scale = 1.0
        if (not first_stage) and getattr(args, 'condition_size_ib_scaling', False):
            condition_ib_scale = (len(condition_modalities(test_condition)) / 3.0) ** getattr(args, 'condition_size_ib_zeta', 0.5)

        def second_stage_uni_penalty(modality, rate):
            if rc_uni_enabled:
                return normalized_rate_penalty(
                    rate,
                    uni_caps[modality],
                    rc_uni_lambdas[modality] * ib_scale * condition_ib_scale,
                    getattr(args, 'rc_kappa_uni', 0.0) * ib_scale * condition_ib_scale,
                    getattr(args, 'rc_eps', 1e-6),
                )
            return ib_penalty(rate, uni_betas[modality] * ib_scale * condition_ib_scale, uni_caps[modality], args.aug_kappa_uni * ib_scale * condition_ib_scale)

        def second_stage_fusion_penalty(rate):
            if rc_fusion_enabled:
                return normalized_rate_penalty(
                    rate,
                    fusion_cap,
                    rc_fusion_lambda * ib_scale,
                    getattr(args, 'rc_kappa_fusion', 0.0) * ib_scale,
                    getattr(args, 'rc_eps', 1e-6),
                )
            return ib_penalty(rate, args.fusion_ib_beta * ib_scale, fusion_cap, args.aug_kappa_fusion * ib_scale)

        def add_DDIB_loss(loss, loss_fn):
            if first_stage or not train or skip_condition_ib:
                return loss
            modalities = condition_modalities(test_condition)

            uni_condition_scale = DDIB_uni_condition_scale(args, test_condition)
            if getattr(args, 'DDIB_uni', False) and uni_condition_scale > 0.0:
                uni_weight_map = {}
                if DDIB_weights is not None:
                    uni_weight_map = DDIB_weights.get('uni', {})
                rate_terms = []
                aux_terms = []
                z_aux_terms = []
                for modality in modalities:
                    branch_rates = []
                    if use_msd:
                        branch_rates.append(uni_kls[modality][0])
                    if use_mcd:
                        branch_rates.append(uni_kls[modality][1])
                    if branch_rates:
                        dominance = uni_weight_map.get(modality, branch_rates[0].new_tensor(1.0)).to(branch_rates[0].device)
                        rate_terms.append(dominance * sum(branch_rates) / len(branch_rates))

                    if getattr(args, 'DDIB_aux_weight', 0.0) > 0:
                        explicit_aux_logits = getattr(model, 'last_DDIB_aux_logits', {})
                        branch_logits = explicit_aux_logits.get(modality)
                        if branch_logits is None:
                            branch_logits = []
                            if use_msd:
                                branch_logits.append(uni_logits[modality][0])
                            if use_mcd:
                                branch_logits.append(uni_logits[modality][1])
                        elif torch.is_tensor(branch_logits):
                            branch_logits = [branch_logits]
                        else:
                            branch_logits = list(branch_logits)
                            aux_head_mode = getattr(args, 'DDIB_aux_head_mode', 'both')
                            if aux_head_mode == 'specific':
                                branch_logits = branch_logits[:1]
                            elif aux_head_mode == 'common':
                                branch_logits = branch_logits[1:2]
                        for logits in branch_logits:
                            aux_terms.append(loss_fn(logits.view(-1, logits.size(2)), labels_, umask))

                    if getattr(args, 'DDIB_z_aux_weight', 0.0) > 0:
                        z_aux_logits = getattr(model, 'last_DDIB_z_aux_logits', {})
                        z_branch_logits = z_aux_logits.get(modality)
                        if z_branch_logits is None:
                            z_branch_logits = []
                        elif torch.is_tensor(z_branch_logits):
                            z_branch_logits = [z_branch_logits]
                        else:
                            z_branch_logits = list(z_branch_logits)
                            z_aux_head_mode = getattr(args, 'DDIB_z_aux_head_mode', 'specific')
                            if z_aux_head_mode == 'specific':
                                z_branch_logits = z_branch_logits[:1]
                            elif z_aux_head_mode == 'common':
                                z_branch_logits = z_branch_logits[1:2]
                        for logits in z_branch_logits:
                            z_aux_terms.append(loss_fn(logits.view(-1, logits.size(2)), labels_, umask))

                if rate_terms and getattr(args, 'DDIB_uni_beta', 0.0) > 0:
                    loss = loss + uni_condition_scale * args.DDIB_uni_beta * ib_scale * sum(rate_terms) / len(rate_terms)
                if aux_terms and getattr(args, 'DDIB_aux_weight', 0.0) > 0:
                    loss = loss + uni_condition_scale * args.DDIB_aux_weight * sum(aux_terms) / len(aux_terms)
                if z_aux_terms and getattr(args, 'DDIB_z_aux_weight', 0.0) > 0:
                    loss = loss + uni_condition_scale * args.DDIB_z_aux_weight * sum(z_aux_terms) / len(z_aux_terms)

            if getattr(args, 'DDIB_fusion', False) and args.fusion_aib and getattr(args, 'DDIB_fusion_beta', 0.0) > 0:
                fusion_weight = current_fusion_kl.new_tensor(1.0)
                if DDIB_weights is not None and 'fusion' in DDIB_weights:
                    fusion_weight = DDIB_weights['fusion'].to(current_fusion_kl.device)
                alpha_scale = float(getattr(model, 'DDIB_fusion_alpha_scale', 1.0))
                loss = loss + args.DDIB_fusion_beta * ib_scale * alpha_scale * fusion_weight * current_fusion_kl
            if getattr(args, 'DDIB_fusion', False) and getattr(args, 'DDIB_fusion_anchor_weight', 0.0) > 0:
                pre_fusion_out = getattr(model, 'last_pre_fusion_out', None)
                if pre_fusion_out is not None:
                    anchor = masked_token_kl(
                        out,
                        pre_fusion_out.detach(),
                        umask,
                        temperature=getattr(args, 'DDIB_fusion_anchor_temperature', 2.0),
                    )
                    loss = loss + args.DDIB_fusion_anchor_weight * anchor
            return loss

        def add_second_stage_decoder_loss(loss, loss_fn):
            if first_stage or not getattr(args, 'second_stage_uni_decoder', False):
                return loss
            if getattr(args, 'second_stage_uni_decoder_single_only', False) and len(condition_modalities(test_condition)) != 1:
                return loss
            decoder_outputs = getattr(model, 'last_second_stage_uni_logits', {})
            decoder_losses = []
            for modality in condition_modalities(test_condition):
                logits = decoder_outputs.get(modality)
                if logits is not None:
                    decoder_losses.append(loss_fn(logits.view(-1, logits.size(2)), labels_, umask))
            if decoder_losses:
                loss = loss + args.second_stage_uni_decoder_weight * sum(decoder_losses) / len(decoder_losses)
            return loss

        def add_condition_distill_loss(loss):
            if first_stage or not train or not getattr(args, 'condition_distill', False):
                return loss
            if test_condition == 'atv':
                return loss
            if getattr(args, 'condition_distill_single_only', False) and len(condition_modalities(test_condition)) != 1:
                return loss

            old_condition = getattr(model, 'test_condition', test_condition)
            was_training = model.training
            input_full, mask_full, umask_full, _ = build_masked_batch(args, data, 'atv', False)
            model.eval()
            with torch.no_grad():
                model.test_condition = 'atv'
                teacher_hidden, teacher_out, *_ = model(input_full, mask_full, umask_full, False)
            if was_training:
                model.train()
            model.test_condition = old_condition

            logit_weight = getattr(args, 'condition_distill_weight', 0.0)
            hidden_weight = getattr(args, 'condition_distill_hidden_weight', 0.0)
            if logit_weight > 0:
                if dataset in ['IEMOCAPFOUR', 'IEMOCAPSIX']:
                    loss = loss + logit_weight * masked_token_kl(
                        out,
                        teacher_out.detach(),
                        umask,
                        getattr(args, 'condition_distill_temperature', 1.0),
                        getattr(args, 'condition_distill_eps', 1e-8),
                    )
                else:
                    loss = loss + logit_weight * masked_hidden_mse(
                        out,
                        teacher_out.detach(),
                        umask,
                        getattr(args, 'condition_distill_eps', 1e-8),
                    )
            if hidden_weight > 0:
                loss = loss + hidden_weight * masked_hidden_mse(
                    hidden,
                    teacher_hidden.detach(),
                    umask,
                    getattr(args, 'condition_distill_eps', 1e-8),
                )
            return loss

        def add_condition_alignment_loss(loss):
            align_enabled = getattr(args, 'condition_align', False)
            sem_enabled = getattr(args, 'condition_semantic_kl', False)
            if first_stage or not train or test_condition == 'atv' or (not align_enabled and not sem_enabled):
                return loss

            old_condition = getattr(model, 'test_condition', test_condition)
            was_training = model.training
            input_full, mask_full, umask_full, _ = build_masked_batch(args, data, 'atv', False)
            model.eval()
            with torch.no_grad():
                model.test_condition = 'atv'
                teacher_hidden, teacher_out, *_ = model(input_full, mask_full, umask_full, False)
                teacher_hidden = getattr(model, 'last_pre_fusion_hidden', teacher_hidden).detach()
                teacher_out = teacher_out.detach()
            if was_training:
                model.train()
            model.test_condition = old_condition

            if align_enabled and getattr(args, 'condition_align_weight', 0.0) > 0:
                cardinality = len(condition_modalities(test_condition))
                margin = getattr(args, 'condition_align_delta0', 0.0) * (1.0 - cardinality / 3.0)
                align_loss = masked_hidden_margin_mse(
                    hidden,
                    teacher_hidden,
                    umask,
                    margin,
                    getattr(args, 'condition_distill_eps', 1e-8),
                )
                loss = loss + args.condition_align_weight * align_loss

            if (
                sem_enabled
                and getattr(args, 'condition_semantic_kl_weight', 0.0) > 0
                and dataset in ['IEMOCAPFOUR', 'IEMOCAPSIX']
            ):
                sem_loss = masked_token_kl(
                    out,
                    teacher_out,
                    umask,
                    1.0,
                    getattr(args, 'condition_distill_eps', 1e-8),
                )
                loss = loss + args.condition_semantic_kl_weight * sem_loss
            return loss

        def add_scrb_aux_loss(loss, loss_fn):
            if first_stage or not train or not getattr(args, 'scrb_aux_aib', False):
                return loss
            aux_out = getattr(model, 'last_scrb_aux_out', None)
            if aux_out is None:
                return loss

            aux_weight = getattr(args, 'scrb_aux_weight', 0.0)
            if aux_weight > 0:
                loss = loss + aux_weight * loss_fn(aux_out.view(-1, aux_out.size(2)), labels_, umask)

            jsd = None
            if dataset in ['IEMOCAPFOUR', 'IEMOCAPSIX']:
                jsd = masked_token_jsd(out, aux_out, umask, getattr(args, 'condition_distill_eps', 1e-8))
                cons_weight = getattr(args, 'scrb_consistency_weight', 0.0)
                if cons_weight > 0:
                    loss = loss + cons_weight * jsd

            rate_weight = getattr(args, 'scrb_rate_weight', 0.0)
            if rate_weight <= 0:
                return loss

            rates = getattr(model, 'last_scrb_rates', {})
            rate_terms = []
            eps = getattr(args, 'rc_eps', 1e-6)
            for modality in condition_modalities(test_condition):
                rate = rates.get(modality)
                cap = uni_caps.get(modality)
                if rate is None or cap is None:
                    continue
                cap_tensor = rate.new_tensor(max(float(cap), eps))
                violation = torch.relu(rate / (cap_tensor + eps) - 1.0)
                lam = rc_uni_lambdas.get(modality, getattr(args, 'rc_lambda_init', 1.0))
                rate_terms.append(lam * violation + 0.5 * getattr(args, 'scrb_rate_kappa', 0.1) * violation.pow(2))
            if not rate_terms:
                return loss

            sufficiency = out.new_tensor(1.0)
            if getattr(args, 'scrb_sufficiency_gate', False) and jsd is not None:
                sufficiency = torch.exp(-jsd.detach() / max(getattr(args, 'scrb_sufficiency_tau', 0.1), 1e-8))
            condition_scale = 1.0
            if getattr(args, 'condition_size_ib_scaling', False):
                condition_scale = (len(condition_modalities(test_condition)) / 3.0) ** getattr(args, 'condition_size_ib_zeta', 0.5)
            loss = loss + rate_weight * sufficiency * condition_scale * sum(rate_terms) / len(rate_terms)
            return loss

        def add_structural_regularization(loss):
            if not train or first_stage or not hasattr(model, 'structural_regularization'):
                return loss
            reg = model.structural_regularization()
            if torch.is_tensor(reg):
                loss = loss + reg
            return loss
        
        # Flatten sequence predictions for masked losses.
        lp_ = out.view(-1, out.size(2))
        lp_intra_a, lp_intra_t, lp_intra_v = outputs_intra_a.view(-1, outputs_intra_a.size(2)), outputs_intra_t.view(-1, outputs_intra_t.size(2)), outputs_intra_v.view(-1, outputs_intra_v.size(2))
        lp_inter_a, lp_inter_t, lp_inter_v = outputs_inter_a.view(-1, outputs_inter_a.size(2)), outputs_inter_t.view(-1, outputs_inter_t.size(2)), outputs_inter_v.view(-1, outputs_inter_v.size(2))
        lp_a, lp_t, lp_v = out_a.view(-1, out_a.size(2)), out_t.view(-1, out_t.size(2)), out_v.view(-1, out_v.size(2))
        lp_a_common, lp_t_common, lp_v_common = out_a_common.view(-1, out_a_common.size(2)), out_t_common.view(-1, out_t_common.size(2)), out_v_common.view(-1, out_v_common.size(2))
        labels_ = label.view(-1)
        task_loss = cls_loss if dataset in ['IEMOCAPFOUR', 'IEMOCAPSIX'] else reg_loss
        if dataset in ['CMUMOSI', 'CMUMOSEI', 'IEMOCAPFOUR', 'IEMOCAPSIX']:
            if first_stage:
                loss_intra_a = task_loss(lp_intra_a, labels_, umask)
                loss_intra_t = task_loss(lp_intra_t, labels_, umask)
                loss_intra_v = task_loss(lp_intra_v, labels_, umask)
                loss_inter_a = task_loss(lp_inter_a, labels_, umask)
                loss_inter_t = task_loss(lp_inter_t, labels_, umask)
                loss_inter_v = task_loss(lp_inter_v, labels_, umask)
                loss_a = task_loss(lp_a, labels_, umask) + ib_penalty(a_kls, ib_beta_a * first_ib_scale, uni_cap_a, args.aug_kappa_uni * first_ib_scale)
                loss_t = task_loss(lp_t, labels_, umask) + ib_penalty(t_kls, ib_beta_t * first_ib_scale, uni_cap_t, args.aug_kappa_uni * first_ib_scale)
                loss_v = task_loss(lp_v, labels_, umask) + ib_penalty(v_kls, ib_beta_v * first_ib_scale, uni_cap_v, args.aug_kappa_uni * first_ib_scale)
                loss_a_common = task_loss(lp_a_common, labels_, umask) + ib_penalty(a_kls_common, ib_beta_a * first_ib_scale, uni_cap_a, args.aug_kappa_uni * first_ib_scale)
                loss_t_common = task_loss(lp_t_common, labels_, umask) + ib_penalty(t_kls_common, ib_beta_t * first_ib_scale, uni_cap_t, args.aug_kappa_uni * first_ib_scale)
                loss_v_common = task_loss(lp_v_common, labels_, umask) + ib_penalty(v_kls_common, ib_beta_v * first_ib_scale, uni_cap_v, args.aug_kappa_uni * first_ib_scale)
                loss_joint = task_loss(lp_, labels_, umask)
            else:
                loss = task_loss(lp_, labels_, umask)
                loss = add_second_stage_decoder_loss(loss, task_loss)
                loss = add_condition_distill_loss(loss)
                loss = add_condition_alignment_loss(loss)
                loss = add_scrb_aux_loss(loss, task_loss)
                loss = add_DDIB_loss(loss, task_loss)
                if args.fusion_aib and (not getattr(args, 'DDIB_fusion', False)) and not skip_condition_ib:
                    loss = loss + second_stage_fusion_penalty(current_fusion_kl)
                if args.second_stage_uni_aib and (not getattr(args, 'DDIB_uni', False)) and not skip_condition_ib:
                    for modality in condition_modalities(test_condition):
                        if use_msd:
                            loss = loss + second_stage_uni_penalty(modality, uni_kls[modality][0])
                        if use_mcd:
                            loss = loss + second_stage_uni_penalty(modality, uni_kls[modality][1])
                loss = add_structural_regularization(loss)
        preds_a.append(lp_intra_a.data.cpu().numpy())
        preds_t.append(lp_intra_t.data.cpu().numpy())
        preds_v.append(lp_intra_v.data.cpu().numpy())
        preds_a_common.append(lp_inter_a.data.cpu().numpy())
        preds_t_common.append(lp_inter_t.data.cpu().numpy())
        preds_v_common.append(lp_inter_v.data.cpu().numpy())
        preds.append(lp_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        if train and first_stage:
            loss_as, loss_ts, loss_vs = None, None, None
            if use_msd:
                loss_as = add_loss(loss_as, loss_a + loss_intra_a)
                loss_ts = add_loss(loss_ts, loss_t + loss_intra_t)
                loss_vs = add_loss(loss_vs, loss_v + loss_intra_v)
            if use_mcd:
                loss_as = add_loss(loss_as, loss_a_common + loss_inter_a)
                loss_ts = add_loss(loss_ts, loss_t_common + loss_inter_t)
                loss_vs = add_loss(loss_vs, loss_v_common + loss_inter_v)
            if loss_as is not None:
                total_stage1_loss = loss_as + loss_ts + loss_vs
                joint_weight = float(getattr(args, 'stage1_joint_weight', 0.0))
                if joint_weight > 0:
                    total_stage1_loss = total_stage1_loss + joint_weight * loss_joint
                if args.prior_reg_lambda > 0 and hasattr(model, 'prior_regularization'):
                    total_stage1_loss = total_stage1_loss + model.prior_regularization() * args.prior_reg_lambda
                loss_value = total_stage1_loss.detach() / (3.0 + joint_weight)
                losses.append(float(loss_value.cpu().item()) * batch_weight)
                total_stage1_loss.backward()
                optimizer_step()
        if train and not first_stage:
            if args.prior_reg_lambda > 0 and hasattr(model, 'prior_regularization'):
                loss = loss + model.prior_regularization() * args.prior_reg_lambda
            losses.append(float(loss.detach().cpu().item()) * batch_weight)
            loss.backward()
            optimizer_step()

    assert preds!=[], f'Error: no dataset in dataloader'
    preds  = np.concatenate(preds)
    preds_a = np.concatenate(preds_a)
    preds_t = np.concatenate(preds_t)
    preds_v = np.concatenate(preds_v)
    preds_a_common = np.concatenate(preds_a_common)
    preds_t_common = np.concatenate(preds_t_common)
    preds_v_common = np.concatenate(preds_v_common)
    labels = np.concatenate(labels)
    masks  = np.concatenate(masks)
    avg_stats = {stat_name: stat_sum / max(stats_weight, 1.0) for stat_name, stat_sum in stats_sums.items()}

    avg_loss = round(np.sum(losses) / np.sum(masks), 4) if losses else 0.0
    if dataset in ['CMUMOSI', 'CMUMOSEI']:
        prediction = preds.reshape(-1)
        valid = (masks > 0) & (labels != 0)
        y_true = labels[valid] > 0
        y_pred = prediction[valid] > 0
        avg_accuracy = accuracy_score(y_true, y_pred)
        avg_fscore = f1_score(y_true, y_pred, average='weighted')
        mae = float(np.mean(np.abs(labels[masks > 0] - prediction[masks > 0])))
        ua = float(np.corrcoef(labels[masks > 0], prediction[masks > 0])[0, 1])
        branch_predictions = [preds_a, preds_t, preds_v, preds_a_common, preds_t_common, preds_v_common]
        branch_accs = [accuracy_score(y_true, branch.reshape(-1)[valid] > 0) for branch in branch_predictions]
        return mae, ua, avg_accuracy, avg_fscore, branch_accs, vidnames, avg_loss, avg_stats

    preds = np.argmax(preds, 1)
    preds_a = np.argmax(preds_a, 1)
    preds_t = np.argmax(preds_t, 1)
    preds_v = np.argmax(preds_v, 1)
    preds_a_common = np.argmax(preds_a_common, 1)
    preds_t_common = np.argmax(preds_t_common, 1)
    preds_v_common = np.argmax(preds_v_common, 1)
    avg_accuracy = accuracy_score(labels, preds, sample_weight=masks)
    avg_fscore = f1_score(labels, preds, sample_weight=masks, average='weighted')
    mae = 0
    ua = recall_score(labels, preds, sample_weight=masks, average='macro')
    avg_acc_a = accuracy_score(labels, preds_a, sample_weight=masks)
    avg_acc_t = accuracy_score(labels, preds_t, sample_weight=masks)
    avg_acc_v = accuracy_score(labels, preds_v, sample_weight=masks)
    avg_acc_a_common = accuracy_score(labels, preds_a_common, sample_weight=masks)
    avg_acc_t_common = accuracy_score(labels, preds_t_common, sample_weight=masks)
    avg_acc_v_common = accuracy_score(labels, preds_v_common, sample_weight=masks)
    return mae, ua, avg_accuracy, avg_fscore, [avg_acc_a, avg_acc_t, avg_acc_v, avg_acc_a_common, avg_acc_t_common, avg_acc_v_common], vidnames, avg_loss, avg_stats
