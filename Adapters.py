import torch
import torch.nn.functional as F


class DDIBConditionMixin:
    def _modality_prior(self, index):
        if not self.modality_prior:
            return None, None
        return self.modality_prior_mu[index], torch.exp(self.modality_prior_logstd[index])

    def _condition_prior(self):
        if not (self.fusion_aib and self.condition_prior):
            return None, None
        index = self.condition_to_idx.get(self.test_condition, self.condition_to_idx['atv'])
        return self.condition_prior_mu[index], torch.exp(self.condition_prior_logstd[index])

    def _residual_aib(self, original, bottlenecked):
        alpha = self.uni_aib_residual_alpha
        if alpha >= 1.0:
            return bottlenecked
        if alpha <= 0.0:
            return original
        return original + alpha * (bottlenecked - original)

    def _residual_fusion_aib(self, original, bottlenecked):
        alpha = self.DDIB_fusion_residual_alpha * self.DDIB_fusion_alpha_scale
        self.last_fusion_residual_gate = None
        if alpha <= 0.0:
            return original
        delta_modes = {'delta_projected_add', 'gated_delta_projected_add', 'gated_delta_lowrank_add'}
        gated_modes = {'gated_projected_add', 'gated_lowrank_add', 'gated_delta_projected_add', 'gated_delta_lowrank_add'}
        lowrank_modes = {'gated_lowrank_add', 'gated_delta_lowrank_add'}
        residual_source = bottlenecked - original if self.DDIB_fusion_residual_mode in delta_modes else bottlenecked
        if self.DDIB_fusion_residual_mode in gated_modes:
            gate_input = torch.cat([original, bottlenecked], dim=-1)
            gate = torch.sigmoid(self.fusion_residual_gate(self.fusion_gate_norm(gate_input)))
            self.last_fusion_residual_gate = gate.detach().mean()
            residual_input = self.fusion_residual_norm(residual_source)
            if self.DDIB_fusion_residual_mode in lowrank_modes:
                residual = self.fusion_residual_up(F.gelu(self.fusion_residual_down(residual_input)))
            else:
                residual = self.fusion_residual_proj(residual_input)
            return original + alpha * gate * residual
        if self.DDIB_fusion_residual_mode in {'projected_add', 'delta_projected_add'}:
            return original + alpha * self.fusion_residual_proj(self.fusion_residual_norm(residual_source))
        if self.DDIB_fusion_residual_mode == 'add':
            return original + alpha * bottlenecked
        if alpha >= 1.0:
            return bottlenecked
        return original + alpha * (bottlenecked - original)

    def _condition_index(self):
        return self.condition_to_idx.get(self.test_condition, self.condition_to_idx['atv'])

    def _condition_modalities(self, condition=None):
        condition = condition or self.test_condition
        if condition == 'all':
            condition = 'atv'
        return [modality for modality in ['a', 't', 'v'] if modality in condition]

    def _condition_embedding(self, condition=None):
        condition = condition or self.test_condition
        members = self.condition_members.get(condition, self.condition_members['atv'])
        member_idx = torch.tensor(members, dtype=torch.long, device=self.device)
        modality_emb = self.condition_modality_embedding(member_idx).mean(dim=0)
        card_idx = torch.tensor([len(members)], dtype=torch.long, device=self.device)
        return modality_emb + self.condition_cardinality_embedding(card_idx).squeeze(0)

    def _apply_condition_denoise_gates(self, raw_reps, specific_reps, common_reps, first_stage):
        if first_stage or not self.condition_denoise_gates:
            return specific_reps, common_reps
        gates = self.condition_gate_values[self._condition_index()]
        gated_specific = []
        gated_common = []
        for idx, (raw, specific, common) in enumerate(zip(raw_reps, specific_reps, common_reps)):
            gated_specific.append(raw + gates[idx, 0] * (specific - raw))
            gated_common.append(raw + gates[idx, 1] * (common - raw))
        return gated_specific, gated_common

    def _apply_condition_adapter(self, hidden, first_stage):
        if first_stage or not self.condition_adapter:
            return hidden
        idx = self._condition_index()
        return hidden + self.condition_adapter_up[idx](self.condition_adapter_down[idx](hidden))

    def _apply_condition_generated_adapter(self, hidden, first_stage):
        if first_stage or not self.condition_generated_adapter:
            return hidden
        condition_emb = self._condition_embedding()
        weights = torch.softmax(self.adapter_bank_pi(condition_emb), dim=-1)
        eta = torch.sigmoid(self.adapter_bank_eta(condition_emb)).view(1, 1, 1)
        delta = hidden.new_zeros(hidden.shape)
        for weight, down, up in zip(weights, self.adapter_bank_down, self.adapter_bank_up):
            delta = delta + weight * up(down(hidden))
        return hidden + eta * delta

    def _scrb_context(self, modality, condition=None):
        condition = condition or self.test_condition
        members = self.condition_members.get(condition, self.condition_members['atv'])
        device = self.scrb_modality_embedding.weight.device
        member_idx = torch.tensor(members, dtype=torch.long, device=device)
        condition_emb = self.scrb_modality_embedding(member_idx).mean(dim=0)
        card_idx = torch.tensor([len(members)], dtype=torch.long, device=device)
        condition_emb = condition_emb + self.scrb_cardinality_embedding(card_idx).squeeze(0)
        modality_idx = torch.tensor(self.modality_to_idx[modality], dtype=torch.long, device=device)
        return self.scrb_context_proj(condition_emb + self.scrb_modality_embedding(modality_idx))

    def structural_regularization(self):
        reg = None
        if self.condition_denoise_gates and self.condition_gate_reg_lambda > 0:
            term = (self.condition_gate_values - 1.0).pow(2).mean() * self.condition_gate_reg_lambda
            reg = term if reg is None else reg + term
        if self.condition_adapter and self.condition_adapter_reg_lambda > 0:
            terms = []
            for down, up in zip(self.condition_adapter_down, self.condition_adapter_up):
                terms.append(down.weight.pow(2).mean() + up.weight.pow(2).mean())
            if terms:
                term = sum(terms) / len(terms) * self.condition_adapter_reg_lambda
                reg = term if reg is None else reg + term
        if self.condition_generated_adapter and self.adapter_bank_reg_lambda > 0:
            terms = []
            for down, up in zip(self.adapter_bank_down, self.adapter_bank_up):
                terms.append(down.weight.pow(2).mean() + up.weight.pow(2).mean())
            etas = []
            for condition in self.condition_to_idx:
                condition_emb = self._condition_embedding(condition)
                etas.append(torch.sigmoid(self.adapter_bank_eta(condition_emb)).pow(2).mean())
            if terms:
                term = (sum(terms) / len(terms) + sum(etas) / len(etas)) * self.adapter_bank_reg_lambda
                reg = term if reg is None else reg + term
        if (
            self.fusion_aib
            and self.DDIB_fusion
            and self.DDIB_fusion_residual_mode in {'gated_projected_add', 'gated_lowrank_add'}
            and self.DDIB_fusion_gate_reg_lambda > 0
        ):
            gate = torch.sigmoid(self.fusion_residual_gate.bias).pow(2).mean()
            term = gate * self.DDIB_fusion_gate_reg_lambda
            reg = term if reg is None else reg + term
        if reg is None:
            return torch.tensor(0.0, device=self.device)
        return reg

    def prior_regularization(self):
        reg = None
        if self.modality_prior:
            std = torch.exp(self.modality_prior_logstd)
            term = 0.5 * (self.modality_prior_mu.pow(2) + std.pow(2) - 2 * self.modality_prior_logstd - 1).mean()
            reg = term if reg is None else reg + term
        if self.fusion_aib and self.condition_prior:
            std = torch.exp(self.condition_prior_logstd)
            term = 0.5 * (self.condition_prior_mu.pow(2) + std.pow(2) - 2 * self.condition_prior_logstd - 1).mean()
            reg = term if reg is None else reg + term
        if reg is None:
            return torch.tensor(0.0, device=self.device)
        return reg
