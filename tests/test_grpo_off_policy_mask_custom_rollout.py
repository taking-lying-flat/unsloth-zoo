# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""A custom rollout_func without vLLM still feeds its sampling logprobs to the off-policy mask.

TRL's GRPOTrainer._compute_loss passes inputs["sampling_per_token_logps"] to get_off_policy_mask
whenever the batch carries it, whatever the generation backend.
"""

import collections
import types

import pytest
import torch

import unsloth_zoo.rl_replacements as rr

# Rollout logprobs arrive right padded (TRL) and must land in the left-packed completion columns.
_IDS = torch.tensor([[0, 0, 1, 2, 3, 4, 5], [1, 2, 6, 7, 3, 4, 6], [0, 1, 2, 3, 5, 0, 0]])
_MASK = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 0, 0]])


def _trl_get_off_policy_mask(advantages, per_token_logps, sampling_per_token_logps, mask, off_policy_threshold):
    # trl/trainer/grpo_trainer.py GRPOTrainer.get_off_policy_mask (main, c6eac19).
    kl_div = sampling_per_token_logps - per_token_logps.detach()
    kl_div = torch.nan_to_num(kl_div, nan = 0.0)
    seq_kl_sum = (kl_div * mask).sum(dim = 1, keepdim = True)
    avg_seq_kl = seq_kl_sum / mask.sum(dim = 1, keepdim = True).clamp(min = 1.0)
    return ((advantages >= 0) | (avg_seq_kl <= off_policy_threshold)).to(dtype = mask.dtype)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(8, 4)
        self.head = torch.nn.Linear(4, 8, bias = False)
        self.device = torch.device("cpu")

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids = None, attention_mask = None, logits_to_keep = None, **kwargs):
        hidden = self.emb(input_ids)
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        return types.SimpleNamespace(logits = hidden)


def _drive(monkeypatch, threshold, use_vllm):
    monkeypatch.setenv("UNSLOTH_GRPO_SEQ_PACKING", "0")
    monkeypatch.setenv("UNSLOTH_GRPO_PREFIX_GROUPER", "0")
    monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "0")
    seen = {}

    def get_off_policy_mask(**kw):
        out = _trl_get_off_policy_mask(**kw)
        seen.update(sampling = kw["sampling_per_token_logps"].detach().clone(), keep = out.clone(), mask = kw["mask"].clone())
        return out

    model = _Model()
    model.train(True)
    trainer = types.SimpleNamespace(
        args = types.SimpleNamespace(
            unsloth_grpo_mini_batch = 1, unsloth_logit_chunk_multiplier = 1, off_policy_mask_threshold = threshold,
            vllm_importance_sampling_mode = "token_truncate", vllm_importance_sampling_cap = 2.0,
        ),
        processing_class = types.SimpleNamespace(pad_token_id = 0),
        model = model,
        accelerator = types.SimpleNamespace(
            unwrap_model = lambda m, keep_fp32_wrapper = False: m, scaler = None, gather = lambda t: t,
        ),
        control = types.SimpleNamespace(should_evaluate = False),
        _metrics = {"train": collections.defaultdict(list), "eval": collections.defaultdict(list)},
        use_vllm = use_vllm,
        vllm_importance_sampling_correction = False,
        _autocast_dtype = None,
        beta = 0.0,
        loss_type = "dapo",
        get_off_policy_mask = get_off_policy_mask,
    )
    # The sampler put probability 1 on every token: KL(pi_sample || pi_theta) = -log pi_theta > threshold.
    sampling = torch.zeros(3, 3)
    rr.grpo_accumulated_loss(
        trainer, _IDS, (_IDS != 0).long(), 3, _MASK, -torch.ones(3), None, None,
        loss_type = "dapo", sampling_per_token_logps = sampling, num_items_in_batch = int(_MASK.sum()),
    )
    return seen, sampling


@pytest.mark.parametrize("use_vllm", [False, True])
def test_rollout_logps_reach_the_off_policy_mask(monkeypatch, use_vllm):
    seen, sampling = _drive(monkeypatch, 0.5, use_vllm)
    torch.testing.assert_close(seen["sampling"], rr.align_logprobs_with_mask(sampling, seen["mask"]))
    # Every row has a negative advantage and a large rollout gap, so TRL drops all of them.
    assert seen["keep"].sum().item() == 0


def test_no_threshold_never_builds_the_mask(monkeypatch):
    seen, _ = _drive(monkeypatch, None, False)
    assert seen == {}
