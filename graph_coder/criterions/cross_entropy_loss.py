# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import pathlib
from typing import Optional

from dpu_utils.utils import RichPath
from fairseq import utils
from fairseq.criterions.fairseq_criterion import FairseqCriterion
from fairseq.logging import metrics

import torch.nn.functional as F
from torch import Tensor

from fairseq import criterions

@criterions.register_criterion("cross_entropy_loss")
class CrossEntropyLoss(FairseqCriterion):
    def __init__(self, task, counter_path, sizes_path):
        super().__init__(task)
        total = 0
        self.weight: Optional[Tensor] = None

        if sizes_path is not None:
            sizes = RichPath.create(sizes_path).read_by_file_suffix()

            total = (sizes <= task.cfg.max_tokens).sum().item()

        if counter_path is not None:
            counter = RichPath.create(counter_path).read_by_file_suffix()
            if total != 0:
                self.weight = total / (task.cfg.num_classes * counter)

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--counter-path", type=lambda s: str(pathlib.Path(s).expanduser()), default=None)
        parser.add_argument("--sizes-path", type=lambda s: str(pathlib.Path(s).expanduser()), default=None)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        masked_tokens = sample.get("masked_tokens", None)
        net_output = model(batched_data=sample, masked_tokens=masked_tokens)
        net_output = model.encoder.embed_out(net_output) + model.encoder.lm_output_learned_bias
        if self.weights is not None:
            self.weights = self.weights.to(net_output.device)
        targets = model.get_targets(sample, net_output)
        loss = F.cross_entropy(
            net_output.view(-1, net_output.size(-1)),
            targets.view(-1),
            reduction="sum" if reduce else "none",
            weight=self.weights,
            ignore_index=self.padding_idx,
        )
        ntokens = masked_tokens.sum().item()
        logging_output = {
            "loss": loss.item(),
            "sample_size": ntokens,
            "ntokens": ntokens,
            "nsentences": net_output.size(0),
        }
        return loss, ntokens, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
