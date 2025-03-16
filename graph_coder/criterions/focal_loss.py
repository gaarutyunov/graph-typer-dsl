import math
import pathlib
from typing import Optional, Sequence

import torch
from dpu_utils.utils import RichPath
from fairseq import utils
from fairseq.criterions.fairseq_criterion import FairseqCriterion
from fairseq.logging import metrics
from torch import nn, Tensor
import torch.nn.functional as F


from fairseq import criterions


def focal_loss(
        x: Tensor,
        y: Tensor,
        alpha: Optional[Tensor] = None,
        gamma: float = 0.,
        reduction: str = 'mean',
        ignore_index: int = -100,
        device='cpu',
        dtype=torch.float32
) -> Tensor:
    """Function for FocalLoss.

    Implementation from: https://github.com/AdeelH/pytorch-multi-class-focal-loss

    Args:
        x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
        alpha (Sequence, optional): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore.
            Defaults to -100.
        device (str, optional): Device to move alpha to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast alpha to.
            Defaults to torch.float32.

    Returns:
        Loss
    """
    if alpha is not None:
        alpha = alpha.to(device=device, dtype=dtype)

    if x.ndim > 2:
        # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
        c = x.shape[1]
        x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
        # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
        y = y.view(-1)

    unignored_mask = y != ignore_index
    y = y[unignored_mask]
    x = x[unignored_mask]

    # compute weighted cross entropy term: -alpha * log(pt)
    # (alpha is already part of self.nll_loss)
    log_p = F.log_softmax(x, dim=-1)
    ce = F.nll_loss(log_p, y, weight=alpha, ignore_index=ignore_index, reduction="none")

    # get true class column from each row
    all_rows = torch.arange(len(x))
    log_pt = log_p[all_rows, y]

    # compute focal term: (1 - pt)^gamma
    pt = log_pt.exp()
    focal_term = (1 - pt) ** gamma

    # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
    loss = focal_term * ce

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

    return loss


@criterions.register_criterion("focal_loss")
class FocalLossCriterion(FairseqCriterion):
    def __init__(self, task, counter_path, sizes_path, gamma, ignore_index):
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

        self.gamma = gamma
        self.ignore_index = ignore_index

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--counter-path", type=lambda s: str(pathlib.Path(s).expanduser()), default=None)
        parser.add_argument("--sizes-path", type=lambda s: str(pathlib.Path(s).expanduser()), default=None)
        parser.add_argument("--gamma", type=float, default=2.0)
        parser.add_argument("--ignore-index", type=int, default=-100)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        masked_tokens = sample.get("masked_tokens", None)
        net_output = model(batched_data=sample, masked_tokens=masked_tokens)
        net_output = model.embed_out(net_output) + model.lm_output_learned_bias
        targets = model.get_targets(sample, net_output)
        loss = focal_loss(
            net_output.view(-1, net_output.size(-1)),
            targets.view(-1),
            alpha=self.weight,
            gamma=self.gamma,
            reduction="sum" if reduce else None,
            ignore_index=self.ignore_index,
            device=net_output.device,
            dtype=net_output.dtype
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
