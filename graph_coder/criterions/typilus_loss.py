import torch
import torch.nn.functional as F

from fairseq.criterions.fairseq_criterion import FairseqCriterion

from fairseq import criterions


@criterions.register_criterion("typilus_loss")
class TypilusLoss(FairseqCriterion):
    def __init__(self, 
                 task, 
                 margin: float, 
                 class_loss: bool, 
                 space_loss: bool, 
                 class_lambda: float,
                 epsilon: float,
                 space_lambda: float):
        super().__init__(task)
        self.margin = margin
        self.class_loss = class_loss
        self.space_loss = space_loss
        self.class_lambda = class_lambda
        self.epsilon = epsilon
        self.space_lambda = space_lambda
        assert self.class_loss or self.space_loss, "At least one of class_loss or space_loss must be True"

    @classmethod
    def add_args(cls, parser):
        """Adds margin argument, an integer"""
        parser.add_argument("--margin", type=float, default=2.)
        parser.add_argument("--class-loss", type=bool, default=True)
        parser.add_argument("--space-loss", type=bool, default=True)
        parser.add_argument("--class-lambda", type=float, default=1.)
        parser.add_argument("--epsilon", type=float, default=1e-10)
        parser.add_argument("--space-lambda", type=float, default=3000.)

    def forward(self, model, sample, reduce=True):
        target_representations = model(batched_data=sample)

        loss = 0

        if self.space_loss:
            loss += self.triplet_loss(
                target_representations,
                sample["y"],
            )

        if self.class_loss:
            loss += self.class_lambda * self.cross_entropy_loss(
                target_representations,
                sample["y"],
                model.encoder
            )

        return loss, sample["y"].size(1), {
            "loss": loss.item(),
        }

    def cross_entropy_loss(self, target_representations: torch.Tensor, labels: torch.Tensor, model):
        target_logits = model.embed_out(target_representations) + model.lm_output_learned_bias
        target_probs = F.softmax(target_logits, dim=-1)

        loss = F.cross_entropy(
            target_probs.view(-1, target_probs.size(-1)),
            labels.view(-1),
            reduction='mean'
        )

        return loss

    def triplet_loss(self, target_representations: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        has_values = labels != -100
        labels = labels[has_values]
        target_representations = target_representations[has_values]

        typed_annotation_pairs_are_equal = labels[:, None] == labels[None, :]
        distances = torch.cdist(target_representations, target_representations, p=1)

        max_positive_distance = torch.max(distances * typed_annotation_pairs_are_equal, dim=-1)[0]
        neg_dist_filter = distances <= (max_positive_distance.unsqueeze(-1) + self.margin)
        pos_mask = typed_annotation_pairs_are_equal + torch.eye(distances.size(0))
        neg_dist_filter = neg_dist_filter.float() * (1 - pos_mask)
        mean_negative_distances = torch.sum(distances * neg_dist_filter, dim=-1) / (torch.sum(neg_dist_filter, dim=-1) + self.epsilon)
        min_negative_distance = torch.min(distances + pos_mask * self.space_lambda, dim=-1)[0]
        pos_dist_filter = (distances >= (min_negative_distance.unsqueeze(-1) - self.margin)).float() * typed_annotation_pairs_are_equal
        mean_positive_distances = torch.sum(distances * pos_dist_filter, dim=-1) / (torch.sum(pos_dist_filter, dim=-1) + self.epsilon)

        triplet_loss = 0.5 * F.relu(mean_positive_distances - min_negative_distance + self.margin)
        triplet_loss += 0.5 * F.relu(max_positive_distance - mean_negative_distances + self.margin)

        return torch.mean(triplet_loss)