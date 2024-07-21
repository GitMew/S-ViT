from dataclasses import dataclass

import torch
from torch import Tensor

from transformers.models.vit import ViTModel, ViTConfig, ViTPreTrainedModel
# from transformers.models.vit_mae import ViTMAEPreTrainedModel, ViTMAEConfig, ViTMAEModel  # ViT-MAE is only a training strategy. They are architecturally identical to ViTs except for the masking in the encoder and the head during pre-training. HuggingFace's ViTMAEModel is actually geared only towards pre-training since it ALWAYS masks.
from sentence_transformers.util import pairwise_cos_sim


class ViTForSimilarity(ViTPreTrainedModel):  # There is no RoBERTa for this task: https://stackoverflow.com/questions/78432122/roberta-for-sentence-similarity

    def __init__(self, config: ViTConfig):
        super().__init__(config)
        self.vit      = ViTModel(config)
        self.pooler   = lambda inp: torch.mean(inp, dim=1)
        self.loss_fct = CoSENTLossFromSimilarities()

    def forward(
        self,
        first_images: Tensor,  # B x b x pic
        second_images: Tensor,  # B x b x pic
        labels: Tensor=None,  # B x b
        return_dict: bool=False,
        **kwargs
    ):
        # First flatten the minibatches so we can process them at once
        print(first_images.size())

        B = first_images.size(0)
        b = first_images.size(1)
        first_images  = torch.flatten(first_images,  start_dim=0, end_dim=1)
        second_images = torch.flatten(second_images, start_dim=0, end_dim=1)

        # Process both images with the same network (siamese)
        print(first_images.size())

        first_hidden_states  = self.vit(first_images,  return_dict=True).last_hidden_state  # (B * b) x patches(pic) x H
        second_hidden_states = self.vit(second_images, return_dict=True).last_hidden_state  # (B * b) x patches(pic) x H

        print(first_hidden_states.size())

        first_embeddings  = self.pooler(first_hidden_states)   # (B * b) x H
        second_embeddings = self.pooler(second_hidden_states)  # (B * b) x H

        print(first_embeddings.size())

        # Compare first and second
        similarities = pairwise_cos_sim(first_embeddings, second_embeddings)  # These are both lists of length B*b. The result is again a list of length B*b comparing embedding i in list 1 with embedding i in list 2. We don't want to compare across pairs (a B*b x B*b matrix), only within one pair.

        print(similarities.size())

        # Loss
        loss = torch.zeros(1)
        if labels is not None:
            for example_idx in range(B):
                print(similarities[example_idx*b:(example_idx+1)*b])
                loss += self.loss_fct(similarities[example_idx*b:(example_idx+1)*b], labels[example_idx,:])
            loss /= B

        if return_dict:
            return ViTMAEForSimilarityOutput(
                loss=loss
            )
        else:
            return (loss,)


@dataclass
class ViTMAEForSimilarityOutput:
    loss: Tensor


class CoSENTLossFromSimilarities:
    """
    According to the documentation of sentence_transformers, CosineSimilarityLoss is apparently antiquated and you should
    instead use a more general batch version called CoSENTLoss which, rather than computing |cos(s1,s2) - y| with y
    the target (you usually you have a dataset scoring relevance from 0 to 5 and this is rescaled to a cosine of  0 to 1,
    where we ignore all the negative cosines since that means "related but opposite"), you now compute
        L = log(1 + exp(cos(s1,s2) - cos(s3,s4)) + exp(...) + ... + exp(...))
    for a batch of pairs {(s1,s2), (s3,s4), ...}. The exponentials are every pair of pairs where the first pair has a
    lower expected similarity y than the second pair. That means ideally, you get:
        L = log(1 + exp(0 - 1) + exp(0 - 1) + ... + exp(0 - 1)) = log(1 + small + small + small + small) ~ log(1) = 0.

    We reimplement it because sentence_transformers.losses.CoSENTLoss expects full sentences as input, and hence the loss
    itself runs the model, which means it assumes what the input is, assumes what the output of the model looks like,
    and so on. Those are not responsibilities of the loss. The loss should simply take predictions and labels.
    """

    def __init__(self, temperature: float=1/20):
        self.scale = 1/temperature

    def __call__(self, similarity_predictions: Tensor, labels: Tensor) -> Tensor:
        """
        :param similarity_predictions: List of B numbers, one for each pair of sentences in the batch.
        :param labels: List of B desired similarities, one for each pair of sentences in the batch.
        """
        similarity_predictions             = similarity_predictions * self.scale
        similarity_predictions_differences = similarity_predictions[:, None] - similarity_predictions[None, :]  # Like an outer product, compare everything with everything.

        # Label matrix indicating which pairs are relevant
        labels = labels[:, None] < labels[None, :]
        labels = labels.float()

        # Mask out irrelevant pairs so they are negligible after exp()
        similarity_predictions_differences = similarity_predictions_differences - (1 - labels) * 1e12

        # The CoSENT loss looks like log(1 + sum(exp)). We turn it into log(sum(exp)) by adding a 0 to the sequence, since exp(0) = 1.
        similarity_predictions_differences = torch.cat((torch.zeros(1).to(similarity_predictions_differences.device), similarity_predictions_differences.view(-1)), dim=0)
        loss = torch.logsumexp(similarity_predictions_differences, dim=0)

        print("CoSENT loss:", loss)

        return loss
