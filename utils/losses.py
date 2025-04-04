import torch
import torch.nn.functional as F


def sup_con(features, temperature=0.07, labels=None, mask=None):
    features_norm = F.normalize(features, p=2, dim=1)
    batch_size = features_norm.shape[0]
    device = features_norm.device

    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(features_norm, features_norm.T),
        temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    exp_logits = torch.exp(logits)

    logits_mask = torch.ones_like(mask).to(device) - torch.eye(batch_size).to(device)
    positives_mask = mask * logits_mask
    negatives_mask = 1. - mask

    num_positives_per_row = torch.sum(positives_mask, axis=1)
    denominator = torch.sum(
        exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
        exp_logits * positives_mask, axis=1, keepdims=True)

    log_probs = logits - torch.log(denominator)
    if torch.any(torch.isnan(log_probs)):
        raise ValueError("Log_prob has nan!")

    log_probs = torch.sum(
        log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[
                    num_positives_per_row > 0]

    # loss
    loss = -log_probs
    loss = loss.mean()
    return loss