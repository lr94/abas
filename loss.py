import numpy as np
import torch
import torch.nn as nn


def entropy_loss(logits):
    p_softmax = torch.nn.functional.softmax(logits, dim=1)
    mask = p_softmax.ge(0.000001)  # greater or equal to; to avoid explosion when computing logarithm
    mask_out = torch.masked_select(p_softmax, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(p_softmax.size(0))


def DANN_loss(ad_out):
    """
    Domain-adversarial training of neural networks, Ganin et al., 2016
    """
    dev = ad_out.device
    bs = ad_out.size(0)
    assert bs % 2 == 0
    bs //= 2

    domain_labels = torch.cat((
        torch.zeros(size=(bs,), device=dev, dtype=torch.long),
        torch.ones(size=(bs,), device=dev, dtype=torch.long)
    ), dim=0)

    return nn.CrossEntropyLoss()(ad_out, domain_labels)


def create_matrix(n, device):
    """
    :param n: matrix size (class num)
    :param device:
    :return a matrix with torch.tensor type:
    for example n=3:
    1     -1/2  -1/2
    -1/2    1   -1/2
    -1/2  -1/2    1
    """

    eye = torch.eye(n, device=device)
    return eye - (torch.ones((n, n), device=device) - eye) / (n - 1)


def ALDA_loss(ad_out_score, labels_source, softmax_out, threshold=0.9):
    """
    :param ad_out_score: the discriminator output (N, C, H, W)
    :param labels_source: the source ground truth (N, H, W)
    :param softmax_out: the model prediction probability (N, C, H, W)
    :return:
    adv_loss: adversarial learning loss
    reg_loss: regularization term for the discriminator
    correct_loss: corrected self-training loss

    Adversarial-Learned Loss for Domain Adaptation, Chen et al., 2020
    https://github.com/ZJULearning/ALDA
    """
    ad_out = torch.sigmoid(ad_out_score)
    dev = ad_out.device

    batch_size = ad_out.size(0) // 2
    class_num = ad_out.size(1)

    labels_source_mask = torch.zeros(batch_size, class_num).to(dev).scatter_(1, labels_source.unsqueeze(1), 1)
    probs_source = softmax_out[:batch_size].detach()
    probs_target = softmax_out[batch_size:].detach()
    maxpred, argpred = torch.max(probs_source, dim=1)
    preds_source_mask = torch.zeros(batch_size, class_num).to(dev).scatter_(1, argpred.unsqueeze(1), 1)
    maxpred, argpred = torch.max(probs_target, dim=1)
    preds_target_mask = torch.zeros(batch_size, class_num).to(dev).scatter_(1, argpred.unsqueeze(1), 1)

    # filter out those low confidence samples
    target_mask = (maxpred > threshold)
    preds_target_mask = torch.where(target_mask.unsqueeze(1), preds_target_mask, torch.zeros(1).to(dev))
    # construct the confusion matrix from ad_out. See the paper for more details.
    confusion_matrix = create_matrix(class_num, device=dev)
    ant_eye = (1 - torch.eye(class_num)).to(dev).unsqueeze(0)
    confusion_matrix = ant_eye / (class_num - 1) + torch.mul(confusion_matrix.unsqueeze(0), ad_out.unsqueeze(
        1))  # (2*batch_size, class_num, class_num)
    preds_mask = torch.cat([preds_source_mask, preds_target_mask], dim=0)  # labels_source_mask
    loss_pred = torch.mul(confusion_matrix, preds_mask.unsqueeze(1)).sum(dim=2)
    # different correction targets for different domains
    loss_target = (1 - preds_target_mask) / (class_num - 1)
    loss_target = torch.cat([labels_source_mask, loss_target], dim=0)
    if not ((loss_pred >= 0).all() and (loss_pred <= 1).all()):
        nan = float('nan')
        nan = torch.tensor(nan, device=dev, requires_grad=True)
        return nan, nan.clone(), nan.clone()
    mask = torch.cat([(maxpred >= 0), target_mask], dim=0)
    adv_loss = nn.BCELoss(reduction='none')(loss_pred, loss_target)[mask]
    adv_loss = torch.sum(adv_loss) / mask.float().sum()

    # reg_loss
    reg_loss = nn.CrossEntropyLoss()(ad_out_score[:batch_size], labels_source)

    # corrected target loss function
    target_probs = 1.0 * softmax_out[batch_size:]
    correct_target = torch.mul(confusion_matrix.detach()[batch_size:], preds_target_mask.unsqueeze(1)).sum(dim=2)
    correct_loss = -torch.mul(target_probs, correct_target)
    correct_loss = torch.mean(correct_loss[target_mask])
    return adv_loss, reg_loss, correct_loss
