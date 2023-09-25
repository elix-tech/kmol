import torch
import torch.nn.functional as F
import numpy as np


def prepare_edl_classification_output(output):
    """
    Prepares output for edl classification.
    """
    true_logits, false_logits = torch.chunk(output, 2, dim=-1)
    true_logits = torch.unsqueeze(true_logits, dim=-1)
    false_logits = torch.unsqueeze(false_logits, dim=-1)
    output = torch.cat((true_logits, false_logits), dim=-1)

    evidence = F.softplus(output)
    alpha = evidence + 1

    return output, alpha

def prepare_edl_classification_target(target):
    """
    Prepares target for edl classification
    """
    target = torch.unsqueeze(target, dim=-1)
    target = torch.cat((target, 1 - target), -1)

    return target


def edl_classification(output, target, epoch=0, loss_annealing=True):
    """
    Computes the EDL loss for classification. The use of EDL with the current implementation is suitable for multitask and can used instead of BCE
    """
    mask = torch.isnan(target)
    target[mask] = 0.0

    output, alpha = prepare_edl_classification_output(output)
    target = prepare_edl_classification_target(target)

    loss = torch.mean(edl_loss(torch.log, target, alpha, epoch, loss_annealing))
    return loss


def edl_classification_masked(output, target, epoch=0, loss_annealing=True):
    mask = target == target
    labels_copy = target.clone()
    labels_copy[~mask] = 0

    output, alpha = prepare_edl_classification_output(output)
    target = prepare_edl_classification_target(target)

    loss = edl_loss(torch.log, target, alpha, epoch, loss_annealing)
    loss = loss.squeeze()
    mask = mask.squeeze()
    loss *= mask

    labels_weights = ((labels_copy * 5) + 1).squeeze()
    loss *= labels_weights
    loss = loss.sum(dim=0) / (mask.sum(dim=0) + 1e-8)

    return loss.mean()


def edl_loss(func, y, alpha, epoch, annealing=True):
    S = torch.sum(alpha, dim=-1, keepdim=True)
    A = torch.sum(y * (func(S) - func(alpha)), dim=-1, keepdim=True)

    kl_alpha = (alpha - 1) * (1 - y) + 1
    annealing_lambda = 1.0
    if annealing:
        annealing_lambda = min(1.0, epoch/10.0)
    kl_div = annealing_lambda * kl_divergence(kl_alpha)

    return A + kl_div


def kl_divergence(alpha):
    ones = torch.ones([1, 2], dtype=torch.float32, device=alpha.get_device())
    sum_alpha = torch.sum(alpha, dim=-1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
        + torch.lgamma(ones).sum(dim=-1, keepdim=True)
        - torch.lgamma(ones.sum(dim=-1, keepdim=True))
    )
    second_term = (alpha - ones).mul(torch.digamma(alpha) - torch.digamma(sum_alpha)).sum(dim=-1, keepdim=True)
    kl = first_term + second_term
    return kl


def nig_nll(y, gamma, v, alpha, beta, reduce=True):
    twoBlambda = 2 * beta * (v + 1)

    nll = (
        0.5 * torch.log(np.pi / v)
        - alpha * torch.log(twoBlambda)
        + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + twoBlambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    return torch.mean(nll) if reduce else nll


def kl_nig(mu1, v1, a1, b1, mu2, v2, a2, b2):
    KL = (
        0.5 * (a1 - 1) / b1 * (v2 * torch.square(mu2 - mu1))
        + 0.5 * v2 / v1
        - 0.5 * torch.log(torch.abs(v2) / torch.abs(v1))
        - 0.5
        + a2 * torch.log(b1 / b2)
        - (torch.lgamma(a1) - torch.lgamma(a2))
        + (a1 - a2) * torch.digamma(a1)
        - (b1 - b2) * a1 / b1
    )
    return KL


def nig_reg(y, gamma, v, alpha, beta, omega=0.01, reduce=True, kl=False):
    error = (y - gamma).abs()

    if kl:
        kl = kl_nig(gamma, v, alpha, beta, gamma, omega, 1 + omega, beta)
        reg = error * kl
    else:
        evi = 2 * v + alpha
        reg = error * evi

    return torch.mean(reg) if reduce else reg


def edl_regression(evidential_output, y_true, coeff=1.0, **kwargs):
    evidential_output = torch.relu(evidential_output)

    targets = y_true.view(-1)
    gamma = evidential_output[:, 0].view(-1)
    v = evidential_output[:, 1].view(-1)
    alpha = evidential_output[:, 2].view(-1)
    beta = evidential_output[:, 3].view(-1)

    machine_epsilon = torch.tensor(np.finfo(np.float32).eps).to(device=evidential_output.device)
    safe_v = torch.max(machine_epsilon, v)
    safe_alpha = torch.max(machine_epsilon, alpha)
    safe_beta = torch.max(machine_epsilon, beta)

    loss_nll = nig_nll(targets, gamma, safe_v, safe_alpha, safe_beta)
    loss_reg = nig_reg(targets, gamma, safe_v, safe_alpha, safe_beta)

    return loss_nll + coeff * loss_reg
