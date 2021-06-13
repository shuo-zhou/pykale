"""Commonly used losses, from domain adaptation package
https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/models/losses.py
"""

import torch
import torch.nn as nn
from torch.autograd import grad
from torch.nn import functional as F


def cross_entropy_logits(linear_output, label, weights=None):
    """Computes cross entropy with logits

    Examples:
        See DANN, WDGRL, and MMD trainers in kale.pipeline.domain_adapter
    """

    class_output = F.log_softmax(linear_output, dim=1)
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    correct = y_hat.eq(label.view(label.size(0)).type_as(y_hat))
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return loss, correct


def entropy_logits(linear_output):
    """Computes entropy logits in CDAN with entropy conditioning (CDAN+E)

    Examples:
        See CDANtrainer in kale.pipeline.domain_adapter
    """
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


def entropy_logits_loss(linear_output):
    """Computes entropy logits loss in semi-supervised or few-shot domain adapatation

    Examples:
        See FewShotDANNtrainer in kale.pipeline.domain_adapter
    """
    return torch.mean(entropy_logits(linear_output))


def gradient_penalty(critic, h_s, h_t):
    """Computes gradient penelty in Wasserstein distance guided representation learning

    Examples:
        See WDGRLtrainer and WDGRLtrainerMod in kale.pipeline.domain_adapter
    """

    alpha = torch.rand(h_s.size(0), 1)
    alpha = alpha.expand(h_s.size()).type_as(h_s)
    # try:
    differences = h_t - h_s

    interpolates = h_s + (alpha * differences)
    interpolates = torch.cat((interpolates, h_s, h_t), dim=0).requires_grad_()

    preds = critic(interpolates)
    gradients = grad(preds, interpolates, grad_outputs=torch.ones_like(preds), retain_graph=True, create_graph=True,)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    # except:
    #     gradient_penalty = 0

    return gradient_penalty


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Code from XLearn: computes the full kernel matrix,
    which is less than optimal since we don't use all of it
    with the linear MMD estimate.

    Examples:
        See DANtrainer and JANtrainer in kale.pipeline.domain_adapter
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def compute_mmd_loss(kernel_values, batch_size):
    """Computes the Maximum Mean Discrepancy (MMD) between domains.

    Examples:
        See DANtrainer and JANtrainer in kale.pipeline.domain_adapter
    """
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i + 1) % batch_size
        t1, t2 = s1 + batch_size, s2 + batch_size
        loss += kernel_values[s1, s2] + kernel_values[t1, t2]
        loss -= kernel_values[s1, t2] + kernel_values[s2, t1]
    return loss / float(batch_size)


def hsic(kx, ky, device):
    """Perform independent test (HSIC) between two set of variables x and y.

    Args:
        kx (2-D tensor): kernel matrix of x, shape (n_samples, n_samples)
        ky (2-D tensor): kernel matrix of y, shape (n_samples, n_samples)

    Returns:
        [tensor]: Independent test score >= 0

    Reference:
        [1] Gretton, Arthur, Bousquet, Olivier, Smola, Alex, and Schölkopf, Bernhard. Measuring Statistical Dependence
            with Hilbert-Schmidt Norms. In Algorithmic Learning Theory (ALT), pp. 63–77. 2005.
        [2] Gretton, Arthur, Fukumizu, Kenji, Teo, Choon H., Song, Le, Schölkopf, Bernhard, and Smola, Alex J. A Kernel
            Statistical Test of Independence. In Advances in Neural Information Processing Systems, pp. 585–592. 2008.
    """

    n = kx.shape[0]
    if ky.shape[0] != n:
        raise ValueError("kx and ky are expected to have the same sample sizes.")
    ctr_mat = torch.eye(n, device=device) - torch.ones((n, n), device=device) / n
    return torch.trace(torch.mm(torch.mm(torch.mm(kx, ctr_mat), ky), ctr_mat)) / (n ** 2)


def euclidean(x1, x2):
    """Compute Eucliean distance

    Args:
        x1 (torch.Tensor): variables set 1
        x2 (torch.Tensor): variables set 2

    Returns:
        torch.Tensor: Eucliean distance
    """
    # n_features = x1.shape[0]
    # if x2.shape[0] != n_features:
    #     raise ValueError("x1, x2 are expected to have the same dimensions.")
    # return ((x1 - x2) ** 2).sum().sqrt() / n_features
    return ((x1 - x2) ** 2).sum().sqrt()
