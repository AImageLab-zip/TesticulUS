

from argparse import Namespace
import torch
from torch import nn
import numpy as np
import wandb


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        # sum all 2N terms of loss instead of getting mean val
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        ''' Cosine similarity or dot similarity for computing loss '''
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)  # I(2Nx2N), identity matrix
        # lower diagonal matrix, N non-zero elements
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        # upper diagonal matrix, N non-zero elements
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))  # [2N, 2N], with 4N elements are non-zero
        mask = (1 - mask).type(torch.bool)  # [2N, 2N], with 4(N^2 - N) elements are "True"
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        # extend the dimensions before calculating similarity
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C), N input samples
        # y shape: (1, 2N, C), 2N output representations
        # v shape: (N, 2N)
        # extend the dimensions before calculating similarity
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        if self.batch_size != zis.shape[0]:
            self.batch_size = zis.shape[0]  # the last batch may not have the same batch size

        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        representations = torch.cat([zjs, zis], dim=0)  # [N, C] => [2N, C]

        similarity_matrix = self.similarity_function(representations, representations)  # [2N, 2N]

        # filter out the scores from the positive samples
        # upper diagonal, N x [left, right] positive sample pairs
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        # lower diagonal, N x [right, left] positive sample pairs
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)  # similarity of positive pairs, [2N, 1]

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)  # [2N, 2N]

        # [2N, 2N+1], the 2N+1 elements of one column are used for one loss term
        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        # labels are all 0, meaning the first value of each vector is the nominator term of CELoss
        # each denominator contains 2N+1-2 = 2N-1 terms, corresponding to all similarities between the sample and other samples.
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        # Don't know why it is divided by 2N, the CELoss can set directly to reduction='mean'
        return loss / (2 * self.batch_size)


class HiCoLoss(nn.Module):
    def __init__(self, device, net: nn.Module, args: Namespace):
        super(HiCoLoss, self).__init__()
        self.device = device
        self.net = net
        self.args = args
        self.nt_xent_criterion = NTXentLoss(self.device, self.args.batch_size,
                                            temperature=0.5, use_cosine_similarity=True)

    def forward(self, x, xi, xj, y):
        x, y = x.to(self.device), y.to(self.device)
        xi, xj = xi.to(self.device), xj.to(self.device)

        ris, kis, zis, labelis = self.net(xi)
        rjs, kjs, zjs, labeljs = self.net(xj)

        sup_loss = nn.functional.cross_entropy(labelis, y) + nn.functional.cross_entropy(labeljs, y)

        zis = nn.functional.normalize(zis, dim=1)
        zjs = nn.functional.normalize(zjs, dim=1)

        # normalize projection feature vectors (p3)
        kis = nn.functional.normalize(kis, dim=1)
        kjs = nn.functional.normalize(kjs, dim=1)

        # normalize projection feature vectors (p2)
        ris = nn.functional.normalize(ris, dim=1)
        rjs = nn.functional.normalize(rjs, dim=1)

        # Peer-level
        loss_ll = self.nt_xent_criterion(ris, rjs)    # ll
        loss_mm = self.nt_xent_criterion(kis, kjs)    # mm
        loss_gg = self.nt_xent_criterion(zis, zjs)    # gg
        loss_peer_level = loss_ll + loss_mm + loss_gg

        # Cross-level
        loss_53 = self.nt_xent_criterion(zis, kis) + self.nt_xent_criterion(zjs, kjs)  # gm
        loss_52 = self.nt_xent_criterion(zis, ris) + self.nt_xent_criterion(zjs, rjs)  # gl
        loss_cross_level = loss_53 + loss_52

        loss = 0.5 * loss_peer_level + (1 - 0.5) * loss_cross_level + 0.2 * sup_loss
        wandb.log({
            "train/sup_loss": sup_loss.item(),
            "train/loss_peer_level": loss_peer_level.item(),
            "train/loss_cross_lebel": loss_cross_level.item(),

        })
        return loss
