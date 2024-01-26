# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class EstimatorCV():
    def __init__(self, feature_num, class_num):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num).cuda()
        self.Ave = torch.zeros(class_num, feature_num).cuda()
        self.Amount = torch.zeros(class_num).cuda()

    def update_CV(self, features, labels):
        # features: N, H
        # labels: N
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot = torch.zeros(N, C).cuda()

        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)


        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = var_temp.pow(2).sum(0).div(Amount_CxA)

        sum_weight_CV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1).expand(C, A)
        )

        weight_CV[weight_CV != weight_CV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul((self.Ave - ave_CxA).pow(2))

        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
                           .mul(weight_CV)).detach() + additional_CV.detach()

        self.Ave = (self.Ave.mul(1 - weight_CV) + ave_CxA.mul(weight_CV)).detach()

        self.Amount += onehot.sum(0)


class EndoLoss(nn.Module):
    def __init__(self, feature_num, class_num):
        super(EndoLoss, self).__init__()

        self.estimator = EstimatorCV(feature_num, class_num)
        self.estimator_ent = EstimatorCV(feature_num, class_num)
        self.estimator_ctx = EstimatorCV(feature_num, class_num)

        self.class_num = class_num

        self.cross_entropy = nn.CrossEntropyLoss()
        self.q = 0.7
        self.gce = GCELoss(q=self.q)


    def isda_aug(self, fc, features, y, labels, cv_matrix, ratio, noise=False):

        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        weight_m = list(fc.parameters())[0]

        NxW_ij = weight_m.expand(N, C, A)

        NxW_kj = torch.gather(NxW_ij,
                              1,
                              labels.view(N, 1, 1)
                              .expand(N, C, A))

        CV_temp = cv_matrix[labels]


        sigma2 = ratio * (weight_m - NxW_kj).pow(2).mul(
            CV_temp.view(N, 1, A).expand(N, C, A)
        ).sum(2)

        if noise:
            aug_result = -self.q*y + 0.5*self.q*self.q*sigma2
        else:
            aug_result = y + 0.5 * sigma2

        return aug_result

    def forward(self, fc, logits, features, target_x, ratio, logits_ent=None, features_ent=None, logits_ctx=None, features_ctx=None, entity_mask=None, noise=False, beta1=0.1, beta2=0.1):
        # logits: N, C
        # features: N, d
        # target_x: N

        y = logits
        y_ent = logits_ent
        y_ctx = logits_ctx

        self.estimator.update_CV(features.detach(), target_x)
        

        isda_aug_y = self.isda_aug(fc, features, y, target_x, self.estimator.CoVariance.detach(), ratio, noise)

        if entity_mask is not None:
            self.estimator_ent.update_CV(features_ent.detach(), target_x)
            self.estimator_ctx.update_CV(features_ctx.detach(), target_x)
            isda_aug_y_ent = self.isda_aug(fc, features_ent, y_ent, target_x, self.estimator_ent.CoVariance.detach(), ratio)
            isda_aug_y_ctx = self.isda_aug(fc, features_ctx, y_ctx, target_x, self.estimator_ctx.CoVariance.detach(), ratio)

            l = entity_mask.size(0)
            c = isda_aug_y.size(-1)

            isda_aug = isda_aug_y + beta1*isda_aug_y_ent*entity_mask.unsqueeze(-1).expand(l, c) + beta2*isda_aug_y_ctx*entity_mask.unsqueeze(-1).expand(l, c)
            # isda_aug = isda_aug_y
        else:
            isda_aug = isda_aug_y

        if noise:
            loss = self.gce(isda_aug, target_x)
        else:
            loss = self.cross_entropy(isda_aug, target_x)

        return loss

class OrthogonalLoss(nn.Module):
    def __init__(self):
        super(OrthogonalLoss, self).__init__()

    def forward(self, tensor1, tensor2):
        assert tensor1.shape == tensor2.shape

        dot_product = (tensor1 * tensor2).sum(dim=-1)

        total_loss = torch.mean(torch.abs(dot_product))
        
        return total_loss



# Generalized Cross Entropy Loss
class GCELoss(nn.Module):

    def __init__(self, q=0.7):
        super(GCELoss, self).__init__()
        self.q = q
             
    def forward(self, logits, targets):

        pred = F.softmax(logits, dim=-1)
        pred = torch.gather(pred, dim=-1, index=torch.unsqueeze(targets, -1))
        loss = (1-pred**self.q) / self.q
        loss = (loss.view(-1)).sum()

        return loss
        