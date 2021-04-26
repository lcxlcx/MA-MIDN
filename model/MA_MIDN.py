import torch
import torch.nn as nn
import torch.nn.functional as F


class MA_MIDN(nn.Module):
    def __init__(self):
        super(MA_MIDN, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )



        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU()
        )



        self.attention1 = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, 2)
        )
        self.attention2 = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print('========================')
        # print(x.shape)
        x = x.squeeze(0)
        # print(x.shape)
        # print('========================')
        H = self.feature_extractor_part1(x)

        H = H.view(-1, 50 * 4 * 4)

        H = self.feature_extractor_part2(H)

        A = self.attention1(H)
        B = self.attention2(H)
        score = F.softmax(A, dim=0) * F.softmax(B, dim=1)

        im_cls_score,_ = torch.max(score,dim=1,keepdim=True)

        im_cls_score = torch.transpose(im_cls_score, 0, 1)

        M = torch.mm(im_cls_score, H)
        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, im_cls_score

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        return neg_log_likelihood, A
