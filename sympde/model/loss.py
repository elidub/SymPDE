import torch

#loss function with rel/abs Lp loss
class LpLoss:
    def __init__(self, p=2, dim = (1,2)):
        self.p = p
        self.dim = dim

    def rel(self, y_pred, y):

        diff_norms  = torch.norm(y_pred - y, p=self.p, dim=self.dim)
        y_norms = torch.norm(y, p=self.p, dim=self.dim)

        loss = diff_norms/y_norms

        loss = loss.mean()

        return loss

    def __call__(self, y_pred, y, **kwargs):
        return self.rel(y_pred, y)