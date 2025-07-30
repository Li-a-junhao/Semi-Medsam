import torch.nn as nn
import torch
import torch.nn.functional as F
from utils_ import ramps


class MT_framework(nn.Module):
    def __init__(self, args, model, model_ema, gpu):
        super(MT_framework, self).__init__()
        self.gpu = gpu
        self.args = args
        self.model = model
        self.model_ema = model_ema
        self.epoch_num = args.epoch_num

    def forward(self, x, y, label_w_ulb):
        mask_w_ulb = torch.where(label_w_ulb == 255)
        pred, aux_loss = self.model(x, y)
        pred_ema, _ = self.model_ema(x, y)
        aux_loss = aux_loss.to(self.gpu)

        if self.model.training:
            pred_lb, pred_ulb = pred[:self.args.batch_size], pred[self.args.batch_size:]
            pred_lb_ema = pred_ema[:self.args.batch_size].detach()
            pred_ulb_ema = pred_ema[self.args.batch_size:].detach()
        else:
            pred_lb, pred_ulb = pred[:1], pred[1:]
            pred_lb_ema, pred_ulb_ema = pred_ema[:1].detach(), pred_ema[1:].detach()

        with torch.no_grad():
            pred_ulb_soft_ema = F.softmax(pred_ulb_ema, dim=1)
            pseudo_label_ema = torch.argmax(pred_ulb_soft_ema, dim=1, keepdim=False)
            confidence, _ = torch.max(pred_ulb_soft_ema, dim=1)
            pseudo_label_ema[confidence < 0.75] = 255
            pseudo_label_ema[mask_w_ulb] = 255

            pred_ulb_soft = F.softmax(pred_ulb, dim=1)
            pseudo_label = torch.argmax(pred_ulb_soft, dim=1, keepdim=False)
            confidence, _ = torch.max(pred_ulb_soft, dim=1)
            pseudo_label[confidence < 0.75] = 255
            pseudo_label[mask_w_ulb] = 255

        return pred_lb, pred_ulb, pred_lb_ema, pred_ulb_ema, pseudo_label, pseudo_label_ema, aux_loss

    def get_current_consistency_weight(self, epoch):
        return ramps.sigmoid_rampup(epoch, self.args.epoch_num)

    def update_ema(self, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(self.model_ema.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
