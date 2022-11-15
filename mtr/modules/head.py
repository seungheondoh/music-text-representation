import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class TripletHead(nn.Module):
    def __init__(self, margin):
        super(TripletHead, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, anchor, positive, negative, size_average=True):
        cosine_positive = nn.CosineSimilarity(dim=-1)(anchor, positive)
        cosine_negative = nn.CosineSimilarity(dim=-1)(anchor, negative)
        losses = self.relu(self.margin - cosine_positive + cosine_negative)
        return losses.mean()

class CLIPHead(nn.Module):
    def __init__(self, logit_scale):
        super(CLIPHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.logit_scale = logit_scale

    def forward(self, h1, h2):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            h1 = SyncFunction.apply(h1)
            h2 = SyncFunction.apply(h2)
        device = h1.device
        temperature = torch.clamp(self.logit_scale.exp(), max=100)
        h1 = nn.functional.normalize(h1, dim=1)
        h2 = nn.functional.normalize(h2, dim=1)
        logits = torch.einsum('nc,mc->nm', [h1, h2]) * temperature.to(device)
        N = logits.shape[0]  # batch size per GPU
        labels = torch.arange(N, dtype=torch.long, device=device)
        return F.cross_entropy(logits, labels)
    
    def acc(self, h1, h2):
        device = h1.device
        temperature = torch.clamp(self.logit_scale.exp(), max=100)
        h1 = nn.functional.normalize(h1, dim=1)
        h2 = nn.functional.normalize(h2, dim=1)
        logits = torch.einsum('nc,mc->nm', [h1, h2]) * temperature.to(device)
        N = logits.shape[0]  # batch size per GPU
        y_pred = logits.max(dim=-1)[1]
        target = torch.arange(N, dtype=torch.long, device=device)
        train_acc = torch.sum(y_pred == target)
        acc = train_acc / N
        return acc

class ContrastiveHead(nn.Module):
    def __init__(self, temperature=0.2):
        super(ContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, pos, neg):
        """
        pos: batch x 1 (sim score)
        neg: batch x neg (sim score)
        """
        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).cuda()
        loss = self.criterion(logits, labels)
        return loss

class ClsHead(nn.Module):
    """Simplest classifier head, with only one fc layer.
    """
    def __init__(self, in_channels, num_classes=1054, with_avg_pool=False):
        super(ClsHead, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fc_cls = nn.Linear(in_channels, num_classes, bias=False) # class centroid
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()

    def forward(self, x,y):
        if self.with_avg_pool:
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
        output = self.fc_cls(x)
        logits = self.sigmoid(output)
        loss = self.loss_fn(logits,y)
        return loss

class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]
        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)
        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_input = grad_input.contiguous() # contiguous error
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)
        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]
