import torch

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_metric


@register_metric('edge_segmentation')
def EdgeSegmentation(**kwargs):
    aux_pred, aux_true, aux_loss = \
        kwargs['aux_pred'], kwargs['aux_true'], kwargs['aux_loss']
    from sklearn.metrics import (
        recall_score,
        roc_auc_score,
        jaccard_score
    )

    true, pred_score = torch.cat(aux_true), torch.cat(aux_pred)
    if len(pred_score.shape) == 1 or pred_score.shape[1] == 1:
        pred_int =  (pred_score > cfg.model.thresh).long()
    else:
        pred_int =  pred_score.max(dim=1)[1]  # 返回最大值的index
    try:
        r_a_score = roc_auc_score(true, pred_score)
    except ValueError:
        r_a_score = 0.0

    return {
        'aux_accuracy': [
            round(x, cfg.round) for x in
            recall_score(true, pred_int, average=None).tolist()
        ],
        'aux_IoU': [
            round(x, cfg.round) for x in
            jaccard_score(true, pred_int, average=None).tolist()
        ],
        'aux_mAcc':
            recall_score(true, pred_int, average='macro').tolist(),
        'aux_mIoU':
            jaccard_score(true, pred_int, average='macro').tolist(),
        'aux_loss': sum(aux_loss) / len(aux_loss)
    }


@register_metric('contrastive_learning')
def ContrastiveLearning(**kwargs):
    """
    对比学习的度量指标
    """
    if 'contrastive_loss' not in kwargs:
        return {}
        
    contrastive_losses = kwargs.get('contrastive_loss', [])
    
    if len(contrastive_losses) == 0:
        return {}
        
    # 计算平均对比学习损失
    avg_contrastive_loss = sum(contrastive_losses) / len(contrastive_losses)
    
    return {
        'contrastive_loss': round(avg_contrastive_loss, cfg.round)
    }
