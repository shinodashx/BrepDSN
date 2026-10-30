# import warnings
# import logging
# from typing import Any, Dict, Optional

# import copy
# import time

# import numpy as np
# import torch
# import os
# import pickle

# from tqdm import tqdm
# from torch.utils.data import DataLoader

# from torch_geometric.data.lightning.datamodule import LightningDataModule
# from torch_geometric.graphgym import create_loader
# from torch_geometric.graphgym.checkpoint import get_ckpt_dir, clean_ckpt, load_ckpt, save_ckpt
# from torch_geometric.graphgym.config import cfg
# from torch_geometric.graphgym.imports import pl
# from torch_geometric.graphgym.logger import LoggerCallback
# from torch_geometric.graphgym.model_builder import GraphGymModule
# from torch_geometric.graphgym.utils.epoch import is_ckpt_epoch, is_eval_epoch, is_train_eval_epoch
# from torch_geometric.graphgym.loss import my_compute_loss, my_compute_aux_loss, my_compute_multi_stage_loss



# class GraphGymDataModule(LightningDataModule):
#     r"""A :class:`pytorch_lightning.LightningDataModule` for handling data
#     loading routines in GraphGym.

#     This class provides data loaders for training, validation, and testing, and
#     can be accessed through the :meth:`train_dataloader`,
#     :meth:`val_dataloader`, and :meth:`test_dataloader` methods, respectively.
#     """
#     def __init__(self):
#         self.loaders = create_loader()
#         super().__init__(has_val=True, has_test=True)

#     def train_dataloader(self) -> DataLoader:
#         return self.loaders[0]

#     def val_dataloader(self) -> DataLoader:
#         # better way would be to test after fit.
#         # First call trainer.fit(...) then trainer.test(...)
#         return self.loaders[1]

#     def test_dataloader(self) -> DataLoader:
#         return self.loaders[2]


# def train(
#     model: GraphGymModule,
#     datamodule: GraphGymDataModule,
#     logger: bool = True,
#     trainer_config: Optional[Dict[str, Any]] = None,
# ):
#     r"""Trains a GraphGym model using PyTorch Lightning.

#     Args:
#         model (GraphGymModule): The GraphGym model.
#         datamodule (GraphGymDataModule): The GraphGym data module.
#         logger (bool, optional): Whether to enable logging during training.
#             (default: :obj:`True`)
#         trainer_config (dict, optional): Additional trainer configuration.
#     """
#     warnings.filterwarnings('ignore', '.*use `CSVLogger` as the default.*')

#     callbacks = []
#     if logger:
#         callbacks.append(LoggerCallback())
#     if cfg.train.enable_ckpt:
#         ckpt_cbk = pl.callbacks.ModelCheckpoint(dirpath=get_ckpt_dir())
#         callbacks.append(ckpt_cbk)

#     trainer_config = trainer_config or {}
#     trainer = pl.Trainer(
#         **trainer_config,
#         enable_checkpointing=cfg.train.enable_ckpt,
#         callbacks=callbacks,
#         default_root_dir=cfg.out_dir,
#         max_epochs=cfg.optim.max_epoch,
#         accelerator=cfg.accelerator,
#         devices='auto' if not torch.cuda.is_available() else cfg.devices,
#     )

#     trainer.fit(model, datamodule=datamodule)
#     trainer.test(model, datamodule=datamodule)



# def my_train(loggers, loaders, model, optimizer, scheduler):
#     """
#     The core training pipeline
#     Args:
#         loggers: List of loggers for train, val, test, respectively
#         loaders: List of loaders
#         model: GNN model
#         optimizer: PyTorch optimizer
#         scheduler: PyTorch learning rate scheduler
#     """
#     start_epoch = 0
#     if cfg.train.auto_resume:
#         start_epoch = load_ckpt(model, optimizer, scheduler, cfg.train.epoch_resume, cfg.train.ckpt_prefix)
#     if start_epoch == cfg.optim.max_epoch:
#         logging.info('Checkpoint found, Task already done')   # 由于在main.py中配置了logging，因此这里可以直接使用，是rootlogger输出
#     else:
#         logging.info('Start from epoch {}'.format(start_epoch))

#     num_splits = len(loggers)
#     split_names = ['val', 'test']
#     for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
#         train_epoch(loggers[0], loaders[0], model, optimizer, scheduler)
#         if is_train_eval_epoch(cur_epoch):
#             loggers[0].write_epoch(cur_epoch)
#         if is_eval_epoch(cur_epoch):
#             for i in range(1, num_splits - 1):
#                 eval_epoch(loggers[i], loaders[i], model, split=split_names[i - 1])
#                 loggers[i].write_epoch(cur_epoch)
#             if loggers[1].best_stat['updated']:
#                 save_ckpt(model, optimizer, scheduler, epoch=1, prefix='best')
#                 loggers[1].best_stat['updated'] = False

#         if is_ckpt_epoch(cur_epoch) and cfg.train.enable_ckpt:
#             save_ckpt(model, optimizer, scheduler, cur_epoch, prefix=cfg.train.ckpt_prefix)

#     for logger in loggers:
#         logger.close()
#     if cfg.train.ckpt_clean:
#         clean_ckpt()

#     logging.info('Task done, results saved in {}'.format(cfg.run_dir))



# def train_epoch(logger, loader, model, optimizer, scheduler):
#     model.train()
#     time_start = time.time()
#     for batch in tqdm(loader):
#         if isinstance(batch, list):
#             for idx, b in enumerate(batch):
#                 if isinstance(b, torch.Tensor):
#                     batch[idx] = b.cuda()  # 不理解这里的if else目的是什么?
#                 else:
#                     b.cuda()
#         else:
#             batch.split = 'train'
#             batch.cuda()
#         optimizer.zero_grad()
#         pred, true = model(batch)
#         # 增加判断，如果有aux(暂时设定为只有一个)，那么pred需要拆分开计算
#         if cfg.model.has_aux:
#             assert isinstance(pred, tuple) and isinstance(true, tuple), "pred and label should be tuples."
#             label, aux_label = true[0], true[1]
#             main_loss, pred_score, label = my_compute_loss(pred[0], label, model=model)
#             aux_loss, aux_pred_score, aux_label = my_compute_aux_loss(pred[1], aux_label)
#             loss = main_loss + cfg.model.aux_weight * aux_loss

#         else:
#             label, aux_label = true, None
#             loss, pred_score, label = my_compute_loss(pred, label, model=model)
#             aux_pred_score = None

#         loss.backward()
#         optimizer.step()

#         logger.update_stats(
#             true=label.detach().cpu(),
#             pred=pred_score.detach().cpu(),
#             loss=loss.item(),
#             lr=scheduler.get_last_lr()[0],
#             time_used=time.time() - time_start,
#             params=cfg.params,
#             node_area=batch[-1]['meta_data']['node_area'],
#             aux_pred=None if aux_pred_score is None else [aux_pred_score.detach().cpu()],
#             aux_loss=None if aux_label is None else [aux_loss.item() * cfg.model.aux_weight],
#             aux_true=None if aux_label is None else [aux_label.detach().cpu()]
#         )
#         time_start = time.time()
#     scheduler.step()



# def my_eval(loggers, loaders, model):
#     """
#     The core evaluation pipeline.
#     """
#     start_epoch = load_ckpt(model, None, None, cfg.train.epoch_resume, cfg.train.ckpt_prefix)
#     num_splits = len(loggers)
#     if num_splits == 3:
#         loaders.pop(0)
#         loggers.pop(0)
#         num_splits -= 1
#     split_names = ['val', 'test']
#     assert num_splits == len(split_names), 'wrong match on the number of data loaders with splits.'

#     for i, split in enumerate(split_names):
#         pred_Write, pred_Write_extra = eval_epoch(loggers[i], loaders[i], model, split=split_names[i])
#         loggers[i].write_epoch(0)
#         if pred_Write is not None:
#             with open(os.path.join(cfg.run_dir, '{}_result.pkl'.format(split)), 'wb') as f:
#                 pickle.dump(pred_Write, f)
#         if pred_Write_extra is not None:
#             with open(os.path.join(cfg.run_dir, '{}_result_extra.pkl'.format(split)), 'wb') as f:
#                 pickle.dump(pred_Write_extra, f)

#     for logger in loggers:
#         logger.close()
#     if cfg.train.ckpt_clean:
#         clean_ckpt()



# @torch.no_grad()
# def eval_epoch(logger, loader, model, split='val'):
#     model.eval()
#     time_start = time.time()

#     file_Orders, extra_infos = [], []
#     for batch in tqdm(loader):
#         if isinstance(batch, list):
#             for idx, b in enumerate(batch):
#                 if isinstance(b, torch.Tensor):
#                     batch[idx] = b.cuda()
#                 else:
#                     b.cuda()
#         else:
#             batch.split = split
#             batch.cuda()

#         if cfg.val.extra_infos:
#             temp_batch = copy.deepcopy(batch)
#             extra_info = model.model.forward_test(temp_batch)   # 获取边上的 attention weights
#             edges = temp_batch[-1].edge_index
#             extra_infos.append((extra_info.cpu().numpy(), edges.cpu().numpy()))

#         pred, true = model(batch)
#         # 将pred和 Label限定在real node中；
#         selections = np.concatenate(batch[-1]['meta_data']['node_real'])
#         selections = torch.from_numpy(selections)
#         nodes_area = batch[-1]['meta_data']['node_area']
#         nodes_selection = batch[-1]['meta_data']['node_real']

#         nodes_area_selection = []
#         for n, s in zip(nodes_area, nodes_selection):
#             if n.shape == s.shape:
#                 nodes_area_selection.append(n[s])
#             else:
#                 nodes_area_selection.append(n)

#         if cfg.model.has_aux:
#             assert isinstance(pred, tuple) and isinstance(true, tuple), "pred and label should be tuples."
#             label, aux_label = true[0], true[1]
#             if label.shape[0] == selections.shape[0]:
#                 label = label[selections]
#                 pred_x = pred[0][selections]
#             else:
#                 pred_x = pred
#             main_loss, pred_score, label = my_compute_loss(pred_x, label, model=model)
#             aux_loss, aux_pred_score, aux_label = my_compute_aux_loss(pred[1], true[1])
#             loss = main_loss + cfg.model.aux_weight * aux_loss
#         else:
#             label, aux_label = true, None
#             if label.shape[0] == selections.shape[0]:
#                 label = label[selections]
#                 pred_x = pred[selections]
#             else:
#                 pred_x = pred
#             loss, pred_score, label = my_compute_loss(pred_x, label, model=model)
#             aux_pred_score = None

#         valid_edges = batch[-1].edge_dual
#         logger.update_stats(
#             true=label.detach().cpu(),
#             pred=pred_score.detach().cpu(),
#             loss=loss.item(),
#             lr=0,
#             time_used=time.time() - time_start,
#             params=cfg.params,
#             node_area=nodes_area_selection,
#             aux_pred=None if aux_pred_score is None else [aux_pred_score.detach().cpu()],
#             aux_true=None if aux_label is None else [aux_label.detach().cpu()],
#             aux_loss=None if aux_label is None else [aux_loss.item() * cfg.model.aux_weight],
#             edge_set=None if aux_label is None else [valid_edges.cpu()]
#         )

#         file_Orders.append(batch[-1].meta_data['filename'])
#         time_start = time.time()

#     if cfg.train.mode == 'eval':
#         data_Cates, dataPreds, dataReals = [], logger['_pred'], logger['_true']
#         if cfg.model.has_aux:
#             dataAuxPreds, dataAuxReals = logger._custom_stats['aux_pred'], logger._custom_stats['aux_true']
#             valid_edges = logger._custom_stats['edge_set']

#             for fileOrder, dataPred, dataReal, dataAuxPred, dataAuxReal, valid_edge in \
#                     zip(file_Orders, dataPreds, dataReals, dataAuxPreds, dataAuxReals, valid_edges):
#                 dataCate = logger._get_pred_int(dataPred).numpy()
#                 # dataAuxCate = logger._get_pred_int(dataAuxPred).numpy()  # 因为 aux branch的输出不一定是
#                 data_Cates.append((
#                     fileOrder, dataCate, dataReal.numpy(), dataAuxPred.numpy(), dataAuxReal.numpy(), valid_edge.numpy()
#                 ))
#         else:
#             for fileOrder, dataPred, dataReal in zip(file_Orders, dataPreds, dataReals):
#                 dataCate = logger._get_pred_int(dataPred).numpy()
#                 data_Cates.append((fileOrder, dataCate, dataReal.numpy()))

#         if cfg.val.extra_infos:
#             dataExtras = []
#             for fileOrder, extra_info in zip(file_Orders, extra_infos):
#                 dataExtras.append((fileOrder, *extra_info))
#             return data_Cates, dataExtras
#         return data_Cates, None
#     else:
#         return None, 


import warnings
import logging
from typing import Any, Dict, Optional, List, Tuple

import copy
import time

import numpy as np
import torch
import os
import pickle
from collections import defaultdict

from tqdm import tqdm
from torch.utils.data import DataLoader

from torch_geometric.data.lightning.datamodule import LightningDataModule
from torch_geometric.graphgym import create_loader
from torch_geometric.graphgym.checkpoint import get_ckpt_dir, clean_ckpt, load_ckpt, save_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.imports import pl
from torch_geometric.graphgym.logger import LoggerCallback
from torch_geometric.graphgym.model_builder import GraphGymModule
from torch_geometric.graphgym.utils.epoch import is_ckpt_epoch, is_eval_epoch, is_train_eval_epoch
from torch_geometric.graphgym.loss import my_compute_loss, my_compute_aux_loss, my_compute_multi_stage_loss

# Optional eval plotting dependencies
try:
    import matplotlib
    matplotlib.use('Agg')  # headless backend
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    from sklearn.metrics import recall_score, jaccard_score
except Exception:
    recall_score = None
    jaccard_score = None



class GraphGymDataModule(LightningDataModule):
    r"""A :class:`pytorch_lightning.LightningDataModule` for handling data
    loading routines in GraphGym.

    This class provides data loaders for training, validation, and testing, and
    can be accessed through the :meth:`train_dataloader`,
    :meth:`val_dataloader`, and :meth:`test_dataloader` methods, respectively.
    """
    def __init__(self):
        self.loaders = create_loader()
        super().__init__(has_val=True, has_test=True)

    def train_dataloader(self) -> DataLoader:
        return self.loaders[0]

    def val_dataloader(self) -> DataLoader:
        # better way would be to test after fit.
        # First call trainer.fit(...) then trainer.test(...)
        return self.loaders[1]

    def test_dataloader(self) -> DataLoader:
        return self.loaders[2]


def train(
    model: GraphGymModule,
    datamodule: GraphGymDataModule,
    logger: bool = True,
    trainer_config: Optional[Dict[str, Any]] = None,
):
    r"""Trains a GraphGym model using PyTorch Lightning.

    Args:
        model (GraphGymModule): The GraphGym model.
        datamodule (GraphGymDataModule): The GraphGym data module.
        logger (bool, optional): Whether to enable logging during training.
            (default: :obj:`True`)
        trainer_config (dict, optional): Additional trainer configuration.
    """
    warnings.filterwarnings('ignore', '.*use `CSVLogger` as the default.*')

    callbacks = []
    if logger:
        callbacks.append(LoggerCallback())
    if cfg.train.enable_ckpt:
        ckpt_cbk = pl.callbacks.ModelCheckpoint(dirpath=get_ckpt_dir())
        callbacks.append(ckpt_cbk)

    trainer_config = trainer_config or {}
    trainer = pl.Trainer(
        **trainer_config,
        enable_checkpointing=cfg.train.enable_ckpt,
        callbacks=callbacks,
        default_root_dir=cfg.out_dir,
        max_epochs=cfg.optim.max_epoch,
        accelerator=cfg.accelerator,
        devices='auto' if not torch.cuda.is_available() else cfg.devices,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)



def my_train(loggers, loaders, model, optimizer, scheduler):
    """
    The core training pipeline
    Args:
        loggers: List of loggers for train, val, test, respectively
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler
    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler, cfg.train.epoch_resume, cfg.train.ckpt_prefix)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')   # 由于在main.py中配置了logging，因此这里可以直接使用，是rootlogger输出
    else:
        logging.info('Start from epoch {}'.format(start_epoch))

    num_splits = len(loggers)
    split_names = ['val', 'test']
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler, cur_epoch)
        if is_train_eval_epoch(cur_epoch):
            loggers[0].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits - 1):
                eval_epoch(loggers[i], loaders[i], model, split=split_names[i - 1])
                loggers[i].write_epoch(cur_epoch)
            if loggers[1].best_stat['updated']:
                save_ckpt(model, optimizer, scheduler, epoch=1, prefix='best')
                loggers[1].best_stat['updated'] = False

        if is_ckpt_epoch(cur_epoch) and cfg.train.enable_ckpt:
            save_ckpt(model, optimizer, scheduler, cur_epoch, prefix=cfg.train.ckpt_prefix)

    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.run_dir))



def train_epoch(logger, loader, model, optimizer, scheduler, cur_epoch):
    model.train()
    contrastive_scale = (1 + np.cos(np.pi * min(max((cur_epoch - 1) / 199, 0), 1))) / 2
    time_start = time.time()
    for batch in tqdm(loader):
        if isinstance(batch, list):
            for idx, b in enumerate(batch):
                if isinstance(b, torch.Tensor):
                    batch[idx] = b.cuda()  # 不理解这里的if else目的是什么?
                else:
                    b.cuda()
        else:
            batch.split = 'train'
            batch.cuda()
        optimizer.zero_grad()
        pred, true = model(batch)
        # 增加判断，如果有aux(暂时设定为只有一个)，那么pred需要拆分开计算
        if cfg.model.has_aux:
            assert isinstance(pred, tuple) and isinstance(true, tuple), "pred and label should be tuples."
            label, aux_label = true[0], true[1]
            main_loss, pred_score, label = my_compute_loss(pred[0], label, model=model)
            aux_loss, aux_pred_score, aux_label = my_compute_aux_loss(pred[1], aux_label)
            loss = main_loss + cfg.model.aux_weight * aux_loss

        else:
            label, aux_label = true, None
            loss, pred_score, label = my_compute_loss(pred, label, model=model)
            aux_pred_score = None

        if cfg.model.has_contrastive:
            contrastive_loss = model.contrastive_loss(batch)
            loss = loss + cfg.model.contrastive_loss_cfg.contrastive_weight * contrastive_scale * contrastive_loss

        loss.backward()
        optimizer.step()

        logger.update_stats(
            true=label.detach().cpu(),
            pred=pred_score.detach().cpu(),
            loss=loss.item(),
            lr=scheduler.get_last_lr()[0],
            time_used=time.time() - time_start,
            params=cfg.params,
            node_area=batch[-1]['meta_data']['node_area'],
            aux_pred=None if aux_pred_score is None else [aux_pred_score.detach().cpu()],
            aux_loss=None if aux_label is None else [aux_loss.item() * cfg.model.aux_weight],
            aux_true=None if aux_label is None else [aux_label.detach().cpu()],
            contrastive_loss=[contrastive_loss.item()] if cfg.model.has_contrastive else None
        )
        time_start = time.time()
    scheduler.step()



def my_eval(loggers, loaders, model):
    """
    The core evaluation pipeline.
    """
    start_epoch = load_ckpt(model, None, None, cfg.train.epoch_resume, cfg.train.ckpt_prefix)
    num_splits = len(loggers)
    if num_splits == 3:
        loaders.pop(0)
        loggers.pop(0)
        num_splits -= 1
    split_names = ['val', 'test']
    assert num_splits == len(split_names), 'wrong match on the number of data loaders with splits.'

    for i, split in enumerate(split_names):
        pred_Write, pred_Write_extra = eval_epoch(loggers[i], loaders[i], model, split=split_names[i])
        loggers[i].write_epoch(0)
        if pred_Write is not None:
            with open(os.path.join(cfg.run_dir, '{}_result.pkl'.format(split)), 'wb') as f:
                pickle.dump(pred_Write, f)
        if pred_Write_extra is not None:
            with open(os.path.join(cfg.run_dir, '{}_result_extra.pkl'.format(split)), 'wb') as f:
                pickle.dump(pred_Write_extra, f)

        # Optionally generate mAcc/mIoU vs number of surfaces plot
        if hasattr(cfg, 'val') and hasattr(cfg.val, 'surface_plot') and getattr(cfg.val.surface_plot, 'enable', False):
            do_split = getattr(cfg.val.surface_plot, 'split', 'test')
            if do_split in ('both', split):
                try:
                    _maybe_plot_macc_miou_vs_surfaces(pred_Write, split)
                except Exception as e:
                    logging.warning(f"Surface-plot generation failed for split '{split}': {e}")

    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()



@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()

    file_Orders, extra_infos = [], []
    for batch in tqdm(loader):
        if isinstance(batch, list):
            for idx, b in enumerate(batch):
                if isinstance(b, torch.Tensor):
                    batch[idx] = b.cuda()
                else:
                    b.cuda()
        else:
            batch.split = split
            batch.cuda()

        if cfg.val.extra_infos:
            temp_batch = copy.deepcopy(batch)
            extra_info = model.model.forward_test(temp_batch)   # 获取边上的 attention weights
            edges = temp_batch[-1].edge_index
            extra_infos.append((extra_info.cpu().numpy(), edges.cpu().numpy()))

        pred, true = model(batch)
        # 将pred和 Label限定在real node中；
        selections = np.concatenate(batch[-1]['meta_data']['node_real'])
        selections = torch.from_numpy(selections)
        nodes_area = batch[-1]['meta_data']['node_area']
        nodes_selection = batch[-1]['meta_data']['node_real']

        nodes_area_selection = []
        for n, s in zip(nodes_area, nodes_selection):
            if n.shape == s.shape:
                nodes_area_selection.append(n[s])
            else:
                nodes_area_selection.append(n)

        if cfg.model.has_aux:
            assert isinstance(pred, tuple) and isinstance(true, tuple), "pred and label should be tuples."
            label, aux_label = true[0], true[1]
            if label.shape[0] == selections.shape[0]:
                label = label[selections]
                pred_x = pred[0][selections]
            else:
                pred_x = pred
            main_loss, pred_score, label = my_compute_loss(pred_x, label, model=model)
            aux_loss, aux_pred_score, aux_label = my_compute_aux_loss(pred[1], true[1])
            loss = main_loss + cfg.model.aux_weight * aux_loss
        else:
            label, aux_label = true, None
            if label.shape[0] == selections.shape[0]:
                label = label[selections]
                pred_x = pred[selections]
            else:
                pred_x = pred
            loss, pred_score, label = my_compute_loss(pred_x, label, model=model)
            aux_pred_score = None

        if cfg.model.has_contrastive:
            contrastive_loss = model.contrastive_loss(batch)
            loss = loss + cfg.model.contrastive_loss_cfg.contrastive_weight * contrastive_loss

        valid_edges = batch[-1].edge_dual
        logger.update_stats(
            true=label.detach().cpu(),
            pred=pred_score.detach().cpu(),
            loss=loss.item(),
            lr=0,
            time_used=time.time() - time_start,
            params=cfg.params,
            node_area=nodes_area_selection,
            aux_pred=None if aux_pred_score is None else [aux_pred_score.detach().cpu()],
            aux_true=None if aux_label is None else [aux_label.detach().cpu()],
            aux_loss=None if aux_label is None else [aux_loss.item() * cfg.model.aux_weight],
            edge_set=None if aux_label is None else [valid_edges.cpu()],
            contrastive_loss=[contrastive_loss.item()] if cfg.model.has_contrastive else None
        )

        file_Orders.append(batch[-1].meta_data['filename'])
        time_start = time.time()

    if cfg.train.mode == 'eval':
        data_Cates, dataPreds, dataReals = [], logger['_pred'], logger['_true']
        if cfg.model.has_aux:
            dataAuxPreds, dataAuxReals = logger._custom_stats['aux_pred'], logger._custom_stats['aux_true']
            valid_edges = logger._custom_stats['edge_set']

            for fileOrder, dataPred, dataReal, dataAuxPred, dataAuxReal, valid_edge in \
                    zip(file_Orders, dataPreds, dataReals, dataAuxPreds, dataAuxReals, valid_edges):
                dataCate = logger._get_pred_int(dataPred).numpy()
                # dataAuxCate = logger._get_pred_int(dataAuxPred).numpy()  # 因为 aux branch的输出不一定是
                data_Cates.append((
                    fileOrder, dataCate, dataReal.numpy(), dataAuxPred.numpy(), dataAuxReal.numpy(), valid_edge.numpy()
                ))
        else:
            for fileOrder, dataPred, dataReal in zip(file_Orders, dataPreds, dataReals):
                dataCate = logger._get_pred_int(dataPred).numpy()
                data_Cates.append((fileOrder, dataCate, dataReal.numpy()))

        if cfg.val.extra_infos:
            dataExtras = []
            for fileOrder, extra_info in zip(file_Orders, extra_infos):
                dataExtras.append((fileOrder, *extra_info))
            return data_Cates, dataExtras
        return data_Cates, None
    else:
        return None, None


def _maybe_plot_macc_miou_vs_surfaces(
    pred_write: List[Tuple], split: str
) -> None:
    """Create and save mAcc/mIoU vs number of surfaces plot and CSV.

    pred_write is a list of tuples per sample:
      - no-aux:   (filename, pred_int_array, true_array)
      - with-aux: (filename, pred_int_array, true_array, ...)
    We only use the first three fields.
    """
    if plt is None or recall_score is None or jaccard_score is None:
        raise RuntimeError('matplotlib or sklearn is unavailable in the environment')

    # Aggregate per sample
    grouped = defaultdict(list)  # n_surfaces -> list of (mAcc, mIoU)
    per_sample_rows = []  # for CSV output (filename, n_surfaces, mAcc, mIoU)

    for item in pred_write:
        if len(item) < 3:
            continue
        fname = item[0]
        pred = np.asarray(item[1]).reshape(-1)
        true = np.asarray(item[2]).reshape(-1)

        # Basic sanity alignment
        n = min(len(pred), len(true))
        if n == 0:
            continue
        pred = pred[:n]
        true = true[:n]

        # Compute per-sample metrics
        macc = recall_score(true, pred, average='macro', zero_division=0)
        miou = jaccard_score(true, pred, average='macro', zero_division=0)
        n_surfaces = int(n)
        grouped[n_surfaces].append((macc, miou))
        per_sample_rows.append((fname, n_surfaces, float(macc), float(miou)))

    if len(grouped) == 0:
        raise RuntimeError('No valid samples to plot (grouped is empty).')

    # Compute averages per surface count
    xs = sorted(grouped.keys())
    macc_means = [float(np.mean([m for m, _ in grouped[x]])) for x in xs]
    miou_means = [float(np.mean([i for _, i in grouped[x]])) for x in xs]
    counts = [len(grouped[x]) for x in xs]

    # Save CSV summary
    csv_path = os.path.join(cfg.run_dir, f'{split}_mAcc_mIoU_by_surfaces.csv')
    try:
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('num_surfaces,avg_mAcc,avg_mIoU,count\n')
            for x, ma, mi, c in zip(xs, macc_means, miou_means, counts):
                f.write(f'{x},{ma:.6f},{mi:.6f},{c}\n')
    except Exception as e:
        logging.warning(f'Failed to save CSV summary: {e}')

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, macc_means, marker='o', label='mAcc')
    ax.plot(xs, miou_means, marker='s', label='mIoU')
    ax.set_xlabel('Number of Surfaces')
    ax.set_ylabel('Score')
    ax.set_title(f'{split.upper()}: mAcc/mIoU vs Number of Surfaces')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    fig.tight_layout()

    png_path = os.path.join(cfg.run_dir, f'{split}_mAcc_mIoU_vs_surfaces.png')
    try:
        fig.savefig(png_path, dpi=200)
    finally:
        plt.close(fig)
