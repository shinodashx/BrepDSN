import logging
import os
import sys
sys.path.append('./')
import custom_graphgym  # noqa, register custom modules
import torch

from torch_geometric import seed_everything
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (
    cfg,
    dump_cfg,
    set_cfg,
    load_cfg,
    set_out_dir,
    set_run_dir,
)
from torch_geometric.graphgym.loader import get_loader, load_dataset
from torch_geometric.graphgym.logger import create_logger, set_printing
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
from torch_geometric.graphgym.register import train_dict, optimizer_dict
from torch_geometric.graphgym.train import my_train, my_eval
from torch_geometric.graphgym.utils.agg_runs import agg_runs
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.device import auto_select_device


def my_set_dataset_info(dataset):
    if cfg.share.dim_node_out == 0:
        try:
            cfg.share.dim_node_out = len(dataset[0].CLASSES)  # 通过数据集推断
            if 'classification' in cfg.dataset.task_type and cfg.share.dim_node_out == 2:
                cfg.share.dim_node_out = 1
        except Exception:
            raise RuntimeError("error when specify model dim_out(in node related task.)")

    if cfg.model.has_aux:
        assert cfg.share.aux_dim_node_out != 0 or cfg.share.aux_dim_edge_out != 0, \
            "auxiliary head will be initialized, plz specify the aux output dimension."

    # count number of dataset splits
    cfg.share.num_splits = len(dataset)



def my_create_dataset():
    r"""
    Create dataset object
    Returns: PyG dataset object
    """
    dataset = load_dataset()
    my_set_dataset_info(dataset)
    return dataset



def my_create_loader():
    # 为了同时加载train, val, test数据集所使用的dataloader;
    dataset = my_create_dataset()
    # train loader
    if cfg.dataset.task == 'graph':
        id = dataset.data['train_graph_index']
        loaders = [get_loader(dataset[id], cfg.train.sampler, cfg.train.batch_size, shuffle=True)]
        delattr(dataset.data, 'train_graph_index')
    else:
        loaders = [get_loader(dataset[0], cfg.train.sampler, cfg.train.batch_size, shuffle=True)]

    # val and test loaders
    for i in range(1, cfg.share.num_splits):
        if cfg.dataset.task == 'graph':
            split_names = ['val_graph_index', 'test_graph_index']
            id = dataset.data[split_names[i]]
            loaders.append(get_loader(dataset[id], cfg.val.sampler, cfg.train.batch_size, shuffle=False))
            delattr(dataset.data, split_names[i])
        else:
            loaders.append(get_loader(dataset[i], cfg.val.sampler, cfg.val.batch_size, shuffle=False))
    return loaders



if __name__ == '__main__':
    # Load cmd line args
    args = parse_args()
    # Load config file
    load_cfg(cfg, args)
    set_out_dir(cfg.out_dir, args.cfg_file)
    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)
    if not args.eval:
        dump_cfg(cfg)
    else:
        cfg.train.mode = 'eval'
        if 'train.epoch_resume' not in args.opts:
            raise ValueError('please give a specific checkpoint index')
    # Repeat for different random seeds
    for i in range(args.repeat):
        set_run_dir(cfg.out_dir)
        set_printing()
        # Set configurations for each run
        cfg.seed = cfg.seed + 1
        seed_everything(cfg.seed)
        auto_select_device()
        # Set machine learning pipeline
        loaders = my_create_loader()
        loggers = create_logger()
        pyg_model = create_model()  # pytorch lightning module / torch.nn.Module
        optimizer = create_optimizer(pyg_model.parameters(), cfg.optim)
        scheduler = create_scheduler(optimizer, cfg.optim)
        # Print model info
        logging.info(pyg_model)
        logging.info(cfg)
        cfg.params = params_count(pyg_model)
        logging.info('Num parameters: %s', cfg.params)
        # Start training, 按照训练或测试模式调用对应API
        if cfg.train.mode == 'standard':
            my_train(loggers, loaders, pyg_model, optimizer, scheduler)
        elif cfg.train.mode == 'eval':
            my_eval(loggers, loaders, pyg_model)
        else:
            train_dict[cfg.train.mode](loggers, loaders, pyg_model, optimizer, scheduler)

    # # Aggregate results from different seeds
    # agg_runs(cfg.out_dir, cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done
    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')














