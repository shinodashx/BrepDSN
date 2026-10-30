import json
import os
import torchvision
import numpy as np
import torch
import pickle
import h5py
import copy

from typing import Any, Dict, Optional
from torch_geometric.data import Data as BaseGraphData
from torchvision.transforms import ToTensor, Normalize, Resize
from torch_geometric.graphgym.config import cfg
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.transforms import *
from torch_geometric.utils import sort_edge_index
from torch_geometric.graphgym.register import register_loader, register_dataset


class DualGraphData(BaseGraphData):

    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, pos=None, time=None, **kwargs):
        """
        对偶图的数据结构需要的属性
        PrimalGraph：
          点index（Brep vertex），点特征；  ==> x
          边和点index的关联，边特征； ==> edge_index, edge_attr
          和DualGraph中的边Index的对应关系； ==> edge_dual
          边的label ==> edge_label
        DualGraph：
          点index（Brep face），点特征；  ==> x
          边index，边特征；  ==> edge_index, edge_attr
          和PrimalGraph中的边Index的对应关系  ==> edge_dual
          点的label ==> y

        Primal/DualGraph都需要meta_data，记录其对偶图中的节点和边数量（PrimalGraph记录DualGraph的节点和边数量，反之同理）
        :param x:
        :param edge_index:
        :param edge_attr:
        :param y:
        :param pos:
        :param time:
        :param kwargs:
        """
        super().__init__(x, edge_index, edge_attr, y, pos, time, **kwargs)

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        """处理对偶图的标号时，返回的应该是对偶图的点和边的数量"""
        if 'batch' in key:
            return int(value.max()) + 1
        elif 'index' in key or key == 'face':
            return self.num_nodes
        elif key in ['edge_dual', 'node_dual_idx']:
            return self.meta_data['num_dual_nodes']
        elif key == 'edge_dual_idx':
            return self.meta_data['num_dual_edges']
        elif key == 'node_orig_idx':
            return self.num_nodes
        else:
            return 0

    def concat_with_other_brep(self, other):
        # 按照DualGraphData
        cat_index = other.edge_index + self.num_nodes
        self.edge_index = torch.cat([self.edge_index, cat_index], dim=-1)
        self.edge_dual = torch.cat([self.edge_dual, other.edge_dual + self.meta_data['num_dual_nodes']], dim=0)
        self.edge_dual_idx = torch.cat([self.edge_dual_idx, other.edge_dual_idx + self.meta_data['num_dual_edges']])

        self.x = torch.cat([self.x, other.x], dim=0)

        self.edge_attr = torch.cat([self.edge_attr, other.edge_attr], dim=0)
        self.meta_data['num_dual_edges'] += other.meta_data['num_dual_edges']
        self.meta_data['num_dual_nodes'] += other.meta_data['num_dual_nodes']
        if self.meta_data.__contains__('node_area'):
            self.meta_data['node_area'] = np.hstack([self.meta_data['node_area'], other.meta_data['node_area']])

        if self.__contains__('y'):
            self.y = torch.cat([self.y, other.y], dim=0)
        if self.__contains__('edge_label'):
            self.edge_label = torch.cat([self.edge_label, other.edge_label], dim=0)



@register_loader('DUAL_BREP')
def load_dataset_BREP(format, name, dataset_dir):
    if format == 'PyG':
        if cfg.dataset.augmentation:  # TBD
            aug = []
        else:
            aug = None
        replace_name = dataset_dir.split('/')[-1]
        json_file_pth = dataset_dir.replace(replace_name, 'train_test.json')
        dataset_train = BRepDualGraphDataset(
            brep_graph_pth=dataset_dir, json_file_pth=json_file_pth, split='train', augmentation=aug)
        dataset_val = BRepDualGraphDataset(brep_graph_pth=dataset_dir, json_file_pth=json_file_pth, split='test')
        dataset_test = BRepDualGraphDataset(brep_graph_pth=dataset_dir, json_file_pth=json_file_pth, split='test')
        return [dataset_train, dataset_val, dataset_test]
    else:
        raise NotImplementedError



@register_loader('DUAL_CC3D')
def load_dataset_CC3D(format, name, dataset_dir):
    if format == 'PyG':
        if cfg.dataset.augmentation:  # TBD
            aug = []
        else:
            aug = None
        replace_name = dataset_dir.split('/')[-1]
        json_file_pth = dataset_dir.replace(replace_name, 'data_split.json')
        dataset_train = CC3DDualGraphDataset(
            brep_graph_pth=dataset_dir, json_file_pth=json_file_pth, split='train', augmentation=aug)
        dataset_val = CC3DDualGraphDataset(brep_graph_pth=dataset_dir, json_file_pth=json_file_pth, split='test')
        dataset_test = CC3DDualGraphDataset(brep_graph_pth=dataset_dir, json_file_pth=json_file_pth, split='test')
        return [dataset_train, dataset_val, dataset_test]
    else:
        raise NotImplementedError



@register_loader('DUAL_BREP_ZONE')
def load_dataset_BREP_ZONE(format, name, dataset_dir):
    if format == 'PyG':
        if cfg.dataset.augmentation:  # TBD
            aug = []
        else:
            aug = None
        replace_name = dataset_dir.split('/')[-1]
        json_file_pth = dataset_dir.replace(replace_name, 'train_test.json')
        dataset_train = BrepZoneDualGraphDataset(
            brep_graph_pth=dataset_dir, json_file_pth=json_file_pth, split='train', augmentation=aug)
        dataset_val = BrepZoneDualGraphDataset(brep_graph_pth=dataset_dir, json_file_pth=json_file_pth, split='test')
        dataset_test = BrepZoneDualGraphDataset(brep_graph_pth=dataset_dir, json_file_pth=json_file_pth, split='test')
        return [dataset_train, dataset_val, dataset_test]
    else:
        raise NotImplementedError



@register_dataset("BREP_DUAL_DATASET")
class BRepDualGraphDataset(InMemoryDataset):
    CLASSES = ("ExtrudeSide", "ExtrudeEnd", "CutSide", "CutEnd", "Fillet", "Chamfer", "RevolveSide", "RevolveEnd")
    def __init__(self, brep_graph_pth, json_file_pth, split='train', transform=None, augmentation=None):
        super().__init__()
        self.split = split
        with open(brep_graph_pth, 'rb') as f:
            self.solid_graphs = pickle.load(f)
        stem = brep_graph_pth.split('/')[-1]
        with open(brep_graph_pth.replace(stem, 'Unit_bbox.pkl'), 'rb') as f:
            self.solid_graphs.update(pickle.load(f))
        # self.solid_graphs = h5py.File(brep_graph_pth, 'r')

        with open(json_file_pth, 'r') as f:
            json_file = json.load(f)
            solid_names = json_file[split]  # 根据data_split中的 train/valid/test获取数据序列
            self.solid_names = list(set(solid_names).intersection(self.solid_graphs.keys()))

        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.center_and_scale

        if augmentation is not None:
            self.augmentation = augmentation
        else:
            if self.split == 'train':
                self.augmentation = []  # self.Roll_Pitch_Yaw_Rotation
            else:
                self.augmentation = None

        src_data = self._load_source_graph('unit_bbox')
        src_data['fn_step_label'] = np.zeros(6, dtype=np.int64)
        self.bbox_vertex_graph, self.bbox_face_graph = self._gen_graph('unit_bbox', **src_data)

        self.bbox_face_graph.x = self.bbox_face_graph.x.permute(0, 3, 1, 2)
        self.bbox_face_graph.edge_attr = self.bbox_face_graph.edge_attr.permute(0, 2, 1)
        self.bbox_vertex_graph.edge_attr = self.bbox_vertex_graph.edge_attr.permute(0, 2, 1)


    def brep_complement(self, veno_graph, orig_graph):
        # 将original solid变成original solid complement
        # veno_graph的x: face normal取反，face point不变；
        # veno_graph的edge_attr: 连接节点a和b的边ab, ba的edge attr互相交换；之后，edge attr中的left face和right face normal取反
        # 最后，在veno_graph中加入新的外包围盒face adjacency graph，这个特征将作为一个bias，让模型可以学习到complement和original的
        # 模式区别；
        veno_graph.x[:, 3:6, ...] *= -1

        edge_index, edge_attr = sort_edge_index(veno_graph.edge_index, veno_graph.edge_attr)
        veno_graph.edge_index = torch.flip(edge_index, dims=[0])
        edge_attr[:, 6:, :-1] *= -1
        # 最后一维，边的凹凸性会发生变化，倒数第二维发生变化，其它属性不改变；
        edge_attr[:, -2, -1] = 1 - edge_attr[:, -2, -1]
        smooth_mask = edge_attr[:, 2, -1] == 0
        smooth_mask = smooth_mask[:, None].to(torch.long)
        reverse = torch.concatenate([smooth_mask, smooth_mask], dim=-1)
        edge_attr[:, :2, -1] = reverse - edge_attr[:, :2, -1]
        veno_graph.edge_attr = edge_attr

        # 将orig_graph的边特征已同样的方式取反，这样同时可以保证对偶关系的一致；
        edge_index, edge_attr = sort_edge_index(orig_graph.edge_index, orig_graph.edge_attr)
        orig_graph.edge_index = torch.flip(edge_index, dims=[0])
        edge_attr[:, 6:, :-1] *= -1
        # 最后一维，边的凹凸性会发生变化，倒数第二维发生变化，其它属性不改变；
        edge_attr[:, -2, -1] = 1 - edge_attr[:, -2, -1]
        smooth_mask = edge_attr[..., 2, -1] == 0
        smooth_mask = smooth_mask[:, None].to(torch.long)
        reverse = torch.concatenate([smooth_mask, smooth_mask], dim=-1)
        edge_attr[:, :2, -1] = reverse - edge_attr[:, :2, -1]
        orig_graph.edge_attr = edge_attr

        # veno_graph label, extrude side <=> cut side, extrude end <=> cut end,
        # revolve side/end <=> cut revolve side/end
        if len(self.CLASSES) == 8:
            extrude_2_cutside = (veno_graph.y == 0).to(torch.long)
            extrude_2_cutend = (veno_graph.y == 1).to(torch.long)
            cut_2_extrudeside = (veno_graph.y == 2).to(torch.long)
            cut_2_extrudeend = (veno_graph.y == 3).to(torch.long)
            reverse_label = extrude_2_cutside * 2 + extrude_2_cutend * 3 + cut_2_extrudeside * 0 + cut_2_extrudeend * 1
            not_reverse_mask = (extrude_2_cutside + extrude_2_cutend + cut_2_extrudeside + cut_2_extrudeend) == 0
            veno_graph.y = veno_graph.y * not_reverse_mask.to(torch.long) + reverse_label

        else:
            raise NotImplementedError("CC3D not implemented now.")

        # TODO: 将外包围盒的brep append到 veno graph和orig graph上；

        return veno_graph, orig_graph


    def Roll_Pitch_Yaw_Rotation(self, veno_graph, orig_graph):
        # 通过random指定在roll, pitch, yaw上的旋转角; 旋转角从90， 180， 270中选择
        rand = np.random.randint(4, size=3) * np.pi / 2
        alpha = rand[0]
        beta = rand[1]
        gamma = rand[2]

        def euler_to_rotation_matrix(alpha, beta, gamma):
            # 计算绕 Z 轴旋转矩阵
            Rz = np.array([
                [np.cos(alpha), -np.sin(alpha), 0],
                [np.sin(alpha), np.cos(alpha), 0],
                [0, 0, 1]
            ])

            # 计算绕 Y 轴旋转矩阵
            Ry = np.array([
                [np.cos(beta), 0, np.sin(beta)],
                [0, 1, 0],
                [-np.sin(beta), 0, np.cos(beta)]
            ])

            # 计算绕 X 轴旋转矩阵
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(gamma), -np.sin(gamma)],
                [0, np.sin(gamma), np.cos(gamma)]
            ])

            # 组合旋转矩阵
            return Rz @ Ry @ Rx

        R = torch.tensor(euler_to_rotation_matrix(alpha, beta, gamma)).to(torch.float32)
        fn, fu, fv, fc = veno_graph.x.shape
        face = veno_graph.x[..., :-1].view(fn, fu, fv, 3, -1)
        aug_face = torch.matmul(R, face)
        veno_graph.x = torch.cat([aug_face.view(fn, fu, fv, fc - 1), veno_graph.x[..., -1:]], dim=-1)

        en, eu, ec = veno_graph.edge_attr.shape
        vedge = veno_graph.edge_attr.view(en, eu, 3, -1)
        aug_vedge = torch.matmul(R, vedge[:, :-1, ...])
        aug_vedge = aug_vedge.view(en, eu - 1, ec)
        veno_graph.edge_attr = torch.cat([aug_vedge, veno_graph.edge_attr[:, -1:, :]], dim=1)

        vn, vc = orig_graph.x.shape
        vertex = orig_graph.x.view(vn, 3, -1)
        aug_vertex = torch.matmul(R, vertex)
        orig_graph.x = aug_vertex.view(vn, vc)

        oedge = orig_graph.edge_attr.view(en, eu, 3, -1)
        aug_oedge = torch.matmul(R, oedge[:, :-1, ...])
        aug_oedge = aug_oedge.view(en, eu - 1, ec)
        orig_graph.edge_attr = torch.cat([aug_oedge, orig_graph.edge_attr[:, -1:, :]], dim=1)

        return veno_graph, orig_graph


    def bounding_box_uvgrid(self, inp: torch.Tensor):
        pts = inp[..., :3].reshape((-1, 3))
        mask = inp[..., 6].reshape(-1)
        point_indices_inside_faces = mask == 1
        pts = pts[point_indices_inside_faces, :]
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        box = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
        return torch.tensor(box)


    def center_and_scale(self, veno_graph, orig_graph):
        bbox = self.bounding_box_uvgrid(veno_graph.x)
        diag = bbox[1] - bbox[0]
        scale = 2.0 / max(diag[0], diag[1], diag[2])
        center = 0.5 * (bbox[0] + bbox[1])

        # 将face_graph和vertex_graph中的属性归一化;
        veno_graph.x[..., :3] -= center
        veno_graph.x[..., :3] *= scale
        veno_graph.edge_attr[:, :-1, :3] -= center
        veno_graph.edge_attr[:, :-1, :3] *= scale

        orig_graph.x[..., :3] -= center
        orig_graph.x[..., :3] *= scale
        orig_graph.edge_attr[:, :-1, :3] -= center
        orig_graph.edge_attr[:, :-1, :3] *= scale

        veno_graph.x = veno_graph.x.permute(0, 3, 1, 2)
        veno_graph.edge_attr = veno_graph.edge_attr.permute(0, 2, 1)
        orig_graph.edge_attr = orig_graph.edge_attr.permute(0, 2, 1)
        return veno_graph, orig_graph


    def __getitem__(self, index):
        """
        :param index:
        :return:
        """
        solid_name = self.solid_names[index]
        src_data = self._load_source_graph(solid_name)
        vertex_graph, face_graph = self._gen_graph(solid_name, **src_data)

        if self.augmentation is not None:
            for aug in self.augmentation:
                face_graph, vertex_graph = aug(face_graph, vertex_graph)

        if self.transform is not None:
            # Optional: 加入point cloud特征，对应数据集加入point cloud模态;
            face_graph, vertex_graph = self.transform(face_graph, vertex_graph)

        # 按照某个概率，决定是否需要应用brep complement
        if np.random.rand(1) >= 1.0 and self.split == 'train':
            face_graph, vertex_graph = self.brep_complement(face_graph, vertex_graph)
            # 向face_graph和vertex_graph中加入外包围盒的face adjacency和vertex adjacency
            face_graph.concat_with_other_brep(self.bbox_face_graph)
            vertex_graph.concat_with_other_brep(self.bbox_vertex_graph)

        return vertex_graph, face_graph


    def _load_source_graph(self, solid_name):
        """
        :param brep_graph_pth: 存储了所有B-Rep提取的图。主要是包含了面、边、点的特征。其中面和边的特征是UV-grid密集采样的结果。
            若后续为了继续提升结果，可以考虑将点云的特征应用到模型中。根据UV-grid的点选取点云的近邻patch;
        :param face_label_pth: 存储了面的label，可以通过面的label，动态确定边的label；
        :param split:
        :return: dict():
            face_edges: list(tupel(edge_start_idx, edge_end_idx))
            vertex_edges:  list(tupel(edge_start_idx, edge_end_idx))
            face_attr: ndarray，包含face UV-grid sampling, shape = (len(face_AAG.nodes), u-grid, v-grid, channel)
            face_coedge_attr: ndarray，包含edge U-grid sampling, shape = (len(face_AAG.edges), u-grid, channel)
            vertex_attr: ndarray，包含vertex的空间坐标, shape = (len(vertex_AAG.nodes), 3)
            vertex_coedge_attr:  ndarray，包含edge U-grid sampling, shape = (len(vertex_AAG.edges), u-grid, channel)
            vf_mapping: vertex_AAG.edges(递增顺序) <=> face_AAG.edges
            fv_mapping: face_AAG.edges(递增顺序) <=> vertex_AAG.edges
            face_label: ndarray，包含face的 语义标签, shape = (len(face_AAG.nodes), 1)
            保证了face_AAG和vertex_AAG的边数量一致;
        """

        # face_attr: 包含face UV-grid sampling；edge_attr包含edge UV-grid sampling；Vertex_attr包含vertex的顶点坐标；
        result = self.solid_graphs[solid_name]
        return result


    def _gen_graph(self, solid_name, **kwargs):
        """
        根据source data 构建 primal and dual graph.

        参数参考 self._load_source_data(...)
        :param face_edges: 对应veno_graph.edges
        :param vertex_edges: 对应orig_graph.edges
        :param face_attr:
        :param edge_attr:
        :param vertex_attr:
        :param face_label:
        :param solid_name:
        :return:
            vertex_graph，DualGraphData，保存了基于brep顶点的图拓扑；
            face_graph，DualGraphData，保存了基于brep面的图拓扑；
        """

        # 根据 source data 构建 dual graph
        veno_edges = list(kwargs['face_edges'])
        veno_edge_idx = np.array(veno_edges)
        veno_edge_idx = veno_edge_idx[:, ::-1].copy()
        veno_edge_idx = torch.from_numpy(veno_edge_idx).t().to(torch.long)
        veno_edge_dual_idx = torch.from_numpy(kwargs['fv_mapping'][1]).to(torch.long).unsqueeze(1)
        veno_edge_attr = torch.from_numpy(kwargs['fcoedge_point_grids']).to(torch.float32)
        veno_node_attr = torch.from_numpy(kwargs['face_point_grids']).to(torch.float32)
        veno_node_label = torch.from_numpy(kwargs['fn_label']).to(torch.long)

        # 根据source data 处理 edge label，并构建 primal graph
        orig_edges = list(kwargs['vertex_edges'])
        orig_edge_idx = np.array(orig_edges)
        orig_edge_idx = orig_edge_idx[:, ::-1].copy()
        orig_edge_idx = torch.from_numpy(orig_edge_idx).t().to(torch.long)
        orig_edge_dual_idx = torch.from_numpy(kwargs['vf_mapping'][1]).to(torch.long).unsqueeze(1)
        orig_edge_attr = torch.from_numpy(kwargs['vcoedge_point_grids']).to(torch.float32)
        orig_edge_label = torch.from_numpy(kwargs['ve_label']).to(torch.long)
        orig_v_attr = torch.from_numpy(kwargs['vertex_point_grids']).to(torch.float32)

        # 根据source data f_node_mapping获取真实的节点个数
        f_node_mapping = np.array(list(kwargs['f_node_mapping'].values()))
        num_real_node = np.sum(f_node_mapping[:, 0] == f_node_mapping[:, 1])
        real_node_selection = np.hstack([
            np.ones(num_real_node), np.zeros(veno_node_attr.shape[0] - num_real_node)]).astype(np.bool)

        # edge_dual: 当前图的对偶图的edge_index, shape=(|E|_dual, 2)
        # edge_dual_idx: 当前图的边的对偶边在edge_dual中的序号; shape=(|E|, 1)
        # |E| 不一定和 |E|_dual相等
        face_graph = DualGraphData(
            x=veno_node_attr, y=veno_node_label,
            edge_index=veno_edge_idx, edge_attr=veno_edge_attr,
            edge_dual=orig_edge_idx.t(), edge_dual_idx=veno_edge_dual_idx,
            meta_data={
                'filename': solid_name, 'num_dual_edges': orig_edge_idx.shape[1],
                'num_dual_nodes': orig_v_attr.shape[0], 'node_area': np.ones(veno_node_attr.shape[0]).astype(np.float32),
                'node_real': real_node_selection
            }
        )
        vertex_graph = DualGraphData(
            x=orig_v_attr,
            edge_index=orig_edge_idx, edge_attr=orig_edge_attr, edge_label=orig_edge_label,
            edge_dual=veno_edge_idx.t(), edge_dual_idx=orig_edge_dual_idx,
            meta_data={
                'filename': solid_name, 'num_dual_edges': veno_edge_idx.shape[1],
                'num_dual_nodes': veno_node_attr.shape[0]
            }
        )
        if isinstance(self, CC3DDualGraphDataset):
            veno_node_step_label = torch.from_numpy(kwargs['fn_step_label']).to(torch.long)
            face_graph.y_step = veno_node_step_label
        # 由于vertex_graph和face_graph在构建时就确保了有向完全图，因此类似于dataset_floorplanSVG.py中self.dual_edge_mapping()来
        #   对对偶边的方向做调整;
        return vertex_graph, face_graph


    def __len__(self):
        return len(self.solid_names)



@register_dataset("CC3D_DUAL_DATASET")
class CC3DDualGraphDataset(BRepDualGraphDataset):
    CLASSES = ("ExtrudeSide", "ExtrudeEnd", "RevolveSide", "Fillet", "Chamfer", "other", "CutExtrudeSide", "CutExtrudeEnd",
               "RevolveEnd", "CutRevolveSide", "CutRevolveEnd")
    # 由于CutRevolveSide和 CutRevolveEnd的数据太稀少（加起来不足1%），
    # 因此跑一个UV_Reproduce观察对比结果之后，将CutRevolve和Revolve合并；
    def __init__(self, brep_graph_pth, json_file_pth, split='train', transform=None, augmentation=None):
        super().__init__(brep_graph_pth, json_file_pth, split, transform, augmentation)
        excluded_keys = [
            'User Library-Part3-10',   # NaN
            'User Library-Suelo Crilamyt',
            'User Library-Ensamblaje Propuesta_TODO Piso nivel 1',
            'User Library-catle rest_Wall',
            'User Library-Ensamblaje Actual_TODO Piso nivel 1',
            'User Library-Ensamblaje Propuesta_Suelo Crilamyt',
            'User Library-Part2-19',
            'User Library-Assem1-27_Part2'
            'User Library-catle rest_bridge',
            'User Library-catle rest_base1',
            'User Library-empirestate building felipe',
            'User Library-torreeiffel',
            'User Library-Part7_MainBlock-1',
            'User Library-Assem1-27_Part3',
            'User Library-tronco',
            'User Library-Part7_MainBlock',
            'User Library-TODO Piso nivel 1',
            'User Library-rrr_TODO Piso nivel 1',
            'User Library-Lycee_piece2',  # Inf
            'User Library-catle rest_bridge',
            'User Library-Pieza3-26',
            'User Library-Halle 1_Boden1',
            'User Library-Assem1-27_Part2',
        ]
        with open(json_file_pth, 'r') as f:
            json_file = json.load(f)
            solid_names = json_file[split]  # 根据data_split中的 train/valid/test获取数据序列
            solid_names = [solid_name.split('/')[-1] for solid_name in solid_names]
            self.solid_names = list(set(solid_names).intersection(self.solid_graphs.keys()))
            self.solid_names = list(set(self.solid_names).difference(set(excluded_keys)))

        self.merge_revolve = False


    def __getitem__(self, index):
        vertex_graph, face_graph = super().__getitem__(index)
        if self.merge_revolve:
            # 将CutRevolve和Revolve合并；
            face_label = face_graph.y
            cut_revolve_side = face_label == 9
            cut_revolve_end = face_label == 10
            merge_mask = torch.logical_not(torch.logical_or(cut_revolve_end, cut_revolve_side)).to(torch.long)
            merge_side = cut_revolve_side.to(torch.long) * 2
            merge_end = cut_revolve_end.to(torch.long) * 8
            merge_face_label = face_label * merge_mask + merge_side + merge_end
            face_graph.y = merge_face_label
        return vertex_graph, face_graph



@register_dataset("BREP_ZONE_DUAL_DATASET")
class BrepZoneDualGraphDataset(BRepDualGraphDataset):
    def __init__(self, brep_graph_pth, json_file_pth, split='train', transform=None, augmentation=None):
        super().__init__(brep_graph_pth, json_file_pth, split, transform, augmentation)
        zone_graph_pth = brep_graph_pth.replace('Fusion360_V1.pkl', 'Fusion360_ZoneGraph.h5')
        self.zone_graphs = h5py.File(zone_graph_pth, 'r')
        self.solid_names = list(set(self.solid_names).intersection(self.zone_graphs.keys()))


    def __getitem__(self, index):
        solid_name = self.solid_names[index]
        src_data = self._load_source_graph(solid_name)
        _, face_graph = self._gen_graph(solid_name, **src_data)
        src_zone_data = self._load_zone_graph(solid_name)
        face_graph, zone_graph, zone_x_graph = self._gen_zone_graph(
            solid_name, face_graph, f_node_mapping=src_data['f_node_mapping'], **src_zone_data
        )

        # TODO: 对vertex_graph，face_graph应用augmentation
        if self.augmentation is not None:
            pass

        # TODO: 对vertex_graph，face_graph应用不同的transform();
        if self.transform is not None:
            # Optional: 加入point cloud特征，对应数据集加入point cloud模态;
            zone_graph, zone_x_graph, face_graph = self.transform(zone_graph, zone_x_graph, face_graph)

        return zone_graph, zone_x_graph, face_graph


    def center_and_scale(self, orig_graph, orig_x_graph, veno_graph):
        """
        :param orig_graph:  这里指zone_graph
        :param orig_x_graph: 这里指为zone_graph中的x提供特征的图，即zone_x_graph
        :param orig_graph: 这里指face_graph
        :return:
        """
        bbox = self.bounding_box_uvgrid(veno_graph.x)
        diag = bbox[1] - bbox[0]
        scale = 2.0 / max(diag[0], diag[1], diag[2])
        center = 0.5 * (bbox[0] + bbox[1])

        # 将face_graph中的属性归一化;
        veno_graph.x[..., :3] -= center
        veno_graph.x[..., :3] *= scale
        veno_graph.edge_attr[:, :-1, :3] -= center
        veno_graph.edge_attr[:, :-1, :3] *= scale
        veno_graph.x = veno_graph.x.permute(0, 3, 1, 2)
        veno_graph.edge_attr = veno_graph.edge_attr.permute(0, 2, 1)

        # 将zone_graph和zone_x_graph的属性归一化
        # orig_graph.x[..., :3] -= center
        # orig_graph.x[..., :3] *= scale
        orig_x_graph.x[..., :3] -= center
        orig_x_graph.x[..., :3] *= scale
        orig_x_graph.x = orig_x_graph.x.permute(0, 3, 1, 2)

        return orig_graph, orig_x_graph, veno_graph


    def _load_zone_graph(self, solid_name):
        """
        dict key = solid_name
        value =
            zone_start: list(ndarray, shape = xx) 表示相邻的zone构造成图的边时，边的起始zone idx;
            zone_end: list(ndarray, shape = xx) 表示相邻的zone构造成图的边是，边的结束zone idx;
            dual_zone_face_mapping: dict (key=zone_idx, value=ndarray) 表示和original solid相邻的zones，
                与original solid哪一个face相邻，这是为了构造对偶分支网络时，设置对偶分支信息交互通路;
            zone_graph_feats: list(
                ndarray, shape = (num_zone_face, PLANE_ADJACENCY_SAMPLEINGS, PLANE_ADJACENCY_SAMPLEINGS, 7)
            ) 一个元素表示一个zone的特征，是zone的表面uv采样特征的合并;
            zone_graph_label: list(ndarray, shape = xx), 表示每一个zone的语义标签，若为-1，则表示该zone和 original solid不相邻
                若为实值，则表示该zone和 original solid相邻，为实值时shape不一定为1，因为一个zone可能和多个original solid face相邻；
                它们的face label不一定相同;
        """
        result = self.zone_graphs[solid_name]
        return result


    def _gen_zone_graph(self, solid_name, face_graph, **kwargs):
        zg_edges = kwargs['zone_edges'][:]
        zg_edges = zg_edges[::-1, :].copy()
        zg_edge_idx = torch.from_numpy(zg_edges).to(torch.long)

        # 利用一层图网络，将zone face attr汇集到zone attr上。为此，构造zone_x_graph
        zones_faces_attr = kwargs['zone_feats'][:]  # list
        zones_faces_attr = torch.from_numpy(zones_faces_attr).to(torch.float32)

        # merge_edge shape = (2, N)，将zone face的特征集中在一起，每一个zone形成一个特征向量
        merge_edge = torch.from_numpy(kwargs['zone_feats_merge'][:]).to(torch.long)
        zone_x_graph = BaseGraphData(x=zones_faces_attr, edge_index=merge_edge)

        # 将zone_label转化成one-hot向量，因为一个zone可能有多个label都符合;
        zone_label = kwargs['zone_label'][:]
        label_onehots = torch.from_numpy(zone_label).to(torch.float)
        label_onehots = label_onehots / torch.sum(label_onehots, dim=-1, keepdim=True)

        # zone_node_attr = (N, 1), 存疑，不知道应该用什么样的特征比较好;
        zone_node_attr = torch.zeros([label_onehots.shape[0], 1]).to(torch.float32)

        # zone_graph的zone,和face_graph的face之间建立对偶通路；
        #   zone_node_dual_idx: 表示当前zone node的对偶面在face_graph中的idx
        #   zone_node_idx: 表示当前zone node在zone_graph中的idx
        #   face_node_dual_idx: 表示当前face的对偶zone在zone_graph中的idx
        #   face_node_idx: 表示当前face node在face_graph中的idx
        temp = kwargs['fz_mapping'][:]
        face_node_dual_idx = torch.from_numpy(temp[1]).to(torch.long)
        face_node_idx = torch.from_numpy(temp[0]).to(torch.long)

        # TODO: face_graph为了构造完整的拓扑结构，存在面冗余，zf_mapping, fz_mapping并未考虑face_graph的冗余面，
        #   根据冗余面到真实面的映射关系构造冗余面的对偶关系，并添加到zf_mapping, fz_mapping之中；
        # 获得超出 real_face_num idx 的 face 的真实 face idx，并将它们和 zone 配对
        selected_fnis = []
        selected_fndis = []
        # 利用face_graph构造的信息 f_node_mapping，记录了face_graph节点idx和真实face idx的映射关系
        for k, v in kwargs['f_node_mapping'].items():
            cur_face_idx, real_face_idx = v[0], v[1]
            if cur_face_idx > real_face_idx:
                selected_fndi = face_node_dual_idx[face_node_idx == real_face_idx]
                selected_fnis.append(torch.ones_like(selected_fndi) * cur_face_idx)
                selected_fndis.append(selected_fndi)

        if len(selected_fnis) > 0:
            extra_face_node_idx = torch.cat(selected_fnis)
            extra_face_node_dual_idx = torch.cat(selected_fndis)
            face_node_idx = torch.cat([face_node_idx, extra_face_node_idx])
            face_node_dual_idx = torch.cat([face_node_dual_idx, extra_face_node_dual_idx])

            # 将face_node_dual_idx排序，就是zone_idx
            zone_order = torch.argsort(face_node_dual_idx)
            zone_node_idx = face_node_dual_idx[zone_order]
            zone_node_dual_idx = face_node_idx[zone_order]

        else:
            temp = kwargs['zf_mapping'][:]
            zone_node_dual_idx = torch.from_numpy(temp[1]).to(torch.long)
            zone_node_idx = torch.from_numpy(temp[0]).to(torch.long)

        zone_graph = DualGraphData(
            x=zone_node_attr, y=label_onehots,
            edge_index=zg_edge_idx, edge_attr=None,
            node_orig_idx=zone_node_idx, node_dual_idx=zone_node_dual_idx,
            meta_data={
                'filename': solid_name, 'num_dual_nodes': face_graph.x.shape[0]
            }
        )
        face_graph.node_orig_idx = face_node_idx
        face_graph.node_dual_idx = face_node_dual_idx
        # face_graph.__delattr__('edge_dual')
        # face_graph.__delattr__('edge_dual_idx')
        face_graph.meta_data['num_dual_nodes'] = zone_node_attr.shape[0]
        # face_graph.meta_data.__delitem__('num_dual_edges')

        return face_graph, zone_graph, zone_x_graph



# 根据occwl库对于BRep的解析，将UV-Net中基于dgl的数据结构迁移到pytorch_geometric上；
if __name__ == "__main__":
    # 统计数据集中aux_label的占比;
    from torch_geometric.graphgym.loader import get_loader, load_dataset
    from tqdm import tqdm

    # 测试 Fusion360数据
    # dataset = BRepDualGraphDataset(
    #     '/data1/ybc2021/Datasets/s2.0.1/UV_processed/Fusion360_V3.h5',
    #     '/data1/ybc2021/Datasets/s2.0.1/UV_processed/train_test.json'
    # )
    # dataloader = get_loader(dataset, 'full_batch', 1, shuffle=False)
    # label_boundary, label_all = 0, 0
    # for data in tqdm(dataloader):
    #     ori_graph, veno_graph = data
    #     veattr = veno_graph.edge_attr
    #     veattr_mapping_from_oe = ori_graph.edge_attr[veno_graph.edge_dual_idx]
    #     res = veattr - veattr_mapping_from_oe.squeeze()
    #
    #     # aux_label = ori_graph.edge_label.numpy()
    #     # label_boundary += np.sum(aux_label)  # 1的数量;
    #     # label_all += len(aux_label)
    #
    # print('{}, {}'.format(label_boundary, label_all))


    # 测试 Fusion360_zone数据
    # dataset = BrepZoneDualGraphDataset(
    #     '/data1/ybc2021/Datasets/s2.0.1/UV_processed/Fusion360_V2.pkl',
    #     '/data1/ybc2021/Datasets/s2.0.1/UV_processed/train_test.json'
    # )
    # dataloader = get_loader(dataset, 'full_batch', 4, shuffle=False)
    # for data in tqdm(dataloader):
    #     ori_graph, ori_x_graph, veno_graph = data
        # veattr = veno_graph.edge_attr
        # veattr_mapping_from_oe = ori_graph.edge_attr[veno_graph.edge_dual_idx]
        # res = veattr - veattr_mapping_from_oe.squeeze()

        # aux_label = ori_graph.edge_label.numpy()
        # label_boundary += np.sum(aux_label)  # 1的数量;
        # label_all += len(aux_label)


    # 测试CC3D数据
    dataset = CC3DDualGraphDataset(
        '/data/ybc2021/Datasets/CC3D/CC3D_V3.pkl',
        '/data/ybc2021/Datasets/CC3D/data_split.json',
        split='test'
    )
    dataloader = get_loader(dataset, 'full_batch', 1, shuffle=False)
    label_boundary, label_all = 0, 0
    for data in tqdm(dataloader):
        ori_graph, veno_graph = data
        if torch.isnan(veno_graph.x).any() or torch.isinf(veno_graph.x).any():
            print(veno_graph.meta_data['filename'])

        if torch.isnan(veno_graph.edge_attr).any() or torch.isinf(veno_graph.edge_attr).any():
            print(veno_graph.meta_data['filename'])



