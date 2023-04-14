import torch.nn as nn
import numpy as np
from torch_geometric.nn import GATConv
import torch

import time
from torch import optim
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATConv, GATv2Conv, PNAConv
import torchvision
from sklearn import metrics
import os
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GraphTransformer(nn.Module):

    def __init__(self, deg, num_atom_feat=3, num_atom_type=100, batch_size=2, dim_node=32, out_dim=10, n_bond_type=20,
                 score_fn='trans', edge_dim=32, dropout=0.1, n_side_effects=964, dim_mlp=32, hid_dim=32, heads=4,
                 depth=2):
        super().__init__()
        self.deg = deg
        self.batch_size = batch_size
        self.dim_node = dim_node
        self.num_atom_type = num_atom_type
        self.num_atom_feat = num_atom_feat
        self.out_dim = out_dim
        self.n_bond_type = n_bond_type
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.num_side_effects = n_side_effects
        self.dim_mlp = dim_mlp
        self.heads = heads
        self.depth = depth
        self.hid_dim = hid_dim

        # self.atom_type_emb = nn.Embedding(self.num_atom_type, self.dim_node, padding_idx=0)
        # nn.init.xavier_normal_(self.atom_type_emb.weight)
        self.atom_node = nn.Linear(self.dim_node + self.num_atom_feat, self.dim_node)

        self.cls_nodes = nn.Parameter(torch.randn(self.batch_size, 4))

        self.atom_proj = nn.Linear(self.dim_node + self.num_atom_feat, self.dim_node)
        self.atom_emb = nn.Embedding(self.num_atom_type + 1, self.dim_node, padding_idx=0)
        self.bond_emb = nn.Embedding(self.n_bond_type + 2, self.edge_dim, padding_idx=0)
        nn.init.xavier_normal_(self.atom_emb.weight)
        nn.init.xavier_normal_(self.bond_emb.weight)

        self.side_effect_emb = None
        if self.num_side_effects is not None:
            self.side_effect_emb = nn.Embedding(self.num_side_effects, self.hid_dim)
            nn.init.xavier_normal_(self.side_effect_emb.weight)

        self.classify = nn.Sigmoid()
        self.final = nn.Linear(self.dim_node, self.dim_node)

        # self.gat_transformer = GATTransformer(dim=self.dim_node, depth=2, heads=4, mlp_dim=self.dim_node,
        #                                       edge_dim=self.edge_dim)

        self.emb_dropout = nn.Dropout(p=self.dropout)

        self.__score_fn = score_fn

        # self.encoder_blocks = nn.ModuleList([])
        # for _ in range(self.depth):
        #     self.encoder_blocks.append(GATTransformerEncoder(dim=self.dim_node, dim_edge=self.edge_dim,
        #                                                      dim_mlp=self.dim_mlp,dropout=self.dropout,
        #                                                      heads=self.heads))
        # self.gat_transformer = GATTransformerEncoder(dim=self.dim_node, dim_edge=self.edge_dim,
        #                                              dim_mlp=self.dim_mlp, dropout=0,
        #                                              heads=self.heads)
        self.pna_transformer = PNATransformerEncoder(dim=self.dim_node, dim_mlp=dim_mlp, dim_edge=self.edge_dim,
                                                     dropout=0, deg=self.deg)

    @property
    def score_fn(self):
        return self.__score_fn

    def forward(self,
                seg_m1, atom_type1, atom_feat1, bond_type1,
                inn_seg_i1, inn_idx_j1, out_seg_i1, out_idx_j1,
                seg_m2, atom_type2, atom_feat2, bond_type2,
                inn_seg_i2, inn_idx_j2, out_seg_i2, out_idx_j2,
                se_idx=None, drug_se_seg=None, entropies=[]):
        #  Create a single graph from both left and right graphs
        adj_list_1 = torch.cat((inn_seg_i1, inn_seg_i2 + atom_type1.shape[0]))
        adj_list_2 = torch.cat((inn_idx_j1, inn_idx_j2 + atom_type1.shape[0]))
        #  Add connect left and right molecule graphs with a virtual node. Take batches into account
        #  Variable seg is just an indicator for the nodes that belong to each graph
        seg = torch.cat((seg_m1, seg_m2))
        #  Size of each graph is calculated so we can use this info to generate adj_list values for classification token
        graph_sizes = torch.bincount(seg)

        # Add attention edges between pairs of molecules
        # use existing outer indices
        att_beg = torch.cat((out_seg_i1, out_seg_i2 + atom_feat1.shape[0]))
        att_dest = torch.cat((out_idx_j1 + atom_feat1.shape[0], out_idx_j2))
        att_feats = torch.tensor(np.full((att_beg.shape[0]), self.n_bond_type, dtype=np.int64), device='cuda:0',
                                 dtype=torch.int64)

        #  Create classification token adjacency index values
        cls_beg_idx = seg.shape[0]
        cls_end_idx = cls_beg_idx + seg.unique().shape[0]
        rep = torch.arange(cls_beg_idx, cls_end_idx, device='cuda:0')
        # if rep.shape[0] != graph_sizes.shape[0]:
        #     print(f'rep {rep.shape[0]} {rep}\ngraph_sizes {graph_sizes.shape[0]} {graph_sizes}\ndata {atom_type1.shape} {atom_type2.shape} ')
        #     return None
        # adj_clt = torch.tensor(data=torch.repeat_interleave(rep, repeats=graph_sizes))
        adj_clt = torch.repeat_interleave(rep, repeats=graph_sizes)
        node_range = torch.arange(0, cls_beg_idx, device='cuda:0')
        beg = torch.cat((node_range, adj_clt, rep))
        dest = torch.cat((adj_clt, node_range, rep))
        a = torch.cat((adj_list_1, att_beg, beg))
        b = torch.cat((adj_list_2, att_dest, dest))
        adj_list = torch.vstack((a, b))

        # x = self.atom_comp(atom_feat1, atom_type1)
        cls_nodes = torch.randn(seg.unique().shape[0], 3, device='cuda:0')
        cls_types = torch.repeat_interleave(torch.tensor([self.num_atom_type], dtype=torch.int64, device='cuda:0'),
                                            seg.unique().shape[0])
        x = self.process_atoms(atom_feat1, atom_feat2, atom_type1, atom_type2, cls_nodes, cls_types)
        x = self.emb_dropout(x)
        # Edge data
        cls_edge_feats = torch.tensor(np.full((beg.shape[0]), self.n_bond_type + 1.0, dtype=np.int64), device='cuda:0',
                                      dtype=torch.int64)

        edge_feats = torch.cat((bond_type1, bond_type2, att_feats, cls_edge_feats))
        emb_edges = self.bond_emb(edge_feats)
        emb_edges = self.emb_dropout(emb_edges)
        # Embedding data

        data = (x, adj_list, emb_edges)
        # data = (x, adj_list)
        res = self.pna_transformer(data)

        # res = None
        # for enc_block in self.encoder_blocks:
        #     res = enc_block(data)

        cls_tokens = res[x.shape[0] - seg.unique().shape[0]:]

        # treat the label that is being checked as a relation
        se_vec = self.emb_dropout(self.side_effect_emb(se_idx))
        cls_tokens = cls_tokens.index_select(0, drug_se_seg)
        res = self.final(cls_tokens).squeeze()
        res = torch.norm(res + se_vec, dim=1)
        return res

    def process_atoms(self, atom_feat1, atom_feat2, atom_type1, atom_type2, cls_feat, cls_type):
        atom_feat = torch.vstack((atom_feat1, atom_feat2, cls_feat))
        atom_type = torch.cat((atom_type1, atom_type2, cls_type))
        emb = self.atom_emb(atom_type)
        # e2 = self.atom_type_emb(atom_type2)
        # a = torch.cat((atom_feat, atom_type.unsqueeze(1)), -1)
        # atom = torch.vstack((a, cls))

        return self.atom_proj(torch.cat((atom_feat, emb), -1))

    def atom_comp(self, atom_feat, atom_idx):
        atom_emb = self.atom_emb(atom_idx)
        node = self.atom_proj(torch.cat([atom_emb, atom_feat], -1))
        return node


class ResidualConnection(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PostAttentionMultiHeadProjection(nn.Module):
    def __init__(self, dim, heads, fn):
        super().__init__()
        self.linear = nn.Linear(dim * heads, dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.linear(self.fn(x, **kwargs))


class GraphEncoderMLP(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, data):
        return self.mlp(data)


class PreLayerNorm(nn.Module):
    def __init__(self, layer_dim, func):
        super().__init__()
        self.func = func
        self.l_norm = nn.LayerNorm(layer_dim)

    def forward(self, data, **kwargs):
        return self.func(self.l_norm(data), **kwargs)


class GATTransformerEncoder(nn.Module):

    def __init__(self, dim, heads, dim_edge, dropout, dim_mlp, depth=4):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ResidualConnection(
                    PreLayerNorm(dim, PostAttentionMultiHeadProjection(dim=dim, heads=heads,
                                                                       fn=GATv2Conv(in_channels=dim, out_channels=dim,
                                                                                    heads=heads, add_self_loops=False,
                                                                                    edge_dim=dim_edge,
                                                                                    dropout=dropout)))),
                ResidualConnection(
                    PreLayerNorm(dim, GraphEncoderMLP(in_dim=dim, hid_dim=dim_mlp, out_dim=dim, dropout=dropout)))
            ]))

    def forward(self, data):
        x, edge_index, edge_attr = data
        for att, ff in self.layers:
            x = att(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = ff(x=x)
        return x


class PNATransformerEncoder(nn.Module):

    def __init__(self, dim, dim_edge, dropout, dim_mlp, deg, aggregators=None, scalers=None, towers=4, depth=4,
                 divide_input=False, pre_layers=1, post_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.aggregators = aggregators or ['mean', 'min', 'max', 'std']
        self.scalers = scalers or ['identity', 'amplification', 'attenuation']
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ResidualConnection(
                    PreLayerNorm(dim, PNAConv(in_channels=dim, out_channels=dim,
                                              aggregators=self.aggregators,
                                              scalers=self.scalers,
                                              towers=towers,
                                              edge_dim=dim_edge,
                                              dropout=dropout,
                                              deg=deg,
                                              divide_input=divide_input,
                                              pre_layers=pre_layers,
                                              post_layers=post_layers))),
                ResidualConnection(
                    PreLayerNorm(dim, GraphEncoderMLP(in_dim=dim, hid_dim=dim_mlp, out_dim=dim, dropout=dropout)))
            ]))

    def forward(self, data):
        x, edge_index, edge_attr = data
        for att, ff in self.layers:
            x = att(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = ff(x=x)
        return x
