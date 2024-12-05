import torch
import math
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)

from dehnn_layers import HyperConvLayer

from torch_geometric.utils.dropout import dropout_edge

class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, out_node_dim, out_net_dim,
                        node_dim = None, 
                        net_dim = None, 
                        num_nodes = None,
                        device = 'cpu'
                    ):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''

        super(GNN_node, self).__init__()
        self.device = device

        self.num_layer = num_layer
        self.node_dim = node_dim
        self.net_dim = net_dim
        self.num_nodes = num_nodes
        self.emb_dim = emb_dim
        self.out_node_dim = out_node_dim
        self.out_net_dim = out_net_dim

        
        self.node_encoder = nn.Sequential(
                nn.Linear(node_dim, emb_dim),
                nn.LeakyReLU(negative_slope = 0.1),
                nn.Linear(emb_dim, emb_dim),
                nn.LeakyReLU(negative_slope = 0.1)
        )

        self.net_encoder = nn.Sequential(
                nn.Linear(net_dim, emb_dim),
                nn.LeakyReLU(negative_slope = 0.1)
        )
                
        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.virtualnode_embedding = torch.nn.Embedding(1, emb_dim)   
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)          

        self.mlp_virtualnode_list = torch.nn.ModuleList()

        self.virtualnode_to_local_list = torch.nn.ModuleList()

        for layer in range(num_layer - 1):
            self.mlp_virtualnode_list.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(emb_dim, emb_dim), 
                        torch.nn.LeakyReLU(negative_slope = 0.1),
                        torch.nn.Linear(emb_dim, emb_dim),
                        torch.nn.LeakyReLU(negative_slope = 0.1)
                    )
            )
        for layer in range(num_layer):
            self.convs.append(HyperConvLayer(emb_dim, emb_dim))

        self.fc1_node = torch.nn.Linear((self.num_layer + 1) * emb_dim, 256)
        self.fc2_node = torch.nn.Linear(256, self.out_node_dim)

        self.fc1_net = torch.nn.Linear((self.num_layer + 1) * emb_dim, 64)
        self.fc2_net = torch.nn.Linear(64, self.out_net_dim)
        

    def forward(self, data, device):
        node_sink_net_index, node_sink_net_weight = gcn_norm(data.edge_index_sink_to_net, add_self_loops=False)
        node_features, net_features, edge_index_sink_to_net, edge_weight_sink_to_net, edge_index_source_to_net, batch, num_vn = data['node_features'].to(device), data['net_features'].to(device), node_sink_net_index, node_sink_net_weight, data.edge_index_source_to_net.to(device), data.batch.to(device), data.num_vn

        edge_index_sink_to_net, edge_mask = dropout_edge(edge_index_sink_to_net, p = 0.4)
        edge_index_sink_to_net = edge_index_sink_to_net.to(device)
        edge_weight_sink_to_net = edge_weight_sink_to_net[edge_mask].to(device)
        
        num_instances = node_features.shape[0]
        
        h_list = [self.node_encoder(node_features)]
        h_net_list = [self.net_encoder(net_features)]

        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(num_vn).to(batch.dtype).to(batch.device))

        for layer in range(self.num_layer):
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]
            
            h_inst, h_net = self.convs[layer](h_list[layer], h_net_list[layer], edge_index_source_to_net, edge_index_sink_to_net, edge_weight_sink_to_net)
            h_list.append(h_inst)
            h_net_list.append(h_net)

            if (layer < self.num_layer - 1):
                virtualnode_embedding_temp = global_mean_pool(h_list[layer], batch) + virtualnode_embedding
                virtualnode_embedding = virtualnode_embedding + self.mlp_virtualnode_list[layer](virtualnode_embedding_temp)
        
        node_representation = torch.cat(h_list, dim = 1)
        net_representation = torch.cat(h_net_list, dim = 1)

        node_representation = torch.nn.functional.leaky_relu(self.fc2_node(torch.nn.functional.leaky_relu(self.fc1_node(node_representation), negative_slope = 0.1)), negative_slope = 0.1)
        net_representation = torch.abs(torch.nn.functional.leaky_relu(self.fc2_net(torch.nn.functional.leaky_relu(self.fc1_net(net_representation), negative_slope = 0.1)), negative_slope = 0.1))

        return node_representation, net_representation
        