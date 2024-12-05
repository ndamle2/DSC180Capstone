import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, GATv2Conv, SimpleConv
from torch.nn import Sequential as Seq, Linear, ReLU
import torch.nn.functional as F


class DEHNN_conv_layer(MessagePassing):
    def __init__(self, net_channels, channels):
        
        super().__init__(aggr='add')
        # initialize channels for hyperedge and node connections
        self.phi = Seq(Linear(channels, channels), ReLU(), Linear(channels, channels))
        self.psi = Seq(Linear(net_channels, net_channels), ReLU(), Linear(net_channels, net_channels))
        self.mlp = Seq(Linear(net_channels * 3, net_channels * 3), ReLU(), Linear(net_channels * 3, net_channels))

        # define and initialize other necessary layers for nodes and hyperedges
        self.conv = SimpleConv()
        self.node_bn = nn.BatchNorm1d(channels)
        self.hyperedge_bn = nn.BatchNorm1d(net_channels)
        self.back_conv = GATv2Conv(channels, channels)
        
    def forward(self, node, net, edge_index_source_to_net, edge_index_sink_to_net, edge_weight_sink_to_net): 
        # use batch normalization on node and hyperedge features
        node = self.node_bn(x)
        net = self.hyperedge_bn(net)
        
        h = self.phi(node)

        # convolution layer for source node and net features
        h_net_source = self.conv((h, net), edge_index_source_to_net)

        #propogate sink node messages through the graph
        h_net_sink = self.propagate(edge_index_sink_to_net, x=(h, net), edge_weight=edge_weight_sink_to_net)
        h_net_sink = self.psi(h_net_sink)
        
        # combine processed features then apply backward convolution
        h_net = self.mlp(torch.concat([net, h_net_source, h_net_sink], dim=1)) + x_net
        h = self.back_conv((h_net, h), torch.flip(edge_index_source_to_net, dims=[0])) + self.back_conv((h_net, h), torch.flip(edge_index_sink_to_net, dims=[0])) + h

        # return updated node and network features
        return h, h_net