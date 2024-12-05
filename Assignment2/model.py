import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool
from dehnn_layer import DEHNN_conv_layer
from torch_geometric.utils.dropout import dropout_edge
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class DEHNN(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, out_dim, node_dim, net_dim, num_nodes):
        #initialize variables (specify hyperparameters of model, eg. num layers)
        super(DEHNN, self).__init__()
        self.device = "cpu"
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.out_dim = 1
        self.node_dim = node_dim
        self.net_dim = net_dim
        self.num_nodes = num_nodes

        # layer to transform nodes/nets to hidden layer dimensions
        self.node = nn.Sequential(nn.Linear(node_dim, emb_dim), nn.ReLU())
        self.net = nn.Sequential(nn.Linear(net_dim, emb_dim), nn.ReLU())
                
        # create and initialize virtual node embedding layer and related modules
        self.vn_embedding = torch.nn.Embedding(1, emb_dim)   
        torch.nn.init.constant_(self.vn_embedding.weight.data, 0)          
        self.vn_mlp = torch.nn.ModuleList()

        # build MLP for hidden layers of virtual nodes 
        for layer in range(num_layers - 1):
            self.vn_mlp.append(torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU()))

        # creates instances of message-passing layer for virtual nodes (aggregation of long range interactions)
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(DEHNN_conv_layer(emb_dim, emb_dim))

        # final layers to reduce dimensionality to output value
        self.fc1 = torch.nn.Linear((self.num_layers + 1) * emb_dim, 256)
        self.fc2 = torch.nn.Linear(256, self.out_node_dim)
        self.output = nn.Linear(self.node_dim, 1)


    

    def forward(self, data, device):
        # normalize the edge index for the sink-to-net connections
        node_sink_net_index, node_sink_net_weight = gcn_norm(data.edge_index_sink_to_net, add_self_loops=False)

        # move the input features to device (cpu as currently written)
        node_features = data['node_features'].to(device)
        net_features = data['net_features'].to(device)

        # prepare edge indices and weights, move to device
        index_sink2net = node_sink_net_index
        weight_sink2net = node_sink_net_weight
        index_source2net = data.edge_index_source_to_net.to(device)
        batch = data.batch.to(device)
        num_vn = data.num_vn

        # apply dropout of edges to avoid overfitting (p reduced from original implementation)
        index_sink2net, edge_mask = dropout_edge(index_sink2net, p = 0.1)
        index_sink2net = index_sink2net.to(device)
        weight_sink2net = weight_sink2net[edge_mask].to(device)

        # intialize node, net, and virtual node feature lists
        node_list = [self.node(node_features)]
        net_list = [self.net(net_features)]
        vn_embedding = self.vn_embedding(torch.zeros(num_vn).to(batch.dtype).to(batch.device))
        
        # iterate over message passing layers to update virtual nodes
        for layer in range(self.num_layers):
            node_list[layer] = node_list[layer] + vn_embedding[batch]
            # update virtual nodes with DEHNN message passing layer
            node_list, h_net = self.convs[layer](node_list[layer], net_list[layer], index_source2net, index_sink2net, weight_sink2net)
            node_list.append(h_inst)
            net_list.append(h_net)

            # update virtual nodes with MLP
            if (layer < self.num_layers - 1):
                temp = global_mean_pool(node_list[layer], batch) + vn_embedding
                vn_embedding = vn_embedding + self.vn_mlp[layer](temp)

        # concatenate all node features from all layers and pass through fc layers
        nodes_cat = torch.cat(node_list, dim = 1)
        nodes = torch.nn.functional.relu(self.fc2_node(torch.nn.functional.relu(self.fc1_node(node_representation))))

        # compute final output and return output and nodes
        ret = self.output(nodes_cat)
        return nodes, ret