import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tnn
import math
from torch.nn.parameter import Parameter
from torch.nn import init
from utils.config import get_hyper

def backbone(name:str, in_features):
  if name.startswith("GCN"):
    return tnn.GCNConv(in_channels=in_features, out_channels= get_hyper().nhid)
  elif name.startswith("GAT"):
    return tnn.GATConv(in_channels=in_features, out_channels= get_hyper().nhid, heads=3, concat=False, dropout=get_hyper().dropout)
  elif name.startswith(("Sage", "SAGE")):
    return tnn.SAGEConv(in_channels=in_features, out_channels= get_hyper().nhid)
  elif name.startswith("GIN"):
    return tnn.GINConv(nn.Sequential(
        nn.Linear(in_features, get_hyper().nhid),
        nn.BatchNorm1d(get_hyper().nhid),
        nn.ReLU(inplace=True),
        nn.Linear(get_hyper().nhid, get_hyper().nhid),
    ))
  else:
    raise RuntimeError("unknown backbone")

class HDSGNN(nn.Module):
  def __init__(self, num_features, num_classes, gnn):
    super(HDSGNN, self).__init__()
    self.num_features = num_features
    self.num_classes = num_classes
    self.layers = get_hyper().L
    self.orders = get_hyper().J
    self.dropout = get_hyper().dropout

    conv_list = []
    linear_list = []
    in_features = num_features
    for i in range(self.layers):
      conv_list.append(backbone(gnn,in_features))
      linear_list.append(nn.Linear(in_features=self.orders ** i * num_features, out_features= get_hyper().nhid))
      in_features = get_hyper().nhid * 2
    self.convs = nn.ModuleList(conv_list)
    self.lins = nn.ModuleList(linear_list)
    self.order_weights = Parameter(torch.Tensor(self.orders))
    self.conv_cls = tnn.GCNConv(in_channels=in_features, out_channels=num_classes)

    self.reset_parameters()

  def reset_parameters(self):
    bound = 1 / math.sqrt(self.order_weights.size(0))
    init.uniform_(self.order_weights, -bound, bound)

  def forward(self, data):
    x, edge_index, feature = data.x, data.edge_index, data.features
    begin_index = 0
    comb_feature = x
    for i in range(self.layers):
      convx = F.relu(self.convs[i](comb_feature, edge_index))
      convx = F.dropout(convx, self.dropout, self.training)

      count = self.orders ** i
      layer_feature = feature[begin_index: begin_index+count]
      if i > 0:
        order_weight = self.order_weights.repeat(self.orders ** (i-1)).view(-1,1,1)
        layer_feature = order_weight * layer_feature
      layer_feature = torch.moveaxis(layer_feature,0,1)
      layer_feature = torch.reshape(layer_feature, (layer_feature.shape[0],-1))
      
      lin_sct = F.relu(self.lins[i](layer_feature))
      lin_sct = F.dropout(lin_sct, self.dropout, self.training)

      comb_feature = torch.cat([lin_sct, convx],dim=-1)
      begin_index += count
      
    convx = self.conv_cls(comb_feature, edge_index)
    return F.log_softmax(convx, dim=-1), convx

class HDSGNN_L(nn.Module):
  def __init__(self, num_features, num_classes, gnn):
    super(HDSGNN_L, self).__init__()
    self.num_features = num_features
    self.num_classes = num_classes
    self.layers = get_hyper().L
    self.orders = get_hyper().J
    self.dropout = get_hyper().dropout

    conv_list = []
    linear_list = []
    order_weights = []
    in_features = num_features
    for i in range(self.layers):
      conv_list.append(backbone(gnn,in_features))
      linear_list.append(nn.Linear(in_features=self.orders ** i * num_features, out_features= get_hyper().nhid))
      order_weights.append(Parameter(torch.Tensor(self.orders)))
      in_features = get_hyper().nhid * 2
    self.convs = nn.ModuleList(conv_list)
    self.lins = nn.ModuleList(linear_list)
    self.order_weights = nn.ParameterList(order_weights)
    self.conv_cls = tnn.GCNConv(in_channels=in_features, out_channels=num_classes)

    self.reset_parameters()

  def reset_parameters(self):
    for weight in self.order_weights:
      bound = 1 / math.sqrt(weight.size(0))
      init.uniform_(weight, -bound, bound)

  def forward(self, data):
    x, edge_index, feature = data.x, data.edge_index, data.features
    begin_index = 0
    comb_feature = x
    for i in range(self.layers):
      convx = F.relu(self.convs[i](comb_feature, edge_index))
      convx = F.dropout(convx, self.dropout, self.training)

      count = self.orders ** i
      layer_feature = feature[begin_index: begin_index+count]
      if i > 0:
        order_weight = self.order_weights[i].repeat(self.orders ** (i-1)).view(-1,1,1)
        layer_feature = order_weight * layer_feature
      layer_feature = torch.moveaxis(layer_feature,0,1)
      layer_feature = torch.reshape(layer_feature, (layer_feature.shape[0],-1))
      
      lin_sct = F.relu(self.lins[i](layer_feature))
      lin_sct = F.dropout(lin_sct, self.dropout, self.training)

      comb_feature = torch.cat([lin_sct, convx],dim=-1)
      begin_index += count
      
    convx = self.conv_cls(comb_feature, edge_index)
    return F.log_softmax(convx, dim=-1)
