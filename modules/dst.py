import torch
import torch_geometric.utils as tgu
from utils.adjacency import diffusionFilterBank
def LRSPreprocess(datalist, J, L=-1, scattering_type = "diffusion"):
  datalist_ = []

  for i in datalist:
    if tgu.contains_isolated_nodes(i.edge_index, num_nodes=i.x.size(0)):
      adj = tgu.to_dense_adj(edge_index = i.edge_index, max_num_nodes=i.x.size(0))[0]
      
      deg = torch.sum(adj, dim=1)
      no_deg = (deg == 0)
      no_deg_index = no_deg.nonzero(as_tuple=True)[0]
      ones = torch.eye(adj.size(0), device=i.edge_index.device)
      adj[no_deg_index] = ones[no_deg_index]
      deg = torch.sum(adj, dim=1)

      i.edge_index, _ = tgu.dense_to_sparse(adj)


    x,edge_index = i.x, i.edge_index
    # normalize adj
    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    # edge_index, edge_weight = tgu.add_self_loops(edge_index=edge_index, edge_weight=edge_weight, num_nodes=x.size(0))
    row, col = edge_index
    deg = tgu.degree(index=col, num_nodes=x.size(0), dtype=x.dtype)

    adj = tgu.to_dense_adj(edge_index = edge_index, max_num_nodes=x.size(0))[0]
    if scattering_type == "diffusion":
      # T = 0.5 * (AD^-1 + I)
      deg_neg1 = torch.diag(deg.pow(-1.))
      adj_norm = adj @ deg_neg1 
      I_n = torch.eye(adj_norm.shape[0], device=edge_index.device)
      T = 0.5 * (adj_norm + I_n)
    elif scattering_type == "geometric":
      # T = 0.5 * (D^-1/2 A D^-1/2  + I )
      deg_neg_sqrt = torch.diag(deg.pow(-0.5))
      adj_norm = deg_neg_sqrt @ adj @ deg_neg_sqrt
      I_n = torch.eye(adj_norm.shape[0], device=edge_index.device)
      T = 0.5 * (adj_norm + I_n)
    else:
      raise NameError("未知散射类型：", scattering_type)

    # scattering
    i.Psi = [diffusionFilterBank(T, J)]
    i.Phi = [deg / torch.norm(deg, 1)]

    # transform
    if L > 0:
      phi = torch.full(size=(1,x.size(0)), fill_value=1/x.size(0))    # 1 x n
      i.sct_feat, i.features = LRSTransform(x, i.Psi[0], torch.abs, phi, L, J)
    datalist_.append(i)
  return datalist_

def LRSTransform(x, H, rho, phi, L, J):
  assert callable(rho)
  if phi is None:
    phi = torch.eye(x.shape[0], device=x.device)
  feature0 = phi @ x  # 1 x d

  graph_reps = [feature0]
  node_reps = [x.unsqueeze(dim = 0)]
  last_hx = [x]
  for l in range(L - 1):
    new_hx = []
    for lastIndex in range(J ** l):
      newHX = rho( H @ last_hx[lastIndex])  # J x N x d
      new_feature = phi @ newHX             # J x d
      new_hx.append(newHX)
      node_reps.append(newHX)
      graph_reps.append(new_feature.squeeze(dim=-2))
    last_hx = torch.cat(new_hx,dim=0)

  graph_features = torch.cat(graph_reps, dim=0)
  node_features = torch.cat(node_reps, dim=0)
  return graph_features, node_features
