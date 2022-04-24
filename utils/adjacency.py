import torch

"""
    T : NxN
    H : JxNxN
"""
def diffusionFilterBank(T:torch.Tensor, J:int):
    I = torch.eye(T.shape[0], device=T.device)
    H = I - T
    HList = [H]
    for j in range(0, J-1):
        p = 2 ** j
        baseT = torch.matrix_power(T, p)
        newH = baseT @ (I - baseT)
        HList.append(newH)
    H = torch.stack(HList)
    return H
    
def diffusionFilterBank_(T:torch.Tensor, J:int):
    I = torch.eye(T.shape[0], device=T.device)
    H = (I - T).unsqueeze_(0)

    # HList = [H]
    for j in range(0, J-1):
        p = 2 ** j
        baseT = torch.matrix_power(T, p)
        baseT = (baseT @ (I - baseT)).unsqueeze_(0)
        H = torch.cat((H, baseT), dim=0)
        # HList.append(newH)
    # H = torch.stack(HList)
    return H

def scattering_2(adj:torch.Tensor, order:int):
    I_n = torch.eye(adj.shape[0], device=adj.device)
    adj_sct = 0.5 * (adj + I_n)
    adj_power = adj_sct
    if order > 1:
        for i in range(order - 1):
            adj_power = torch.mm(adj_power, adj_sct)
        adj_int = torch.mm((adj_power - I_n), adj_power)
    else:
        adj_int = torch.mm((adj_power - I_n), adj_power)
    return adj_int

