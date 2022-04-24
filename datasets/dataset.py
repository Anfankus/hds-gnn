from abc import ABC, abstractproperty

from torch_geometric.datasets import Amazon, CitationFull, Coauthor, Planetoid
from torch_geometric.transforms import AddTrainValTestMask
from utils.config import get_hyper
from modules.dst import LRSPreprocess


class GraphDataset(ABC):
  @abstractproperty
  def meta(self): 
    pass

class DatasetManufactory():
  @staticmethod
  def getDataset(root_dir:str, dataset_name)->GraphDataset:
    pl_dataset = ['Cora',"CiteSeer","PubMed"]
    cite_dataset = ["DBLP"]
    amazon_dataset = ["Computers", "Photo"]
    coauthor_dataset = ["CS"]
    if dataset_name in pl_dataset:
        return PlanetoidDataset(root=root_dir, name=dataset_name)
    elif dataset_name in cite_dataset:
        return NodeRandomDataset(datacol_name= CitationFull, root=root_dir, name=dataset_name)
    elif dataset_name in amazon_dataset:
        return NodeRandomDataset(datacol_name= Amazon, root= root_dir, name = dataset_name)
    elif dataset_name in coauthor_dataset:
        return NodeRandomDataset(datacol_name= Coauthor, root= root_dir, name = dataset_name)


class NodeRandomDataset(GraphDataset):
    def __init__(self, datacol_name, root, name) -> None:
        super().__init__()
        self.origin_ds = datacol_name(root=root, name=name, transform=AddTrainValTestMask(split="random"))
    def preprocess(self):
        return LRSPreprocess(self.origin_ds, get_hyper().J, get_hyper().L, get_hyper().sct_type)[0]
    @property
    def meta(self):
      return {
        "name":self.origin_ds.name,
        "num_features":self.origin_ds.num_features,
        "num_classes":self.origin_ds.num_classes,
        "num_graphs":1,
      }

class PlanetoidDataset(GraphDataset):
    def __init__(self, root, name) -> None:
        super().__init__()
        self.origin_ds = Planetoid(root=root, name=name)
    def preprocess(self):
        return LRSPreprocess(self.origin_ds, get_hyper().J, get_hyper().L, get_hyper().sct_type)[0]
    @property
    def meta(self):
      return {
        "name":self.origin_ds.name,
        "num_features":self.origin_ds.num_features,
        "num_classes":self.origin_ds.num_classes,
        "num_graphs":1,
      }
