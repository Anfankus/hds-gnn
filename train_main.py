import random
import argparse
import os
import time
import torch
import numpy as np

from datasets.dataset import GraphDataset, DatasetManufactory
import modules.model as m

import torch.nn.functional as F
from utils.config import load_config
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from utils.config import get_hyper

import time
import hashlib
class Launcher():
  def __init__(self, model, writer:SummaryWriter, save_root):
    self.args = get_hyper()
    self.model = model
    self.save_root = save_root
    self.writer = writer
    self.epochs = self.args.epochs
    self.patience = self.args.patience
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.patience, factor=self.args.factor, verbose=True)
    self.stop_patience = self.args.stop_patience

  def train_epoch(self, data:Data):
    self.model.train()
    self.optimizer.zero_grad(set_to_none=True)

    out = self.model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    # loss = F.cross_entropy(out, data.y)
    loss.backward()
    self.optimizer.step()

    _, pred = out.max(dim=1)
    correct = int(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
    acc = correct / int(data.train_mask.sum())

    return acc, loss

  def test(self, data:Data, mask):
    self.model.eval()
    with torch.no_grad():
        out = self.model(data)
        loss = F.nll_loss(out[mask], data.y[mask])
        _, pred = out.max(dim=1)
        correct = int(pred[mask].eq(data.y[mask]).sum().item())
        acc = correct / int(mask.sum())
        # loss_test += F.cross_entropy(out, data.y).item()
    return acc, loss


  def train(self, data:Data, filename = "ckpt", prefix = "", loop = 0):
    best_filename = os.path.join(self.save_root, f"{filename}-{loop}.pth")
    temp_filename = os.path.join(self.save_root, f"{filename}-{loop}-temp.pth")

    t = time.time()
    best_val_loss = np.inf
    best_val_acc = np.inf
    best_train_acc = np.inf
    best_train_loss = np.inf

    patience = 0
    for epoch in range(self.epochs):
      acc_train, loss_train = self.train_epoch(data)
      acc_val, loss_val = self.test(data, data.val_mask)
      self.scheduler.step(loss_val)

      print(  '({:s}-{:d}) {:d}'.format(prefix,loop,patience),
              'Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
              'acc_train: {:.6f}'.format(acc_train), 'loss_val: {:.6f}'.format(loss_val),
              'acc_val: {:.6f}'.format(acc_val), 'time: {:.6f}s'.format(time.time() - t))
      self.writer.add_scalars(f"loss-{prefix}/{loop}",{
        "train":loss_train, "val":loss_val
      },epoch)
      self.writer.add_scalars(f"acc-{prefix}/{loop}",{
        "train":acc_train, "val":acc_val
      },epoch)

      torch.save(self.model.state_dict(), temp_filename)

      patience += 1
      if patience > self.stop_patience:
        break
      if loss_val < best_val_loss:
          best_val_loss = loss_val
          best_val_acc = acc_val
          best_train_loss = loss_train
          best_train_acc = acc_train
          patience = 0

          torch.save(self.model.state_dict(), best_filename)
          print("best loss model saved {:f}, {:f} - {:s}".format(best_val_loss, best_val_acc, best_filename))
    return best_train_acc, best_train_loss, best_val_acc, best_val_loss

  def load_best(self, filename="ckpt", loop = 0):
    best_filename = os.path.join(self.save_root, f"{filename}-{loop}.pth")
    self.model.load_state_dict(torch.load(best_filename))
  def load_latest(self, filename="ckpt", loop = 0):
    latest_filename = os.path.join(self.save_root, f"{filename}-{loop}-temp.pth")
    self.model.load_state_dict(torch.load(latest_filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True, type=str)
    args = parser.parse_args()
    hash_tag = hashlib.sha256(str(args.config).encode("UTF-8")).hexdigest()
    config = load_config(args.config)

    torch.manual_seed(666)
    torch.cuda.manual_seed_all(666)
    np.random.seed(666)
    random.seed(666)
    dataset:GraphDataset = DatasetManufactory.getDataset(root_dir="datasets", dataset_name=config.dataset)
    data = dataset.preprocess()
    data = data.cuda()

    save_root = os.path.join("test","ckpt",config['model'],config['sct_type'],config['tag'])
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    

    model_constructor = m.HDSGNN

    train_list = []
    val_list = []
    test_list = []
    best_list = []

    seed = config.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"seed: {str(seed)}")
    begin = time.time()
    for i in range(config.loop_round):
        print(config)

        ckpt_filename = config.dataset
        writer = SummaryWriter(log_dir=f"log/tensorboard")


        model_train = model_constructor(
          num_features=dataset.meta['num_features'],
          num_classes=dataset.meta['num_classes'],
          gnn = config.tag
        ).to("cuda:0")
        trainer = Launcher(model=model_train,save_root=save_root, writer = writer)
        acc_train, loss_train, acc_val, loss_val = trainer.train(
            data, filename=ckpt_filename, 
            prefix=f"{config.model},{config.dataset},{config.tag}", loop=i
            )
        train_list.append(acc_train)
        val_list.append(acc_val)

        # test latest
        model_test = model_constructor(
          num_features=dataset.meta['num_features'],
          num_classes=dataset.meta['num_classes'],
          gnn = config.tag
        ).to("cuda:0")
        tester = Launcher(model=model_train,save_root= save_root, writer = writer)
        tester.load_latest(filename=ckpt_filename, loop=i)
        acc_test, loss_test = tester.test(data, mask=data.test_mask)
        print('************ latest results, loss = {:.6f}, accuracy = {:.6f} *************'.format(loss_test, acc_test))
        test_list.append(acc_test)

        # test best
        tester.load_best(filename=ckpt_filename, loop=i)
        acc_test, loss_test = tester.test(data, mask=data.test_mask)
        print('************ best results, loss = {:.6f}, accuracy = {:.6f} *************'.format(loss_test, acc_test))
        best_list.append(acc_test)

        writer.close()
        # torch.cuda.empty_cache()
    
    print(config)
    print("total time: {:.6f}h".format((time.time()-begin) / 3600))
    print(train_list, "train ->",np.mean(train_list), np.std(train_list) )
    print(val_list, "val ->", np.mean(val_list), np.std(val_list) )
    print(test_list, "latest ->", np.mean(test_list), np.std(test_list), )
    print(best_list, "best ->", np.mean(best_list), np.std(best_list), )
