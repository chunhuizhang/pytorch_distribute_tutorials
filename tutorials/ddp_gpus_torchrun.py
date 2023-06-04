import os, sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup():
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    # rank 0 process
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "12355"
    # nccl：NVIDIA Collective Communication Library 
    # 分布式情况下的，gpus 间通信
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    
class Trainer:
    def __init__(self, 
                 model: torch.nn.Module, 
                 train_dataloader: DataLoader, 
                 optimizer: torch.optim.Optimizer, 
                 ) -> None:
        self.gpu_id = int(os.environ['LOCAL_RANK'])
        self.model = model.to(self.gpu_id)
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.model = DDP(model, device_ids=[self.gpu_id])
        

    
    def _run_batch(self, xs, ys):
        self.optimizer.zero_grad()
        output = self.model(xs)
        loss = F.cross_entropy(output, ys)
        loss.backward()
        self.optimizer.step()
    
    def _run_epoch(self, epoch):
        batch_size = len(next(iter(self.train_dataloader))[0])
        print(f'[GPU: {self.gpu_id}] Epoch: {epoch} | Batchsize: {batch_size} | Steps: {len(self.train_dataloader)}')
        self.train_dataloader.sampler.set_epoch(epoch)
        for xs, ys in self.train_dataloader:
            xs = xs.to(self.gpu_id)
            ys = ys.to(self.gpu_id)
            self._run_batch(xs, ys)
    
    def train(self, max_epoch: int):
        for epoch in range(max_epoch):
            self._run_epoch(epoch)

class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]

def main(max_epochs: int, batch_size: int):
    ddp_setup()
    
    train_dataset = MyTrainDataset(2048)
    train_dataloader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              pin_memory=True, 
                              shuffle=False, 
                              # batch input: split to each gpus (且没有任何 overlaping samples 各个 gpu 之间)
                              sampler=DistributedSampler(train_dataset))
    model = torch.nn.Linear(20, 1)
    optimzer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    trainer = Trainer(model=model, optimizer=optimzer, train_dataloader=train_dataloader)
    trainer.train(max_epochs)
    
    destroy_process_group()

    
if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--max_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
#    world_size = torch.cuda.device_count()
    main(args.max_epochs, args.batch_size)
    
    