import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time


def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

# def run(rank, size):
#     """ Distributed function to be implemented later. """
#     time.sleep(2)
#     print(rank, size)


# gloo
# def run(rank, size):
#     tensor = torch.zeros(1)
#     if rank == 0:
#         tensor += 1
#         # Send the tensor to process 1
#         dist.send(tensor=tensor, dst=1)
#     else:
#         # Receive tensor from process 0
#         dist.recv(tensor=tensor, src=0)
#     print('Rank ', rank, ' has data ', tensor[0])
    
# def run(rank, size):
#     # torch.cuda.set_device(rank)
#     tensor = torch.zeros(1).to(rank)
#     # req = None
#     if rank == 0:
#         tensor += 1
#         # Send the tensor to process 1
#         req = dist.send(tensor=tensor, dst=1)
#     else:
#         # Receive tensor from process 0
#         print('init tentor', tensor)
#         req = dist.recv(tensor=tensor, src=0)
#     # req.wait()
#     print(f'Rank: {rank}, has data {tensor}')
    
""" All-Reduce example."""
def run(rank, size):
    """ Simple collective communication. """
    group = dist.new_group([0, 1])
    if rank == 0:
        tensor = torch.tensor([1., 2., 3.])
    else:
        tensor = torch.tensor([4., 5., 6.])
    tensor = tensor.to(rank)
    print(f'Rank: {rank}, random tensor: {tensor}')
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print(f'Rank: {rank}, has data: {tensor}')

if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    # for p in processes:
    #     p.join()
    
    print('finished')