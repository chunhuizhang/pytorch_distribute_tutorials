# encoding: utf8

import logging
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp


def print_rank_0(msg, *args, **kwargs):
    rank = dist.get_rank()
    if rank == 0:
        logging.info(msg, *args, **kwargs)


def dist_allgather():
    print_rank_0("allgather:")
    dist.barrier()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    input_tensor = torch.tensor(rank * 2 + 1)
    tensor_list = [torch.zeros(1, dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(tensor_list, input_tensor)
    logging.info(f"allgather, rank: {rank}, input_tensor: {repr(input_tensor)}, output tensor_list: {tensor_list}")
    dist.barrier()


def dist_allreduce():
    print_rank_0("all_reduce:")
    dist.barrier()

    rank = dist.get_rank()
    # world_size = torch.distributed.get_world_size()

    if rank == 0:
        tensor = torch.tensor([1., 2.])
    else:
        tensor = torch.tensor([2., 3.])
    input_tensor = tensor.clone()
    dist.all_reduce(tensor)

    logging.info(f"all_reduce, rank: {rank}, before allreduce tensor: {repr(input_tensor)}, after allreduce tensor: {repr(tensor)}")
    dist.barrier()


def dist_reducescatter():
    print_rank_0("reduce_scatter:")
    dist.barrier()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    output = torch.empty(1, dtype=torch.int64)
    input_list = [torch.tensor(rank*2+1), torch.tensor(rank*2+2)]
    dist.reduce_scatter(output, input_list, op=ReduceOp.SUM)
    dist.barrier()
    logging.info(f"reduce_scatter, rank: {rank}, input_list: {input_list}, tensor: {repr(output)}")
    dist.barrier()


def dist_broadcast():
    print_rank_0("broadcast:")
    dist.barrier()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    src_rank = 0
    tensor = torch.tensor(world_size) if rank == src_rank else torch.zeros(1, dtype=torch.int64)
    before_tensor = tensor.clone()
    dist.broadcast(tensor, src=src_rank)
    logging.info(f"broadcast, rank: {rank}, before broadcast tensor: {repr(before_tensor)} after broadcast tensor: {repr(tensor)}")
    dist.barrier()


def dist_scatter():
    print_rank_0("scatter:")
    dist.barrier()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    tensor = torch.zeros(world_size)
    before_tensor = tensor.clone()
    if dist.get_rank() == 0:
        # Assumes world_size of 2.
        # Only tensors, all of which must be the same size.
        t_ones = torch.ones(world_size)
        t_fives = torch.ones(world_size) * 5
        # [[1, 1], [5, 5]]
        scatter_list = [t_ones, t_fives]
    else:
        scatter_list = None
    dist.scatter(tensor, scatter_list, src=0)
    logging.info(f"scatter, rank: {rank}, before scatter: {repr(before_tensor)} after scatter: {repr(tensor)}")
    dist.barrier()

def dist_gather():
    print_rank_0("gather:")
    dist.barrier()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    tensor = torch.tensor([rank*2+1], dtype=torch.float32)
    before_tensor = tensor.clone()
    
    gather_list = [torch.zeros(1) for _ in range(world_size)] if rank == 0 else None

    dist.gather(tensor, gather_list, dst=0)
    
    logging.info(f"gather, rank: {rank}, before gather: {repr(before_tensor)} after gather: {repr(gather_list)}")
    dist.barrier()
    
    
def dist_reduce():
    print_rank_0("reduce:")
    dist.barrier()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    tensor = torch.tensor([rank*2 + 1], dtype=torch.float32)
    before_tensor = tensor.clone()

    dist.reduce(tensor, op=ReduceOp.SUM, dst=0)
    
    logging.info(f"reduce, rank: {rank}, before reduce: {repr(before_tensor)} after reduce: {repr(tensor)}")
    dist.barrier()


def main():
    dist.init_process_group("nccl")

    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    torch.set_default_device(f"cuda:{local_rank}")
    logging.info(f'main, rank: {rank}, local_rank: {local_rank}, default_device: cuda:{local_rank}')
    
    # dist_scatter()
    # dist_gather()
    # dist_broadcast()
    # dist_reduce()
    # dist_allreduce()
    # dist_allgather()
    dist_reducescatter()

    


if __name__ == "__main__":
    logging.basicConfig(format=logging.BASIC_FORMAT, level=logging.INFO)
    main()