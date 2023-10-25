# model_checkpoint="facebook/esm2_t48_15B_UR50D" # 15B params
# model_checkpoint="facebook/esm2_t36_3B_UR50D"
# model_checkpoint="facebook/esm2_t33_650M_UR50D"
# model_checkpoint="facebook/esm2_t30_150M_UR50D"
# model_checkpoint="facebook/esm2_t12_35M_UR50D"
# model_checkpoint = "facebook/esm2_t6_8M_UR50D"  # 8M params

# torchrun train.py --train_sample_count=50000 --model_id="facebook/esm2_t33_650M_UR50D" --num_epochs=3

import os
import copy
from timeit import default_timer as timer
import torch
from torch.optim import AdamW
from args_parser import parse_args
from training_helper import cleanup, evaluation_metrics, init_distributed_training, load_data, train, eval, training_metrics

from transformers import (
    EsmForMaskedLM,
    set_seed,
    get_scheduler,
)
from transformers.models.esm.configuration_esm import get_default_vocab_list

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
    transformer_auto_wrap_policy
)
import functools
from transformers.models.esm.modeling_esm import EsmLayer
from torch.distributed.fsdp import ShardingStrategy


if __name__ == "__main__":
    
    run_start = timer()
    args = parse_args()

    if args.seed is not None:
        set_seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    local_rank, world_size, global_rank = init_distributed_training(args)
    
    train_dataset, train_sampler, train_loader, eval_loader = load_data(args, world_size, global_rank)

    train_metrics = training_metrics(args, train_loader, world_size)
    
    eval_metrics = evaluation_metrics(args, eval_loader, world_size)
    
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy, 
        transformer_layer_cls = {
            EsmLayer
        }
    )

    torch.cuda.set_device(local_rank)
    
    ## Load model
    model = EsmForMaskedLM.from_pretrained(args.model_id)
    if args.pretrain:
        my_config = copy.deepcopy(model.config)
        my_config.vocab_list = get_default_vocab_list()
        my_config.vocab_size = len(my_config.vocab_list)
        model = EsmForMaskedLM(my_config)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model.to(local_rank)
    
    model = FSDP(model, auto_wrap_policy=auto_wrap_policy, 
                 sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                 cpu_offload=CPUOffload(offload_params=True),
                 device_id=torch.cuda.current_device())
    
    optimizer = AdamW(model.parameters(), args.lr)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=train_metrics["num_total_training_steps"]
    )

    if global_rank == 0:
        print("Training with {}". format(train_metrics))

    
    starting_epoch = 0

    init_start_event.record()
    # Start training loop
    for epoch in range(starting_epoch, args.num_epochs):
        train(args, model, epoch, local_rank, train_loader, optimizer, lr_scheduler, train_sampler, train_metrics)
        eval(model, epoch, local_rank, eval_loader, eval_metrics)

    init_end_event.record()

    if global_rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")

    # Save checkpoint for evaluation (xm.save ensures only one process save)
    dist.barrier()
    if global_rank == 0:
        model = model.module if hasattr(model, "module") else model
        os.makedirs(args.model_dir, exist_ok=True)
        checkpoint = {"state_dict": model.state_dict()}
        path = f"{args.model_dir}/checkpoint.pt"
        torch.save(checkpoint, path)

        print("##### Model saved to: ", f"{args.model_dir}/checkpoint.pt")
        print(f"Run completed in {timer() - run_start} sec.")
    
    cleanup()