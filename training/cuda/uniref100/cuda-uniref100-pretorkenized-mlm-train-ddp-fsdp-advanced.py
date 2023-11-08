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

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
    CPUOffload
)

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing
)

from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
    transformer_auto_wrap_policy
)
import functools
from transformers.models.esm.modeling_esm import EsmLayer


def mp_policy(args):
    bf16_ready = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
    )
    
    bfSixteen = MixedPrecision(
        param_dtype=torch.float32,
        # Gradient communication precision.
        reduce_dtype=torch.bfloat16,
        # Buffer precision.
        buffer_dtype=torch.bfloat16,
    )

    if (args.apply_mixed_precision == 1 and bf16_ready):
        mp_policy = bfSixteen
    else:
        mp_policy = None
    return mp_policy

def activation_checkpointing(model):
    check_fn = lambda submodule:isinstance(submodule, EsmLayer)
    non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT
        )
    apply_activation_checkpointing(
            model, 
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=check_fn
        )

def save_model_checkpoint_fsdp(model, path, global_rank):
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = model.state_dict()

    if global_rank == 0:
        print(f"--> Saving model ...")
        torch.save(cpu_state, path)
        print("--> ##### Model saved to: ", f"{path}")


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
    sharding_strategy = ShardingStrategy.HYBRID_SHARD
    mp_policy = mp_policy(args)    

    if global_rank == 0:
        print("--> Mixed precision training policy : {}".format(mp_policy))

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
    
    model = FSDP(model, auto_wrap_policy=auto_wrap_policy, 
                 sharding_strategy=sharding_strategy,
                 mixed_precision=mp_policy,
                 cpu_offload=CPUOffload(offload_params=True),
                 device_id=torch.cuda.current_device())
    
    if (args.apply_activation_checkpointing):
        activation_checkpointing(model)

        if global_rank == 0:
            print("--> Activation Checkpoiting Enabled.")

    optimizer = AdamW(model.parameters(), args.lr)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=train_metrics["num_total_training_steps"]
    )

    if global_rank == 0:
        print("--> Training with {}". format(train_metrics))
    
    starting_epoch = 0

    init_start_event.record()
    # Start training loop
    for epoch in range(starting_epoch, args.num_epochs):
        train(args, model, epoch, local_rank, train_loader, optimizer, lr_scheduler, train_sampler, train_metrics)
        eval(model, epoch, local_rank, eval_loader, eval_metrics)

        if global_rank == 0:
            print(f"--> Epoch {epoch} completed.")

        path = f"{args.model_dir}/model-chkp-epoch-{epoch}.pt"
        save_model_checkpoint_fsdp(model, path, global_rank)

    init_end_event.record()

    if global_rank == 0:
        print(f"--> CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")
    
    print(f"--> Run completed in {timer() - run_start} sec.")

    dist.barrier()
    cleanup()