
import os
import math
from timeit import default_timer as timer
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datetime import timedelta

from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from uniref100_torkenized_dataset import Uniref100TorkenizedDataset
from uniref100_torkenized_dynamodb_dataset import DynamoDBDataset

from os import listdir
import torch.distributed as dist


def calc_perplexity(loss):
    try:
        perplexity = math.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return perplexity

def training_metrics(args, train_loader, world_size):

    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps) # per GPU
    num_total_training_steps = args.num_epochs * num_update_steps_per_epoch # per GPU

    total_train_batch_size = args.per_device_train_batch_size * world_size
    total_train_batch_size_with_gradient_accumulation = total_train_batch_size * args.gradient_accumulation_steps
    
    samples_processed_per_logging_update = total_train_batch_size_with_gradient_accumulation * args.logging_steps
    tokens_processed_per_logging_update = (
        samples_processed_per_logging_update * args.max_length
    )
    total_tokens_per_batch = total_train_batch_size_with_gradient_accumulation * args.max_length
    
    return {
        "total_no_of_batches_per_gpu" : len(train_loader),
        "per_device_train_batch_size" : args.per_device_train_batch_size,
        "epochs" :  args.num_epochs,
        "num_update_steps_per_epoch": num_update_steps_per_epoch,
        "num_total_training_steps": num_total_training_steps,
        "total_train_batch_size": total_train_batch_size,
        "total_train_batch_size_with_gradient_accumulation" : total_train_batch_size_with_gradient_accumulation,
        "samples_processed_per_logging_update": samples_processed_per_logging_update,
        "tokens_processed_per_logging_update": tokens_processed_per_logging_update,
        "total_tokens_per_batch" : total_tokens_per_batch
    }

def evaluation_metrics(args, eval_loader, world_size):
    num_eval_steps_per_epoch = len(eval_loader)
    total_eval_batch_size = args.per_device_eval_batch_size * world_size
    samples_processed_per_eval = total_eval_batch_size * num_eval_steps_per_epoch
    tokens_processed_per_eval = samples_processed_per_eval * args.max_length

    return {
        "num_eval_steps_per_epoch" : num_eval_steps_per_epoch,
        "total_eval_batch_size" : total_eval_batch_size,
        "samples_processed_per_logging_update" : samples_processed_per_eval,
        "tokens_processed_per_logging_update" : tokens_processed_per_eval
    }

def report_metrics(
    local_rank, start_time, loss, epoch, step, global_batch_size, global_token_size, prefix=None
):
    reported_loss = loss.detach().float()
    now = timer()
    duration = now - start_time
    samples_per_sec = global_batch_size / duration
    tokens_per_sec = global_token_size / duration
    perplexity = calc_perplexity(reported_loss)
    if prefix:
        prefix = prefix + " "
    if local_rank == 0:
        print(
            f"Epoch: {epoch}, Step: {step}, {prefix}Loss: {reported_loss:0.4f}, {prefix}Perplexity: {perplexity:0.4f}, {prefix}Samples/sec: {samples_per_sec:0.4f}, {prefix}Tokens/sec: {tokens_per_sec:0.4f}, {prefix}Total time taken for the step :{duration}"
        )



def get_index_file_index_path(base_path, folder):
    index_file_folder = os.path.join(base_path, folder)
    #print("Index file folder is [{}]".format(index_file_folder))
    
    csv_files = [file for file in listdir(index_file_folder) if file.endswith('.csv')]
    index_file_location = os.path.join(index_file_folder, csv_files[0])
    
    #print("Index file location is [{}]".format(index_file_location))
    return index_file_location

def init_distributed_training(args):

    """Initializes distributed training settings."""
    local_rank=int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    global_rank = int(os.environ["RANK"])
    
    if local_rank == 0:
        print("Local Rank is : {}".format(os.environ["LOCAL_RANK"]))
        print("Worldsize is : {}".format(os.environ["WORLD_SIZE"]))
        print("Rank is : {}".format(os.environ["RANK"]))
        
        print("Master address is : {}".format(os.environ['MASTER_ADDR']))
        print("Master port is : {}".format(os.environ["MASTER_PORT"]))
    
    dist.init_process_group(backend="nccl", world_size=world_size, rank=global_rank, init_method="env://", timeout=timedelta(seconds=120))

    return local_rank, world_size, global_rank

def cleanup():
    dist.destroy_process_group()

def load_data(args, world_size, global_rank, num_workers=16):
    """Loads training and evaluation datasets."""
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.model_max_length = args.max_length
    
    # train_dataset = Uniref100TorkenizedDataset(args.training_dir, get_index_file_index_path(args.training_dir, args.train_index_file_path))
    train_dataset = DynamoDBDataset(table_name="uniref100-esm2-tokenized", total_items=319594, region='us-east-1')
    
    test_dataset = Uniref100TorkenizedDataset(args.test_dir, get_index_file_index_path(args.test_dir, args.test_index_file_path))
    #train_dataset = UnirefInMemoryCSVDataset(args.training_dir, tokenizer, args.max_length)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=global_rank)
    
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15
    )
    
    train_loader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler,
        shuffle=False if train_sampler else True,
        num_workers = num_workers
    )
    eval_loader = DataLoader(
        test_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
        sampler=test_sampler,
        shuffle=False if test_sampler else True,
        num_workers = num_workers
    )
    
    return train_dataset, train_sampler, train_loader, eval_loader


def train(args, model, epoch, local_rank, train_loader, optimizer, lr_scheduler, train_sampler):
    
    model.train()
    train_sampler.set_epoch(epoch)

    ddp_acc_loss_and_steps = torch.zeros(2).to(local_rank)
    optimizer.zero_grad()  # Ensure gradients are zeroed out at the start
    
    total_current_device_samples_per_step = 0
    
    step_start_time = timer()
    
    for idx, batch in enumerate(train_loader):
            
        total_current_device_samples_per_step += batch['input_ids'].size(0)
        
        batch = {
                k: v.to(local_rank) for k, v, in batch.items()
        }  # Transfer data to accelerator

        outputs = model(**batch)  # Forward pass
        
        loss = outputs.loss  # Calculate loss
        loss = loss / args.gradient_accumulation_steps # Normalize the loss
        loss.backward()  # Calculate new gradients with backprop

        ddp_acc_loss_and_steps[0] += loss.item() * args.gradient_accumulation_steps  # Scale loss back up

        if ((idx + 1) % args.gradient_accumulation_steps == 0) or (
                idx + 1 == len(train_loader) # last batch
            ):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            ddp_acc_loss_and_steps[0] /= args.gradient_accumulation_steps
            ddp_acc_loss_and_steps[1] += 1
            
            total_global_device_samples_per_step = total_current_device_samples_per_step * dist.get_world_size()
            
            if (ddp_acc_loss_and_steps[1] % args.logging_steps == 0):
                dist.all_reduce(ddp_acc_loss_and_steps, op=dist.ReduceOp.AVG)
                #ddp_acc_loss_and_steps[0] /= dist.get_world_size()
                report_metrics(
                        local_rank,
                        step_start_time,
                        ddp_acc_loss_and_steps[0],
                        epoch,
                        ddp_acc_loss_and_steps[1],
                        total_global_device_samples_per_step,
                        total_global_device_samples_per_step * args.max_length,
                        "Training",
                    )
            ddp_acc_loss_and_steps[0] = 0.0 # Reset accumulated loss after reporting
            total_current_device_samples_per_step = 0
            step_start_time = timer()


def eval(args, model, epoch, local_rank, eval_loader):
    eval_start_time = timer()
    model.eval()
    eval_running_loss = 0
    total_current_device_samples = 0 
    
    with torch.no_grad():
        for batch in eval_loader:
            batch_size = batch['input_ids'].size(0)
            total_current_device_samples += batch_size
            batch = {k: v.to(local_rank) for k, v, in batch.items()}
            outputs = model(**batch)
            eval_loss = outputs.loss
            eval_running_loss += eval_loss.detach().float()
    
    average_loss = eval_running_loss / len(eval_loader)
    
    completed_steps = (epoch + 1) * len(eval_loader)
    
    dist.all_reduce(average_loss, op=dist.ReduceOp.AVG)
    report_metrics(
        local_rank, 
        eval_start_time, 
        average_loss, 
        epoch, 
        completed_steps,
        total_current_device_samples * dist.get_world_size(),
        total_current_device_samples * args.max_length,
        "Eval"
    )