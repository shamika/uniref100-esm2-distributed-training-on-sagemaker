import argparse
import os
from transformers import (
    SchedulerType,
)

def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate to use for training."
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max length of sequence for collator.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
        help="Path to model output folder.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="facebook/esm2_t33_650M_UR50D",
        help="Model id to use for training.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of epochs to train."
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--training_dir",
        type=str,
        default=os.environ["SM_CHANNEL_TRAIN"],
        help="Path to train dataset.",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=os.environ["SM_CHANNEL_TEST"],
        help="Path to evaluation dataset.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Number of steps between logging updates.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps between gradient optimization.",
    )
    parser.add_argument(
        "--train_sample_count",
        type=int,
        default=None,
        help="Number of training samples to pre-process.",
    )
    parser.add_argument(
        "--steps_this_run",
        type=int,
        default=None,
        help="Max number of steps.",
    )
    parser.add_argument(
        "--pretrain",
        type=int,
        default=0,
        help="Initialize random weights?",
    )
    parser.add_argument(
        "--train_index_file_path",
        type=str,
        default="train_index_map",
        help="",
    )
    parser.add_argument(
        "--test_index_file_path",
        type=str,
        default="test_index_map",
        help="",
    )
    parser.add_argument(
        "--apply_activation_checkpointing",
        type=int,
        default=0,
        help="Whether to apply activation checkpointing or not.",
    )

    args, _ = parser.parse_known_args()
    return args