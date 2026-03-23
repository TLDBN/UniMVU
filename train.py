# Separate-task training entrypoint for the released UniMVU model.

import copy
import os
import pathlib
import sys
from bisect import bisect_right
from itertools import accumulate

import torch
from transformers import TrainerCallback
from torch.utils.data import ConcatDataset

from utils import prepare_video_model_v2

from typing import List, Optional, Tuple

from libs.utils.model_trainer import VideoModelTrainer
from libs.dataset.base_dataset import (
    LazySupervisedVideoDataset,
    DataCollatorForSupervisedVideoDataset,
)
from libs.utils.train_utils import (
    parse_argument_classes,
    get_peft_state_maybe_zero_3,
    safe_save_model_for_hf_videotrainer,
    get_peft_state_non_lora_maybe_zero_3_with_state_dict,
)

class ShuffledConcatDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets: List[torch.utils.data.Dataset],
        modes: List[str],
        seed: int = 0,
        batch_size: int = 1,
        world_size: int = 1,
    ):
        if len(datasets) != len(modes):
            raise ValueError("Length of datasets and modes must be the same.")
        self.datasets = list(datasets)
        self.modes = list(modes)
        self.cumulative_sizes = list(accumulate(len(d) for d in self.datasets))
        self.dataset_offsets = [0] + self.cumulative_sizes[:-1]
        self.batch_size = max(int(batch_size), 1)
        self.world_size = max(int(world_size), 1)
        self._length = self.cumulative_sizes[-1] if self.cumulative_sizes else 0
        self.base_seed = int(seed)
        self._epoch = 0

        self._shuffled_order: Optional[List[int]] = None
        self._current_modality_lengths: Optional[List[int]] = None
        self._dropped_remainder_per_dataset: List[int] = [0] * len(self.datasets)
        self._dropped_tail_blocks: int = 0
        self.set_epoch(0)

    def __len__(self):
        return self._length

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
        if self._length == 0:
            self._shuffled_order = []
            self._current_modality_lengths = []
            self._dropped_remainder_per_dataset = [0] * len(self.datasets)
            self._dropped_tail_blocks = 0
            return
        generator = torch.Generator()
        generator.manual_seed(self.base_seed + self._epoch)
        blocks: List[Tuple[List[int], int]] = []
        dropped_remainders: List[int] = [0] * len(self.datasets)

        for dataset_idx, dataset in enumerate(self.datasets):
            dataset_len = len(dataset)
            if dataset_len == 0:
                continue
            permuted = torch.randperm(dataset_len, generator=generator).tolist()
            offset = self.dataset_offsets[dataset_idx]
            sign = -1 if self.modes[dataset_idx] == 'audio' else 1
            for start in range(0, len(permuted), self.batch_size):
                block = permuted[start:start + self.batch_size]
                global_block = [offset + local_idx for local_idx in block]
                blocks.append((global_block, sign))
            dropped_remainders[dataset_idx] = 0

        if not blocks:
            self._shuffled_order = []
            self._current_modality_lengths = []
            self._length = 0
            self._dropped_remainder_per_dataset = dropped_remainders
            self._dropped_tail_blocks = 0
            return

        block_perm = torch.randperm(len(blocks), generator=generator).tolist()
        shuffled_blocks: List[Tuple[List[int], int]] = [blocks[idx] for idx in block_perm]

        dropped_tail_blocks = 0
        if self.world_size > 1 and len(shuffled_blocks) >= self.world_size:
            remainder_blocks = len(shuffled_blocks) % self.world_size
            if remainder_blocks:
                # Align block count with world size so each rank sees whole, single-dataset batches.
                dropped_tail_blocks = remainder_blocks
                shuffled_blocks = shuffled_blocks[:-remainder_blocks]

        shuffled_order: List[int] = []
        shuffled_signs: List[int] = []
        for block, sign in shuffled_blocks:
            shuffled_order.extend(block)
            shuffled_signs.extend([sign] * len(block))

        self._shuffled_order = shuffled_order
        self._current_modality_lengths = shuffled_signs
        self._length = len(shuffled_order)
        self._dropped_remainder_per_dataset = dropped_remainders
        self._dropped_tail_blocks = dropped_tail_blocks

    def _resolve_index(self, idx: int):
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of bounds for dataset of length {self._length}")
        if self._shuffled_order is None:
            dataset_idx = bisect_right(self.cumulative_sizes, idx)
            prev_cum = 0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
            sample_idx = idx - prev_cum
        else:
            global_idx = self._shuffled_order[idx]
            dataset_idx = bisect_right(self.cumulative_sizes, global_idx)
            prev_cum = 0 if dataset_idx == 0 else self.cumulative_sizes[dataset_idx - 1]
            sample_idx = global_idx - prev_cum
        return dataset_idx, sample_idx

    def __getitem__(self, idx: int):
        dataset_idx, sample_idx = self._resolve_index(idx)
        return self.datasets[dataset_idx][sample_idx]

    @property
    def modality_lengths(self) -> List[int]:
        if self._current_modality_lengths is None:
            base: List[int] = []
            for mode, dataset in zip(self.modes, self.datasets):
                sign = -1 if mode == 'audio' else 1
                base.extend([sign] * len(dataset))
            return base
        return self._current_modality_lengths

    @property
    def unique_modes(self):
        return set(self.modes)


def _build_training_dataset(
    tokenizer,
    data_args,
    seed: int = 0,
    batch_size: int = 1,
    world_size: int = 1,
    shuffle: bool = True,
):
    collator = DataCollatorForSupervisedVideoDataset(tokenizer=tokenizer)

    if not isinstance(data_args.annotation_path, list):
        dataset = LazySupervisedVideoDataset(
            anno_path=data_args.annotation_path,
            fast_path_mapping_path=data_args.fast_path_mapping_path,
            tokenizer=tokenizer,
            data_args=data_args,
        )
        return dict(train_dataset=dataset, eval_dataset=None, data_collator=collator)

    annotation_paths: List[str] = data_args.annotation_path
    fast_mapping_paths: List[str] = data_args.fast_path_mapping_path
    data_roots: List[str] = data_args.data_root

    slow_mapping_paths = data_args.slow_path_mapping_path if isinstance(data_args.slow_path_mapping_path, list) else None
    slow_data_roots = data_args.slow_path_data_root if isinstance(data_args.slow_path_data_root, list) else None

    dataset_count = len(annotation_paths)
    if hasattr(data_args, "to_per_dataset_arguments"):
        per_dataset_args = data_args.to_per_dataset_arguments()
        if len(per_dataset_args) != dataset_count:
            raise ValueError(
                f"Expected {dataset_count} dataset argument entries, but got {len(per_dataset_args)}."
            )
        inferred_fast_types = [arg.fast_feat_type for arg in per_dataset_args]
    else:
        per_dataset_args = None
        inferred_fast_types = [data_args.fast_feat_type] * dataset_count

    fast_types = inferred_fast_types
    if len(fast_types) != dataset_count:
        raise ValueError(
            f"Expected {dataset_count} fast feature types, but received {len(fast_types)}: {fast_types}"
        )

    datasets = []
    for idx in range(dataset_count):
        if per_dataset_args is not None:
            per_args = per_dataset_args[idx]
        else:
            per_args = copy.deepcopy(data_args)
            per_args.annotation_path = annotation_paths[idx]
            per_args.fast_path_mapping_path = fast_mapping_paths[idx]
            per_args.data_root = data_roots[idx]
        per_args.fast_feat_type = fast_types[idx]
        for attr_name, attr_value in data_args.__dict__.items():
            if attr_name.startswith("_"):
                continue
            if attr_name in ("annotation_path", "fast_path_mapping_path", "data_root", "slow_path_mapping_path", "slow_path_data_root", "fast_feat_type"):
                continue
            if attr_name not in per_args.__dict__ or per_args.__dict__[attr_name] is None:
                per_args.__dict__[attr_name] = attr_value
        if slow_mapping_paths is not None:
            per_args.slow_path_mapping_path = slow_mapping_paths[idx]
            per_args.slow_path_data_root = slow_data_roots[idx]

        dataset = LazySupervisedVideoDataset(
            anno_path=per_args.annotation_path,
            fast_path_mapping_path=per_args.fast_path_mapping_path,
            tokenizer=tokenizer,
            data_args=per_args,
        )
        datasets.append(dataset)
    if shuffle:
        train_dataset = ShuffledConcatDataset(
            datasets,
            modes=fast_types,
            seed=seed,
            batch_size=batch_size,
            world_size=world_size,
        )
    else:
        train_dataset = ConcatDataset(datasets)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=collator)


class _DatasetShuffleCallback(TrainerCallback):
    def __init__(self, dataset: ShuffledConcatDataset):
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch is not None else 0
        self.dataset.set_epoch(epoch)
        return control


def train_unimvu(attn_implementation=None):
    model_args, data_args, training_args = parse_argument_classes(sys.argv[1:], return_name=False)

    compute_dtype = (
        torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    model, tokenizer = prepare_video_model_v2(training_args, model_args, data_args, compute_dtype, attn_implementation)

    data_module = _build_training_dataset(
        tokenizer=tokenizer,
        data_args=data_args,
        seed=getattr(training_args, "seed", 0),
        batch_size=training_args.per_device_train_batch_size,
        world_size=getattr(training_args, "world_size", 1),
        shuffle=getattr(training_args, "shuffle", True),
    )

    train_dataset = data_module.get("train_dataset")
    if hasattr(train_dataset, "unique_modes") and len(train_dataset.unique_modes) > 1:
        training_args.group_by_modality_length = False
        training_args.shuffle = False
        training_args.dataloader_drop_last = True

    trainer = VideoModelTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )
    print("Using VideoModelTrainer with UniMVU")
    if isinstance(train_dataset, ShuffledConcatDataset):
        trainer.add_callback(_DatasetShuffleCallback(train_dataset))

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        torch.use_deterministic_algorithms(False)
        trainer.train()

    trainer.save_state()
    model.config.use_cache = True
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3_with_state_dict(
            model.state_dict(), special_key=training_args.extra_trainable_modules or []
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_videotrainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train_unimvu(attn_implementation="flash_attention_2")
