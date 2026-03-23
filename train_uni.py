# Mixed-dataset training entrypoint for UniMVU with grouped alpha sampling.

import copy
import math
import os
import pathlib
import sys
from collections import OrderedDict
from typing import List, Sequence

import torch
from torch.utils.data import ConcatDataset
from transformers import TrainerCallback

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
        self.batch_size = max(int(batch_size), 1)
        self.world_size = max(int(world_size), 1)
        self.base_seed = int(seed)
        self._epoch = 0
        self._samples = []
        self._modality_signs = []
        self._length = 0
        self._dropped_tail_blocks = 0
        self.set_epoch(0)

    def __len__(self):
        return self._length

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
        generator = torch.Generator()
        generator.manual_seed(self.base_seed + self._epoch)
        blocks = []

        for dataset_idx, dataset in enumerate(self.datasets):
            dataset_len = len(dataset)
            if dataset_len == 0:
                continue
            permuted = torch.randperm(dataset_len, generator=generator).tolist()
            sign = -1 if self.modes[dataset_idx] == "audio" else 1
            for start in range(0, len(permuted), self.batch_size):
                chunk = permuted[start:start + self.batch_size]
                blocks.append(([(dataset_idx, sample_idx) for sample_idx in chunk], sign))

        if not blocks:
            self._samples = []
            self._modality_signs = []
            self._length = 0
            self._dropped_tail_blocks = 0
            return

        shuffled_blocks = [blocks[idx] for idx in torch.randperm(len(blocks), generator=generator).tolist()]
        self._dropped_tail_blocks = 0
        if self.world_size > 1 and len(shuffled_blocks) >= self.world_size:
            remainder_blocks = len(shuffled_blocks) % self.world_size
            if remainder_blocks:
                self._dropped_tail_blocks = remainder_blocks
                shuffled_blocks = shuffled_blocks[:-remainder_blocks]

        self._samples = []
        self._modality_signs = []
        for block, sign in shuffled_blocks:
            self._samples.extend(block)
            self._modality_signs.extend([sign] * len(block))
        self._length = len(self._samples)

    def __getitem__(self, idx: int):
        dataset_idx, sample_idx = self._samples[idx]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def modality_lengths(self) -> List[int]:
        return self._modality_signs

    @property
    def unique_modes(self):
        return set(self.modes)


class AlphaReweightedConcatDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datasets: List[torch.utils.data.Dataset],
        modes: List[str],
        alpha: float,
        seed: int = 0,
        batch_size: int = 1,
        world_size: int = 1,
    ):
        if len(datasets) != len(modes):
            raise ValueError("Length of datasets and modes must be the same.")
        self.datasets = list(datasets)
        self.modes = list(modes)
        self.alpha = float(alpha)
        self.dataset_lengths = [len(d) for d in self.datasets]
        self.batch_size = max(int(batch_size), 1)
        self.world_size = max(int(world_size), 1)
        self.base_seed = int(seed)
        self._epoch = 0
        self._target_counts = self._compute_target_counts()
        self._samples = []
        self._modality_signs = []
        self._latest_dataset_counts = [0] * len(self.datasets)
        self._length = 0
        self._dropped_tail_blocks = 0
        self.set_epoch(0)

    def _compute_target_counts(self) -> List[int]:
        lengths = self.dataset_lengths
        if not lengths:
            return []
        total = sum(lengths)
        if total == 0:
            return [0] * len(lengths)
        if abs(self.alpha - 1.0) < 1e-8:
            return list(lengths)

        weights = [length ** self.alpha if length > 0 else 0.0 for length in lengths]
        weight_sum = sum(weights)
        raw = [total * (weight / weight_sum) for weight in weights]
        allocations = [int(math.floor(value)) for value in raw]

        for idx, (length, count) in enumerate(zip(lengths, allocations)):
            if length > 0 and count == 0:
                allocations[idx] = 1

        remainder = total - sum(allocations)
        if remainder > 0:
            ranked = sorted(
                ((raw[idx] - allocations[idx], idx) for idx in range(len(raw))),
                key=lambda item: item[0],
                reverse=True,
            )
            for _, idx in ranked:
                if remainder <= 0:
                    break
                allocations[idx] += 1
                remainder -= 1
        elif remainder < 0:
            remaining = -remainder
            ranked = sorted(
                ((allocations[idx] - raw[idx], idx) for idx in range(len(raw))),
                key=lambda item: item[0],
                reverse=True,
            )
            for _, idx in ranked:
                if remaining <= 0:
                    break
                max_removable = allocations[idx] - (1 if self.dataset_lengths[idx] > 0 else 0)
                if max_removable <= 0:
                    continue
                remove = min(remaining, max_removable)
                allocations[idx] -= remove
                remaining -= remove
        return allocations

    def __len__(self):
        return self._length

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
        generator = torch.Generator()
        generator.manual_seed(self.base_seed + self._epoch)
        blocks = []

        for dataset_idx, dataset in enumerate(self.datasets):
            target = self._target_counts[dataset_idx]
            dataset_len = self.dataset_lengths[dataset_idx]
            if target <= 0 or dataset_len == 0:
                continue
            repetitions = max(1, math.ceil(target / dataset_len))
            reservoir = []
            for _ in range(repetitions):
                reservoir.extend(torch.randperm(dataset_len, generator=generator).tolist())
            selected = reservoir[:target]
            sign = -1 if self.modes[dataset_idx] == "audio" else 1
            for start in range(0, len(selected), self.batch_size):
                chunk = selected[start:start + self.batch_size]
                blocks.append(([(dataset_idx, sample_idx) for sample_idx in chunk], sign))

        if not blocks:
            self._samples = []
            self._modality_signs = []
            self._latest_dataset_counts = [0] * len(self.datasets)
            self._length = 0
            self._dropped_tail_blocks = 0
            return

        shuffled_blocks = [blocks[idx] for idx in torch.randperm(len(blocks), generator=generator).tolist()]
        self._dropped_tail_blocks = 0
        if self.world_size > 1 and len(shuffled_blocks) >= self.world_size:
            remainder_blocks = len(shuffled_blocks) % self.world_size
            if remainder_blocks:
                self._dropped_tail_blocks = remainder_blocks
                shuffled_blocks = shuffled_blocks[:-remainder_blocks]

        self._samples = []
        self._modality_signs = []
        self._latest_dataset_counts = [0] * len(self.datasets)
        for block, sign in shuffled_blocks:
            for dataset_idx, sample_idx in block:
                self._samples.append((dataset_idx, sample_idx))
                self._modality_signs.append(sign)
                self._latest_dataset_counts[dataset_idx] += 1
        self._length = len(self._samples)

    def __getitem__(self, idx: int):
        dataset_idx, sample_idx = self._samples[idx]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def modality_lengths(self) -> List[int]:
        return self._modality_signs

    @property
    def unique_modes(self):
        return set(self.modes)

    @property
    def target_counts(self) -> List[int]:
        return list(self._target_counts)

    @property
    def sampled_counts(self) -> List[int]:
        return list(self._latest_dataset_counts)


class _DatasetShuffleCallback(TrainerCallback):
    def __init__(self, dataset):
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch) if state.epoch is not None else 0
        self.dataset.set_epoch(epoch)
        return control


def _ensure_all_equal(args: Sequence[object], attr: str, dataset_name: str):
    first = getattr(args[0], attr, None)
    for arg in args[1:]:
        other = getattr(arg, attr, None)
        if other != first:
            raise ValueError(
                f"Inconsistent '{attr}' for dataset '{dataset_name}': {first} vs {other}."
            )
    return first


def _merge_optional_list(args: Sequence[object], attr: str):
    values = [getattr(arg, attr, None) for arg in args]
    if all(v is None for v in values):
        return None
    return values


def _merge_sample_ratio(args: Sequence[object]):
    ratios = [getattr(arg, "data_sample_ratio", None) for arg in args]
    if all(r is None for r in ratios):
        return None
    if any(r is None for r in ratios):
        raise ValueError("Mixed presence of data_sample_ratio within a grouped dataset is not supported.")
    return ",".join(str(r) for r in ratios)


def _merge_dataset_args(args: List[object], dataset_name: str):
    merged = copy.deepcopy(args[0])
    merged.annotation_path = [arg.annotation_path for arg in args]
    merged.fast_path_mapping_path = [arg.fast_path_mapping_path for arg in args]
    merged.data_root = [arg.data_root for arg in args]
    merged.slow_path_mapping_path = _merge_optional_list(args, "slow_path_mapping_path")
    merged.slow_path_data_root = _merge_optional_list(args, "slow_path_data_root")
    merged.second_sides_data_root = _merge_optional_list(args, "second_sides_data_root")
    merged.data_sample_ratio = _merge_sample_ratio(args)

    merged.fast_feat_type = _ensure_all_equal(args, "fast_feat_type", dataset_name)
    merged.use_fast = _ensure_all_equal(args, "use_fast", dataset_name)
    merged.use_fast_feat = _ensure_all_equal(args, "use_fast_feat", dataset_name)
    merged.use_slow = _ensure_all_equal(args, "use_slow", dataset_name)
    merged.use_slow_feat = _ensure_all_equal(args, "use_slow_feat", dataset_name)
    merged.use_second_sides = _ensure_all_equal(args, "use_second_sides", dataset_name)
    merged.second_sides_type = _ensure_all_equal(args, "second_sides_type", dataset_name)
    merged.video_loading_backbone = _ensure_all_equal(
        args, "video_loading_backbone", dataset_name
    )
    merged.modalities = _ensure_all_equal(args, "modalities", dataset_name)
    return merged


def _resolve_dataset_names(data_args, dataset_count: int) -> List[Optional[str]]:
    if hasattr(data_args, "_resolve_dataset_names"):
        try:
            names = data_args._resolve_dataset_names(dataset_count)
            if len(names) == dataset_count:
                return names
            print(
                f"[DataDebug] dataset name count mismatch ({len(names)} vs {dataset_count}); "
                "falling back to per-entry grouping."
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"[DataDebug] Unable to resolve dataset names: {exc}")
    return [None] * dataset_count


def _group_dataset_arguments(
    per_dataset_args: List[object], dataset_names: Optional[List[Optional[str]]]
):
    if dataset_names is None or len(dataset_names) != len(per_dataset_args):
        dataset_names = [None] * len(per_dataset_args)

    groups = OrderedDict()
    for idx, (arg, name) in enumerate(zip(per_dataset_args, dataset_names)):
        key = name or f"dataset_{idx}"
        groups.setdefault(key, []).append(arg)

    grouped_args = []
    grouped_fast_types = []
    grouped_names = []
    all_names_none = all(name is None for name in dataset_names)
    for name, args in groups.items():
        if len(args) == 1 or all_names_none:
            merged = args[0]
        else:
            merged = _merge_dataset_args(args, dataset_name=name)
        grouped_args.append(merged)
        grouped_fast_types.append(getattr(merged, "fast_feat_type", ""))
        grouped_names.append(name)
    return grouped_args, grouped_fast_types, grouped_names


def _build_training_dataset_grouped(
    tokenizer,
    data_args,
    seed: int = 0,
    batch_size: int = 1,
    world_size: int = 1,
    shuffle: bool = True,
    mix_alpha: float = 1.0,
):
    from libs.dataset.base_dataset import (
        LazySupervisedVideoDataset,
        DataCollatorForSupervisedVideoDataset,
    )

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

    slow_mapping_paths = (
        data_args.slow_path_mapping_path if isinstance(data_args.slow_path_mapping_path, list) else None
    )
    slow_data_roots = (
        data_args.slow_path_data_root if isinstance(data_args.slow_path_data_root, list) else None
    )

    dataset_count = len(annotation_paths)
    dataset_names: Optional[List[Optional[str]]] = None
    if hasattr(data_args, "to_per_dataset_arguments"):
        per_dataset_args = data_args.to_per_dataset_arguments()
        if len(per_dataset_args) != dataset_count:
            raise ValueError(
                f"Expected {dataset_count} dataset argument entries, but got {len(per_dataset_args)}."
            )
        dataset_names = _resolve_dataset_names(data_args, dataset_count)
    else:
        per_dataset_args = None

    if per_dataset_args is None:
        per_dataset_args = []
        for idx in range(dataset_count):
            per_args = copy.deepcopy(data_args)
            per_args.annotation_path = annotation_paths[idx]
            per_args.fast_path_mapping_path = fast_mapping_paths[idx]
            per_args.data_root = data_roots[idx]
            if slow_mapping_paths is not None:
                per_args.slow_path_mapping_path = slow_mapping_paths[idx]
                per_args.slow_path_data_root = slow_data_roots[idx]
            per_dataset_args.append(per_args)
        dataset_names = dataset_names or [None] * dataset_count
    else:
        dataset_names = dataset_names or [None] * dataset_count

    grouped_args, grouped_fast_types, grouped_names = _group_dataset_arguments(
        per_dataset_args, dataset_names
    )

    datasets = []
    for per_args in grouped_args:
        dataset = LazySupervisedVideoDataset(
            anno_path=per_args.annotation_path,
            fast_path_mapping_path=per_args.fast_path_mapping_path,
            tokenizer=tokenizer,
            data_args=per_args,
        )
        datasets.append(dataset)

    if shuffle:
        alpha = float(mix_alpha)
        if abs(alpha - 1.0) > 1e-6:
            train_dataset = AlphaReweightedConcatDataset(
                datasets,
                modes=grouped_fast_types,
                alpha=alpha,
                seed=seed,
                batch_size=batch_size,
                world_size=world_size,
            )
        else:
            train_dataset = ShuffledConcatDataset(
                datasets,
                modes=grouped_fast_types,
                seed=seed,
                batch_size=batch_size,
                world_size=world_size,
            )
    else:
        train_dataset = ConcatDataset(datasets)

    try:
        dataset_lengths = [len(ds) for ds in datasets]
    except Exception:
        dataset_lengths = None

    group_summary = ", ".join(
        f"{name}({len(arg.annotation_path) if isinstance(arg.annotation_path, (list, tuple)) else 1})"
        for name, arg in zip(grouped_names, grouped_args)
    )

    if isinstance(train_dataset, AlphaReweightedConcatDataset):
        print(
            "[DataDebug] Alpha dataset summary | "
            f"grouped={group_summary} | "
            f"per_dataset_lengths={dataset_lengths} | "
            f"target_counts={train_dataset.target_counts} | "
            f"sampled_counts={train_dataset.sampled_counts} | "
            f"length={len(train_dataset)} | "
            f"dropped_tail_blocks={train_dataset._dropped_tail_blocks} | "
            f"batch_size={batch_size} | world_size={world_size}"
        )
    elif isinstance(train_dataset, ShuffledConcatDataset):
        print(
            "[DataDebug] Shuffled dataset summary | "
            f"grouped={group_summary} | "
            f"per_dataset_lengths={dataset_lengths} | "
            f"length={len(train_dataset)} | "
            f"dropped_tail_blocks={train_dataset._dropped_tail_blocks} | "
            f"batch_size={batch_size} | world_size={world_size}"
        )
    else:
        print(
            "[DataDebug] Concat dataset summary | "
            f"grouped={group_summary} | "
            f"per_dataset_lengths={dataset_lengths} | "
            f"length={len(train_dataset)}"
        )
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=collator)


def train_unimvu_mix(attn_implementation=None):
    from utils import prepare_video_model_v2
    from libs.utils.model_trainer import VideoModelTrainer

    model_args, data_args, training_args = parse_argument_classes(sys.argv[1:], return_name=False)

    compute_dtype = (
        torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    model, tokenizer = prepare_video_model_v2(training_args, model_args, data_args, compute_dtype, attn_implementation)

    data_module = _build_training_dataset_grouped(
        tokenizer=tokenizer,
        data_args=data_args,
        seed=getattr(training_args, "seed", 0),
        batch_size=training_args.per_device_train_batch_size,
        world_size=getattr(training_args, "world_size", 1),
        shuffle=getattr(training_args, "shuffle", True),
        mix_alpha=getattr(training_args, "mix_sampling_alpha", 1.0),
    )

    train_dataset = data_module.get("train_dataset")
    if hasattr(train_dataset, "unique_modes") and len(train_dataset.unique_modes) > 1:
        training_args.shuffle = False
        training_args.dataloader_drop_last = True

    if isinstance(train_dataset, AlphaReweightedConcatDataset):
        print(
            f"Alpha sampling enabled with alpha={getattr(training_args, 'mix_sampling_alpha', 1.0):.4f}. "
            f"Planned counts per dataset: {train_dataset.target_counts}. "
            f"Initial sampled counts (after world-size alignment): {train_dataset.sampled_counts}."
        )

    trainer = VideoModelTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )
    try:
        train_dataloader = trainer.get_train_dataloader()
        dataloader_len = len(train_dataloader)
    except Exception as exc:
        train_dataloader = None
        dataloader_len = f"error:{exc}"
    if train_dataloader is not None and isinstance(train_dataloader, torch.utils.data.DataLoader):
        try:
            sampler_len = len(train_dataloader.sampler)
        except Exception:
            sampler_len = "unknown"
    else:
        sampler_len = "unknown"
    print(
        "[DataDebug] TrainDataLoader summary | "
        f"len={dataloader_len} | sampler_len={sampler_len} | "
        f"per_device_batch_size={training_args.per_device_train_batch_size} | "
        f"train_batch_size={training_args.train_batch_size} | "
        f"gradient_accumulation_steps={training_args.gradient_accumulation_steps} | "
        f"world_size={training_args.world_size}"
    )
    print(f"trainer.args.max_steps: {trainer.args.max_steps}")
    print("Using VideoModelTrainer with UniMVU mixed-dataset alpha sampling")

    if isinstance(train_dataset, (ShuffledConcatDataset, AlphaReweightedConcatDataset)):
        trainer.add_callback(_DatasetShuffleCallback(train_dataset))

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        torch.use_deterministic_algorithms(False)
        trainer.train()
    try:
        state = trainer.state
        print(
            "[TrainDebug] Completed train() | "
            f"global_step={state.global_step} | "
            f"max_steps={state.max_steps} | "
            f"epoch={state.epoch} | "
            f"train_runtime={state.train_runtime}"
        )
    except Exception as exc:
        print(f"[TrainDebug] Unable to read trainer state: {exc}")

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
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
            
    else:
        safe_save_model_for_hf_videotrainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train_unimvu_mix(attn_implementation="flash_attention_2")
