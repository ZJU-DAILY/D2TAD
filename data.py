from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset, DistributedSampler


class ProtoSequenceDataset(TorchDataset):
    """Pack proto trajectory arrays into fixed-length token blocks."""

    def __init__(
        self,
        file_path: Path,
        block_size: int,
        pad_token: Optional[int] = None,
        eos_token: Optional[int] = None,
        add_eos: bool = True,
        mask_eos: bool = True,
        missing_placeholder_token: Optional[int] = None,
    ):
        self.file_path = Path(file_path).expanduser().resolve()
        if not self.file_path.exists():
            raise FileNotFoundError(f"Proto dataset not found: {self.file_path}")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if (pad_token is None) ^ (eos_token is None):
            raise ValueError("pad_token and eos_token must be provided together.")
        if pad_token is not None and pad_token == eos_token:
            raise ValueError("pad_token and eos_token must be different.")

        raw = np.load(self.file_path, allow_pickle=True)
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.missing_placeholder_token = (
            None if missing_placeholder_token is None else int(missing_placeholder_token)
        )
        if self.missing_placeholder_token is not None:
            if self.missing_placeholder_token < 0:
                raise ValueError("missing_placeholder_token must be non-negative.")
            if pad_token is not None and self.missing_placeholder_token == int(pad_token):
                raise ValueError("missing_placeholder_token must differ from pad_token.")
            if eos_token is not None and self.missing_placeholder_token == int(eos_token):
                raise ValueError("missing_placeholder_token must differ from eos_token.")

        if pad_token is None:
            flat_stream = self._flatten_sequences(
                raw,
                missing_placeholder_token=self.missing_placeholder_token,
            )
            usable_tokens = (len(flat_stream) // block_size) * block_size
            if usable_tokens == 0:
                raise ValueError(
                    f"Not enough tokens ({len(flat_stream)}) for block size {block_size}."
                )
            trimmed = np.asarray(flat_stream[:usable_tokens], dtype=np.int64)
            self.tokens = torch.from_numpy(trimmed.reshape(-1, block_size))
            self.valid_mask = None
            return

        tokens_list = [
            self._flatten_sequence(
                seq,
                missing_placeholder_token=self.missing_placeholder_token,
            )
            for seq in raw
        ]
        padded, masks = self._pad_and_mask(
            tokens_list,
            block_size=block_size,
            pad_token=int(pad_token),
            eos_token=int(eos_token),
            add_eos=add_eos,
            mask_eos=mask_eos,
        )
        self.tokens = torch.tensor(padded, dtype=torch.long)
        self.valid_mask = torch.tensor(masks, dtype=torch.bool)

    @staticmethod
    def _flatten_sequences(
        array: np.ndarray,
        missing_placeholder_token: Optional[int] = None,
    ) -> List[int]:
        stream: List[int] = []
        for seq in array:
            stream.extend(
                ProtoSequenceDataset._flatten_sequence(
                    seq,
                    missing_placeholder_token=missing_placeholder_token,
                )
            )
        return stream

    @staticmethod
    def _flatten_sequence(
        seq,
        missing_placeholder_token: Optional[int] = None,
    ) -> List[int]:
        tokens: List[int] = []
        seq_iter: Iterable = seq.tolist() if isinstance(seq, np.ndarray) else seq
        for token in seq_iter:
            ProtoSequenceDataset._append_token(
                token,
                tokens,
                missing_placeholder_token=missing_placeholder_token,
            )
        return tokens

    @staticmethod
    def _append_token(
        token,
        stream: List[int],
        missing_placeholder_token: Optional[int] = None,
    ) -> None:
        if isinstance(token, (int, np.integer)):
            stream.append(int(token))
            return

        token_list = token.tolist() if isinstance(token, np.ndarray) else token

        if ProtoSequenceDataset._looks_like_token_pair(token_list):
            stream.append(int(token_list[0]))
            return

        if ProtoSequenceDataset._looks_like_missing_placeholder_pair(token_list):
            if missing_placeholder_token is None:
                raise ValueError(
                    "missing_placeholder_token is required to load keep-slot placeholders."
                )
            stream.append(int(missing_placeholder_token))
            return

        if token is None:
            if missing_placeholder_token is None:
                raise ValueError(
                    "missing_placeholder_token is required to load keep-slot placeholders."
                )
            stream.append(int(missing_placeholder_token))
            return

        if isinstance(token_list, (list, tuple)):
            for sub in token_list:
                ProtoSequenceDataset._append_token(
                    sub,
                    stream,
                    missing_placeholder_token=missing_placeholder_token,
                )
            return

        raise TypeError(f"Unsupported token type: {type(token)}")

    @staticmethod
    def _looks_like_timed_item(token) -> bool:
        return (
            isinstance(token, (list, tuple))
            and len(token) == 2
            and isinstance(token[1], (list, tuple, np.ndarray))
        )

    @staticmethod
    def _looks_like_token_pair(token) -> bool:
        return (
            ProtoSequenceDataset._looks_like_timed_item(token)
            and isinstance(token[0], (int, np.integer))
        )

    @staticmethod
    def _looks_like_missing_placeholder_pair(token) -> bool:
        return ProtoSequenceDataset._looks_like_timed_item(token) and token[0] is None

    @staticmethod
    def _pad_and_mask(
        sequences: List[List[int]],
        block_size: int,
        pad_token: int,
        eos_token: int,
        add_eos: bool,
        mask_eos: bool,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        padded: List[List[int]] = []
        masks: List[List[int]] = []
        content_budget = max(block_size - (1 if add_eos else 0), 0)

        for seq in sequences:
            tokens = list(seq[:content_budget]) if content_budget > 0 else []
            if add_eos and len(tokens) < block_size:
                tokens.append(eos_token)
            tokens = tokens[:block_size]
            tokens.extend([pad_token] * (block_size - len(tokens)))

            specials = {pad_token}
            if mask_eos and add_eos:
                specials.add(eos_token)
            mask = [0 if token in specials else 1 for token in tokens]
            padded.append(tokens)
            masks.append(mask)
        return padded, masks

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = {"input_ids": self.tokens[idx]}
        if self.valid_mask is not None:
            example["valid_mask"] = self.valid_mask[idx]
        return example


def cycle_loader(dataloader, sampler=None):
    while True:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for batch in dataloader:
            yield batch


def _is_proto_dataset(name: str) -> bool:
    return str(name).strip().lower() == "proto"


def resolve_special_tokens(config) -> Tuple[int, int]:
    data_cfg = getattr(config, "data", None)
    pad_token = getattr(data_cfg, "pad_token_id", None) if data_cfg is not None else None
    eos_token = getattr(data_cfg, "eos_token_id", None) if data_cfg is not None else None
    vocab_size = int(config.tokens)

    if pad_token is None:
        pad_token = vocab_size - 1
    if eos_token is None:
        eos_token = vocab_size - 2

    pad_token = int(pad_token)
    eos_token = int(eos_token)
    if pad_token == eos_token:
        raise ValueError("pad_token_id and eos_token_id must be different.")
    if pad_token >= vocab_size or eos_token >= vocab_size:
        raise ValueError("pad/eos token ids must be smaller than config.tokens.")
    if pad_token < 0 or eos_token < 0:
        raise ValueError("pad/eos token ids must be non-negative.")
    return pad_token, eos_token


def resolve_missing_placeholder_token(config) -> Optional[int]:
    graph_type = str(getattr(getattr(config, "graph", None), "type", "")).lower()
    if graph_type != "absorb":
        return None

    resolved_token = int(config.tokens)
    vocab_limit = int(config.tokens) + 1
    resolved_token = int(resolved_token)
    if resolved_token < 0 or resolved_token >= vocab_limit:
        raise ValueError(f"missing_placeholder_token must be in [0, {vocab_limit}).")
    return resolved_token


def _build_proto_dataset(config, split: str, block_size: int) -> ProtoSequenceDataset:
    proto_cfg = getattr(config.data, "proto", None)
    if proto_cfg is None:
        raise ValueError("config.data.proto must be set for proto dataset.")

    base_dir = getattr(proto_cfg, "base_dir", None)
    if not base_dir:
        raise ValueError("config.data.proto.base_dir is required.")

    file_key = "train_file" if split == "train" else "valid_file"
    file_name = getattr(proto_cfg, file_key, None)
    if not file_name:
        raise ValueError(f"config.data.proto.{file_key} is required.")

    pad_token, eos_token = resolve_special_tokens(config)
    missing_token = resolve_missing_placeholder_token(config)
    mask_eos = bool(getattr(config.data, "mask_eos_token", True))
    file_path = Path(base_dir).expanduser().resolve() / str(file_name)
    return ProtoSequenceDataset(
        file_path=file_path,
        block_size=block_size,
        pad_token=pad_token,
        eos_token=eos_token,
        add_eos=True,
        mask_eos=mask_eos,
        missing_placeholder_token=missing_token,
    )


def get_dataloaders(config, distributed=True):
    if not _is_proto_dataset(config.data.train) or not _is_proto_dataset(config.data.valid):
        raise ValueError("Submission build supports only data.train=proto and data.valid=proto.")
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError("training.batch_size must divide ngpus * training.accum.")
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError("eval.batch_size must divide ngpus * training.accum.")

    train_set = _build_proto_dataset(config, split="train", block_size=config.model.length)
    valid_set = _build_proto_dataset(config, split="valid", block_size=config.model.length)

    train_sampler = DistributedSampler(train_set) if distributed else None
    valid_sampler = DistributedSampler(valid_set) if distributed else None
    train_batch_size = config.training.batch_size // (config.ngpus * config.training.accum)
    valid_batch_size = config.eval.batch_size // (config.ngpus * config.training.accum)
    num_workers = int(getattr(config.data, "num_workers", 4))

    train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=train_sampler is None,
        persistent_workers=num_workers > 0,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=valid_batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=valid_sampler is None,
        persistent_workers=num_workers > 0,
    )
    return cycle_loader(train_loader, train_sampler), cycle_loader(valid_loader, valid_sampler)
