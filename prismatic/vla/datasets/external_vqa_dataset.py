"""
external_vqa_dataset.py

Unified PyTorch Dataset for external VQA co-training data (VSR, GQA, COCO-Karpathy).
Used for CO-VLA style co-training where VQA data trains the VLM backbone while
action data only trains the action head (stop-gradient).
"""

import json
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.vla.constants import IGNORE_INDEX


class ExternalVQADataset(Dataset):
    """Loads VSR, GQA, and COCO-Karpathy as a unified VQA co-training dataset.

    Implements uniform source sampling: each __getitem__ call picks one of the
    three sources based on `idx % 3`, ensuring balanced representation regardless
    of individual dataset sizes.
    """

    SOURCES = ["vsr", "gqa", "coco_caption"]

    def __init__(
        self,
        data_root: str,
        tokenizer: PreTrainedTokenizerBase,
        image_transform: Callable,
        prompt_builder_fn: Type[PromptBuilder],
    ):
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Load all three sources
        self.vsr_data = self._load_vsr()
        self.gqa_data = self._load_gqa()
        self.coco_data = self._load_coco()

        self.source_data = {
            "vsr": self.vsr_data,
            "gqa": self.gqa_data,
            "coco_caption": self.coco_data,
        }

        # Total = max across sources * 3 (so each source gets equal sampling)
        self._max_source_len = max(len(self.vsr_data), len(self.gqa_data), len(self.coco_data))

        print(f"[ExternalVQADataset] Loaded {len(self.vsr_data)} VSR, "
              f"{len(self.gqa_data)} GQA, {len(self.coco_data)} COCO samples")

    def _load_vsr(self) -> List[Dict]:
        """Load VSR train split: spatial true/false questions."""
        path = self.data_root / "vsr" / "train.jsonl"
        img_dir = self.data_root / "vsr" / "images"
        data = []
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                img_path = img_dir / row["image"]
                if not img_path.exists():
                    continue
                question = (
                    f'Is the following statement true or false?\n'
                    f'Statement: "{row["caption"]}"'
                )
                answer = "true" if row["label"] == 1 else "false"
                data.append({
                    "image_path": str(img_path),
                    "question": question,
                    "answer": answer,
                    "source": "vsr",
                })
        return data

    def _load_gqa(self) -> List[Dict]:
        """Load GQA testdev balanced questions."""
        questions_path = self.data_root / "gqa" / "testdev_balanced_questions.json"
        img_dir = self.data_root / "gqa" / "images"
        data = []
        with open(questions_path) as f:
            questions = json.load(f)
        for qid, row in questions.items():
            img_path = img_dir / f"{row['imageId']}.jpg"
            if not img_path.exists():
                continue
            data.append({
                "image_path": str(img_path),
                "question": row["question"],
                "answer": row["answer"],
                "source": "gqa",
            })
        return data

    def _load_coco(self) -> List[Dict]:
        """Load COCO-Karpathy test split: image captioning."""
        meta_path = self.data_root / "coco_karpathy" / "test_metadata.json"
        img_dir = self.data_root / "coco_karpathy" / "images"
        data = []
        with open(meta_path) as f:
            metadata = json.load(f)
        for row in metadata:
            img_path = img_dir / row["filename"]
            if not img_path.exists():
                continue
            data.append({
                "image_path": str(img_path),
                "question": "Describe this image in a single sentence.",
                "answer": row["caption"].strip(),
                "source": "coco_caption",
            })
        return data

    def __len__(self):
        return self._max_source_len * 3

    def __getitem__(self, idx):
        # Uniform source sampling
        source = self.SOURCES[idx % 3]
        source_data = self.source_data[source]
        sample_idx = (idx // 3) % len(source_data)
        sample = source_data[sample_idx]

        # Load and transform image
        image = Image.open(sample["image_path"]).convert("RGB")
        pixel_values = self.image_transform(image)

        # Build prompt using PurePromptBuilder
        prompt_builder = self.prompt_builder_fn("openvla")
        prompt_builder.add_turn("human", sample["question"])
        prompt_builder.add_turn("gpt", sample["answer"])
        prompt_text = prompt_builder.get_prompt()

        # Tokenize
        input_ids = self.tokenizer(prompt_text, add_special_tokens=True).input_ids

        # Build labels: mask everything except the answer tokens with IGNORE_INDEX
        labels = list(input_ids)

        # Find where the answer starts: tokenize just the question portion
        prompt_builder_q = self.prompt_builder_fn("openvla")
        prompt_builder_q.add_turn("human", sample["question"])
        # The question portion ends with "Out: " — get its token length
        question_prompt = prompt_builder_q.get_prompt()
        question_ids = self.tokenizer(question_prompt, add_special_tokens=True).input_ids
        question_len = len(question_ids)

        # Mask question tokens (keep answer tokens for CE loss)
        for i in range(min(question_len, len(labels))):
            labels[i] = IGNORE_INDEX

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "dataset_name": sample["source"],
        }


class VQACollator:
    """Collator for VQA batches: pads sequences and stacks images."""

    def __init__(self, model_max_length: int, pad_token_id: int):
        self.model_max_length = model_max_length
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [item["input_ids"][:self.model_max_length] for item in batch]
        labels = [item["labels"][:self.model_max_length] for item in batch]

        # Pad sequences
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        # Attention mask: 1 for real tokens, 0 for padding
        attention_mask = input_ids_padded.ne(self.pad_token_id).long()

        # Stack pixel values
        pixel_values = torch.stack([item["pixel_values"] for item in batch])

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids_padded,
            "labels": labels_padded,
            "attention_mask": attention_mask,
        }
