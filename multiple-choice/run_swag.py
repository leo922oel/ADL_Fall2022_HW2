#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for multiple choice.
"""
# You can also adapt this script on your own multiple choice task. Pointers for this are left as comments.

import logging
import os
import sys
import json
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Union

import numpy as np
import random
import torch
from tqdm import tqdm
from datasets import load_dataset

from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import PaddingStrategy, check_min_version, send_example_telemetry

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.22.2")

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model"
        }
    )
    config_name: Optional[str] = field(
        default=None, 
        metadata={
            "help": "Pretrained config name"
        }
    )
    tokenizer_name: Optional[str] = field(
        default=None, 
        metadata={
            "help": "Pretrained tokenizer name"
        }
    )
    cache_dir: Optional[str] = field(
        default=None, 
        metadata={
            "help": "Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used"
            }
        )
    use_fast_tokenizer: bool = field(
        default=True, 
        metadata={
            "help": "Whether to use one of the fast tokenizer"
        }
    )
    revision: str = field(
        default='main', 
        metadata={
            "help": "The specific model version to use"
        }
    )
    use_auth_token: bool = field(
        default=False, 
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script ""with private models)"
        }
    )

@dataclass
class DataTrainingArguments:
    train_file: Optional[str] = field(
        default=None, 
        metadata={
            "help": "The input training data file (a text file)"
        }
    )
    valid_file: Optional[str] = field(
        default=None, 
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)"
        }
    )
    test_file: Optional[str] = field(
        default=None, 
        metadata={
            "help": "An optional input testing data file to evaluate the perplexity on (a text file)"
        }
    )
    context_file: Optional[str] = field(
        default=None, 
        metadata={
            "help": "An optional input context data file to evaluate the perplexity on (a text file)"
        }
    )
    output_file: Optional[str] = field(
        default=None, 
        metadata={
            "help": "An optional output file"
        }
    )
    overwrite_cache: bool = field(
        default=None, 
        metadata={
            "help": "Overwrite the cached training and evaluation sets"
        }
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, 
        metadata={
            "help": "The number of processes to use for the preprocessing"
        }
    )
    max_seq_length: Optional[int] = field(
        default=None, 
        metadata={
            "help": ("The maximum total input sequence length after tokenization. If passed, sequences longer "
            "than this will be truncated, sequences shorter will be padded.")
        }
    )
    pad_to_max_length: bool = field(
        default=False, 
        metadata={
            "help": ("Whether to pad all samples to the maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU.")
        }
    )
    max_train_samples: Optional[int] = field(
        default=None, 
        metadata={
            "help": ("For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set.")
        }
    )
    max_eval_samples: Optional[int] = field(
        default=None, 
        metadata={
            "help": ("For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set.")
        }
    )
    max_test_samples: Optional[int] = field(
        default=None, 
        metadata={
            "help": ("For debugging purposes or quicker training, truncate the number of testing examples to this "
            "value if set.")
        }
    )

    def __post_init__(self):
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], f"`train_file` should be a csv or a json file. (error: {extension})"
        if self.valid_file is not None:
            extension = self.valid_file.split(".")[-1]
            assert extension in ["csv", "json"], f"`valid_file` should be a csv or a json file. (error: {extension})"

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))
        # flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch