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
from multiprocessing import context
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
import datasets
from datasets import load_dataset

import transformers
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
        if self.test_file is not None:
            extension = self.test_file.split(".")[-1]
            assert extension in ["csv", "json"], f"`test_file` should be a csv or a json file. (error: {extension})"

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

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detect checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output dirt ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train."
            )

    logging.basicConfig(
        format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt = "%m/%d/%Y %H:%M:%S",
        handlers = [logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    if data_args.train_file is not None or data_args.valid_file is not None or data_args.test_file is not None:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]

        if data_args.valid_file is not None:
            data_files["valid"] = data_args.valid_file
            extension = data_args.valid_file.split(".")[-1]

        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        
        dataset = load_dataset(
            extension,
            data_files=data_files,
            field="data",
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        dataset = load_dataset(
            "swag",
            "regular",
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    with open(data_args.context_file, 'r') as f:
        context_file = json.load(f)
    
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir = model_args.cache_dir,
        revision = model_args.revision,
        use_auth_token = True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir = model_args.cache_dir,
        use_fast = model_args.use_fast_tokenizer,
        revision = model_args.revision,
        use_auth_token = True if model_args.use_auth_token else None,
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        model_args.model_name_or_path,
        from_tf = bool(".ckpt" in model_args.model_name_or_path),
        config = config,
        cache_dir = model_args.cache_dir,
        revision = model_args.revision,
        use_auth_token = True if model_args.use_auth_token else None,
    )

    question_name = "question"
    paragraphs_idx_name = "paragraphs"
    relevant_name = "relevant"

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    
    def preprocessing_train(examples):
        first_sentences = [[question] * 7 for question in examples[question_name]]
        paragraphs_idx = [idx + random.sample(set(range(len(context_file))) - set(idx), 7 - len(idx)) for idx in examples[paragraphs_idx_name]]
        second_sentences = [[context_file[i] for i in idx] for idx in paragraphs_idx]

         # Flatten out
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length" if data_args.pad_to_max_length else False,
        )
        # un-flatten
        tokenized_inputs = {k: [v[i:i+7] for i in range(0, len(v), 7)] for k, v in tokenized_examples.items()}
        if relevant_name in examples.keys():
            relevant = examples[relevant_name]
            tokenized_inputs["label"] = [paragraphs.index(rlv) for rlv, paragraphs in zip(relevant, paragraphs_idx)]
        else:
            tokenized_inputs["label"] = [0 for _ in paragraphs_idx]

        return tokenized_inputs
    
    def preprocessing_valid(examples):
        first_sentences = [[question] * 7 for question in examples[question_name]]
        paragraphs_idx = [idx + random.sample(set(range(len(context_file))) - set(idx), 7 - len(idx)) for idx in examples[paragraphs_idx_name]]
        second_sentences = [[context_file[i] for i in idx] for idx in paragraphs_idx]

         # Flatten out
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length" if data_args.pad_to_max_length else False,
        )
        # un-flatten
        tokenized_inputs = {k: [v[i:i+7] for i in range(0, len(v), 7)] for k, v in tokenized_examples.items()}
        if relevant_name in examples.keys():
            relevant = examples[relevant_name]
            tokenized_inputs["label"] = [paragraphs.index(rlv) for rlv, paragraphs in zip(relevant, paragraphs_idx)]
        else:
            tokenized_inputs["label"] = [0 for _ in paragraphs_idx]

        return tokenized_inputs
    
    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")

        train_examples = dataset["train"]
        if data_args.max_train_samples is not None:
            # We will select sample from whole data if agument is specified
            max_train_samples = min(len(train_examples), data_args.max_train_samples)
            train_examples = train_examples.select(range(max_train_samples))
        # Create train feature from dataset
        train_dataset = train_examples.map(
            preprocessing_train,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc = "Running tokenizer on train dataset",
        )
    if training_args.do_eval:
        if "valid" not in dataset:
            raise ValueError("--do_eval requires a valid dataset")

        valid_examples = dataset["valid"]
        if data_args.max_eval_samples is not None:
            # We will select sample from whole data if agument is specified
            max_eval_samples = min(len(valid_examples), data_args.max_eval_samples)
            valid_examples = valid_examples.select(range(max_eval_samples))
        # Create valid feature from dataset
        valid_dataset = valid_examples.map(
            preprocessing_valid,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc = "Running tokenizer on valid dataset",
        )
    if training_args.do_predict:
        if "test" not in dataset:
            raise ValueError("--do_predict requires a test dataset")

        test_examples = dataset["test"]
        if data_args.max_test_samples is not None:
            # We will select sample from whole data if agument is specified
            max_test_samples = min(len(test_examples), data_args.max_test_samples)
            test_examples = test_examples.select(range(max_test_samples))
        # Create test feature from dataset
        test_dataset = test_examples.map(
            preprocessing_valid,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc = "Running tokenizer on test dataset",
        )

    # collator.
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorForMultipleChoice(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

    def compute_metrics(prediction):
        preds, label_ids = prediction 
        preds = np.argmax(preds, axis=1)
        
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}
    


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()