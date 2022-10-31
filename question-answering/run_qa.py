#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
Fine-tuning the library models for question answering using a slightly adapted version of the 🤗 Trainer.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import logging
from optparse import Option
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset

import evaluate
import transformers
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from utils_qa import postprocess_qa_predictions


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
    dataset_name: Optional[str] = field(
        default=None, 
        metadata={
            "help": "The name of the dataset to use"
        }
    )
    dataset_config_name: Optional[str] = field(
        default=None, 
        metadata={
            "help": "The config name of the dataset to use"
        }
    )
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
    version_2_with_negative: bool = field(
        default=False, 
        metadata={
            "help": "If true, some of the examples do not have an answer."
        }
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": ("The threshold used to select the null answer: if the best answer has a score that is less than "
                "the score of the null answer minus this threshold, the null answer is selected for this example. "
                "Only useful when `version_2_with_negative=True`.")
        }
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        }
    )
    n_best_size: int = field(
        default=20,
        metadata={
            "help": "The total number of n-best predictions to generate when looking for an answer."
        }
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": ("The maximum length of an answer that can be generated. This is needed because the start "
                "and end predictions are not conditioned on one another.")
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None and
            self.train_file is None and
            self.valid_file is None and
            self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a train/valid/test file")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], f"`train_file` should be a csv or a json file. (error: {extension})"
            if self.valid_file is not None:
                extension = self.valid_file.split(".")[-1]
                assert extension in ["csv", "json"], f"`valid_file` should be a csv or a json file. (error: {extension})"
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], f"`test_file` should be a csv or a json file. (error: {extension})"

def main():
    pass

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()