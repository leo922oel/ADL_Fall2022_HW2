from curses import use_default_colors
from dataclasses import dataclass, field
from transformers import (AutoConfig,
    AutoTokenizer,
    # DataCollatorWithPadding,
    HfArgumentParser,
    # PreTrainedTokenizerFast,
    # TrainingArguments,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    # EvalPrediction,
    # default_data_collator,
    # set_seed
    )
from typing import Optional

@dataclass
class ModelArguments:
    pretrained_model_name_or_path:str = field(
        default='./pretrained/model', 
        metadata={
            "help": "Path to pretrained model"
        }
    )
    config_name:Optional[str] = field(
        default=None, 
        metadata={
            "help": "Pretrained config name"
        }
    )
    tokenizer_name:Optional[str] = field(
        default=None, 
        metadata={
            "help": "Pretrained tokenizer name"
        }
    )
    cache_dir:Optional[str] = field(
        default=None, 
        metadata={
            "help": "Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used"
        }
    )
    revision:str = field(
        default='main', 
        metadata={
            "help": "The specific model version to use"
        }
    )
    use_auth_token:bool = field(
        default=False, 
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script ""with private models)"
        }
    )

# def main():
parser = HfArgumentParser(ModelArguments)
model_args = parser.parse_args_into_dataclasses()

config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.pretrained_model_name_or_path,
    cache_dir = model_args.cache_dir,
    revision = model_args.revision,
    use_auth_token = True if model_args.use_auth_token else None,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.pretrained_model_name_or_path,
    cache_dir = model_args.cache_dir,
    revision = model_args.revision,
    use_auth_token = True if model_args.use_auth_token else None,
)

qa_model = AutoModelForQuestionAnswering.from_pretrained(
    model_args.pretrained_model_name_or_path,
    cache_dir = model_args.cache_dir,
    revision = model_args.revision,
    use_auth_token = True if model_args.use_auth_token else None,
)
mc_model = AutoModelForMultipleChoice.from_pretrained(
    model_args.pretrained_model_name_or_path,
    cache_dir = model_args.cache_dir,
    revision = model_args.revision,
    use_auth_token = True if model_args.use_auth_token else None,
)