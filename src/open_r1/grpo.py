# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import logging
import os
import random
import string
import sys

import datasets
import torch
from datasets import Dataset, DatasetDict
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config


logger = logging.getLogger(__name__)


# --- Dynamic Dataset Generation ---

def generate_random_text(min_length=5, max_length=25):
    """Generates a random string of letters, digits, spaces, single quotes, and double quotes.
    Aims for ~80% of strings to contain at least one space or quote."""
    length = random.randint(min_length, max_length)
    base_chars = string.ascii_letters + string.digits
    special_chars = ' ' + "'\""
    all_chars = base_chars + special_chars

    # Generate the initial random string using all allowed characters
    random_string_list = [random.choice(all_chars) for _ in range(length)]

    # Check if it already contains a special character
    contains_special = any(c in special_chars for c in random_string_list)

    # If it doesn't contain a special char, and we hit the 80% chance, force one in
    # Ensure length > 0 before trying to replace (min_length=5 guarantees this)
    if not contains_special and random.random() < 0.8 and length > 0:
        # Choose a random position to replace
        replace_pos = random.randint(0, length - 1)
        # Choose a special character to insert
        char_to_insert = random.choice(special_chars)
        # Replace the character at that position
        random_string_list[replace_pos] = char_to_insert

    return "".join(random_string_list)

def create_dynamic_json_dataset(num_samples=400, prompt_column="instruction"):
    """Creates a DatasetDict with dynamically generated JSON data."""
    logger.info(f"Dynamically generating {num_samples} samples for JSON dataset...")
    data = []
    for _ in range(num_samples):
        instruction_text = generate_random_text()
        # We don't need an 'output' column for GRPO as it generates completions,
        # but we keep the 'instruction' column name consistent with the config.
        data.append({prompt_column: instruction_text})

    # Create a datasets.Dataset object
    dynamic_dataset = Dataset.from_list(data)

    # Create a DatasetDict (assuming train split only for simplicity, adjust if needed)
    # You might want a validation split too for monitoring, generated similarly.
    dataset_dict = DatasetDict({"train": dynamic_dataset})
    logger.info("Dynamic JSON dataset created.")
    return dataset_dict

# --- End Dynamic Dataset Generation ---


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)
    logger.info("--- Starting main function ---")

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset or generate dynamically
    logger.info("--- Loading/Generating dataset ---")
    if script_args.dataset_name == "dynamic_json_dataset":
        # Use the dynamic dataset generation function
        # You might want to pass num_samples from script_args if you add it there
        dataset = create_dynamic_json_dataset(prompt_column=script_args.dataset_prompt_column)
        logger.info("--- Dynamic dataset generated ---")
    else:
        # Load dataset from Hub or path
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
        logger.info(f"--- Dataset '{script_args.dataset_name}' loaded ---")


    ################
    # Load tokenizer
    ################
    logger.info("--- Loading tokenizer ---")
    tokenizer = get_tokenizer(model_args, training_args)

    # Get reward functions from the registry
    logger.info("--- Loading reward functions ---")
    reward_funcs = get_reward_funcs(script_args)
    logger.info("--- Reward functions loaded ---")


    # Format into conversation
    logger.info("--- Mapping dataset to conversation format ---")
    def make_conversation(example, prompt_column: str = script_args.dataset_prompt_column):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        if prompt_column not in example:
            raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")

        prompt.append({"role": "user", "content": example[prompt_column]})
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)
    logger.info("--- Dataset mapped ---")

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    #############################
    # Initialize the GRPO trainer
    #############################
    logger.info("--- Initializing GRPOTrainer ---")
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )
    logger.info("--- GRPOTrainer initialized ---")

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    logger.info(f"--- Starting training (resume_from_checkpoint={checkpoint}) ---")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    logger.info("--- Training finished ---")
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    logger.info("--- Saving model state ---")
    trainer.save_model(training_args.output_dir)
    logger.info(f"--- Model saved to {training_args.output_dir} ---")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        logger.info("--- Starting evaluation ---")
        metrics = trainer.evaluate()
        logger.info("--- Evaluation finished ---")
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    try:
        main(script_args, training_args, model_args)
        logger.info("--- Main function finished successfully ---")
    except Exception as e:
        logger.error(f"--- Main function failed with exception: {e} ---", exc_info=True)
        raise e
