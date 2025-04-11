# scripts/test_json_models.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
import random
import string # Needed for generate_random_text if copied directly
import json # Needed for value_correctness_reward
import logging
import sys
from tqdm import tqdm
import os

# Adjust path to import from src
# Assuming the script is run from the root directory 'open-r1'
# or that 'src' is in the PYTHONPATH
try:
    # Need to define generate_random_text here as it's not directly importable easily
    # without potentially complex path manipulation or making src a package.
    # Copied from src/open_r1/grpo.py
    def generate_random_text(min_length=5, max_length=15):
        """Generates a random string of letters, digits, spaces, single quotes, and double quotes.
        Aims for ~80% of strings to contain at least one space or quote."""
        length = random.randint(min_length, max_length)
        base_chars = string.ascii_letters + string.digits
        special_chars = ' ' + "'\""
        all_chars = base_chars + special_chars

        random_string_list = [random.choice(all_chars) for _ in range(length)]
        contains_special = any(c in special_chars for c in random_string_list)

        if not contains_special and random.random() < 0.8 and length > 0:
            replace_pos = random.randint(0, length - 1)
            char_to_insert = random.choice(special_chars)
            random_string_list[replace_pos] = char_to_insert

        return "".join(random_string_list)

    # Import necessary reward functions
    from src.open_r1.rewards import value_correctness_reward, key_exclusivity_reward, logger as reward_logger
except ImportError:
    # Fallback if running from scripts/ directory directly and src isn't found
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    try:
        from src.open_r1.rewards import value_correctness_reward, key_exclusivity_reward, logger as reward_logger
        # Define generate_random_text here as well for the fallback case
        def generate_random_text(min_length=5, max_length=15):
            length = random.randint(min_length, max_length)
            base_chars = string.ascii_letters + string.digits
            special_chars = ' ' + "'\""
            all_chars = base_chars + special_chars
            random_string_list = [random.choice(all_chars) for _ in range(length)]
            contains_special = any(c in special_chars for c in random_string_list)
            if not contains_special and random.random() < 0.8 and length > 0:
                replace_pos = random.randint(0, length - 1)
                char_to_insert = random.choice(special_chars)
                random_string_list[replace_pos] = char_to_insert
            return "".join(random_string_list)
    except ImportError as e:
        print(f"Error: Could not import required functions from src/open_r1. Ensure you run this script from the project root or have 'src' in your PYTHONPATH. Details: {e}")
        sys.exit(1)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
# Set reward logger level to avoid excessive output unless needed
# Check if reward_logger was successfully imported before setting level
if 'reward_logger' in globals():
    reward_logger.setLevel(logging.WARNING)


# --- Constants ---
ORIGINAL_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
FINETUNED_MODEL_PATH = "data/Qwen2.5-0.5B-JSON-GRPO" # Local path
CONFIG_PATH = "recipes/Qwen2.5-0.5B-Instruct/grpo/config_json.yaml"
NUM_SAMPLES = 100 # Reduced sample size for faster testing
MAX_NEW_TOKENS = 60 # Should be enough for {"data": "..."}

# --- Helper Functions ---

def load_model_and_tokenizer(model_id_or_path):
    """Loads model and tokenizer."""
    logger.info(f"Loading model and tokenizer from: {model_id_or_path}")
    try:
        # Force CPU usage
        device = torch.device("cpu")
        logger.info(f"Attempting to load model on: {device}")

        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.float32, # Use float32 for CPU
            trust_remote_code=True # Qwen models might require this
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path,
            trust_remote_code=True
        )
        # Set pad token if not set
        if tokenizer.pad_token is None:
             if tokenizer.eos_token:
                 tokenizer.pad_token = tokenizer.eos_token
                 logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
             else:
                 tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                 model.resize_token_embeddings(len(tokenizer))
                 logger.warning("Added new pad_token '[PAD]'")

        model.to(device)
        model.eval() # Set model to evaluation mode
        logger.info(f"Model loaded successfully on device: {model.device}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model/tokenizer from {model_id_or_path}: {e}", exc_info=True)
        raise

def load_system_prompt(config_path):
    """Loads system prompt from YAML config."""
    logger.info(f"Loading system prompt from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        system_prompt = config.get('system_prompt')
        if not system_prompt:
            logger.warning(f"System prompt not found or empty in {config_path}. Proceeding without it.")
            return None
        logger.info("System prompt loaded successfully.")
        return system_prompt
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading system prompt from {config_path}: {e}", exc_info=True)
        raise

def prepare_prompts(texts, system_prompt, tokenizer):
    """Prepares prompts in the required conversational format."""
    logger.info("Preparing prompts...")
    prompts_for_reward = []
    formatted_prompts_for_generation = []
    for text in texts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text})
        prompts_for_reward.append(messages) # Keep original structure for reward func

        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_prompts_for_generation.append(formatted_prompt)
        except Exception as e:
             logger.error(f"Error applying chat template for text '{text}': {e}. Using basic formatting.")
             prompt_str = ""
             if system_prompt:
                 prompt_str += f"System: {system_prompt}\n"
             prompt_str += f"User: {text}\nAssistant:"
             formatted_prompts_for_generation.append(prompt_str)

    logger.info(f"Prepared {len(formatted_prompts_for_generation)} prompts.")
    return prompts_for_reward, formatted_prompts_for_generation

def run_inference(model, tokenizer, prompts_for_generation):
    """Runs inference for the given model and prompts."""
    model_name = model.config._name_or_path # Get model name for description
    logger.info(f"Running inference for {len(prompts_for_generation)} prompts using {model_name}...")
    completions = []
    device = model.device
    with torch.no_grad():
        for prompt in tqdm(prompts_for_generation, desc=f"Generating with {model_name}"):
            inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=512).to(device)
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=False # Greedy decoding
                )
                # Decode the whole sequence and remove the prompt part
                full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Simple prompt removal (might need refinement based on template)
                completion_text = full_text[len(prompt):].strip() if full_text.startswith(prompt) else tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

                completions.append([{"role": "assistant", "content": completion_text}])
            except Exception as e:
                logger.error(f"Error during generation for prompt '{prompt[:50]}...': {e}", exc_info=True)
                completions.append([{"role": "assistant", "content": "[GENERATION ERROR]"}])

    logger.info(f"Inference completed for {model_name}.")
    return completions

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("--- Starting JSON Model Comparison Test ---")

    # 1. Load System Prompt
    system_prompt = load_system_prompt(CONFIG_PATH)

    # 2. Load Models and Tokenizers
    logger.info("--- Loading Original Model ---")
    original_model, original_tokenizer = load_model_and_tokenizer(ORIGINAL_MODEL_ID)
    logger.info("--- Loading Fine-tuned Model ---")
    finetuned_model, finetuned_tokenizer = load_model_and_tokenizer(FINETUNED_MODEL_PATH)

    # 3. Generate Test Data
    logger.info(f"--- Generating {NUM_SAMPLES} random text samples ---")
    test_texts = [generate_random_text() for _ in range(NUM_SAMPLES)]
    logger.info("Test data generated.")

    # 4. Prepare Prompts
    logger.info("--- Preparing prompts for models ---")
    # Use the fine-tuned tokenizer's template for consistency
    prompts_for_reward, prompts_for_generation = prepare_prompts(test_texts, system_prompt, finetuned_tokenizer)

    # 5. Run Inference
    logger.info("--- Running Inference: Original Model ---")
    original_completions = run_inference(original_model, original_tokenizer, prompts_for_generation)

    logger.info("--- Running Inference: Fine-tuned Model ---")
    finetuned_completions = run_inference(finetuned_model, finetuned_tokenizer, prompts_for_generation)

    # 6. Evaluate Results
    logger.info("--- Evaluating Results using Combined Score (Key Exclusivity AND Value Correctness) ---")

    # Make sure reward functions are available
    if 'value_correctness_reward' not in globals() or 'key_exclusivity_reward' not in globals():
        logger.error("Required reward functions (value_correctness_reward, key_exclusivity_reward) not found. Exiting.")
        sys.exit(1)

    # Calculate individual rewards first
    logger.info("Calculating individual rewards for Original Model...")
    original_val_rewards = value_correctness_reward(prompts=prompts_for_reward, completions=original_completions)
    original_key_rewards = key_exclusivity_reward(completions=original_completions)

    logger.info("Calculating individual rewards for Fine-tuned Model...")
    finetuned_val_rewards = value_correctness_reward(prompts=prompts_for_reward, completions=finetuned_completions)
    finetuned_key_rewards = key_exclusivity_reward(completions=finetuned_completions)

    # Calculate combined rewards
    original_combined_rewards = []
    finetuned_combined_rewards = []
    valid_evals_original = 0
    valid_evals_finetuned = 0

    for i in range(NUM_SAMPLES):
        # Original model combined score
        orig_val = original_val_rewards[i] if i < len(original_val_rewards) else None
        orig_key = original_key_rewards[i] if i < len(original_key_rewards) else None
        if orig_val is not None and orig_key is not None:
            original_combined_rewards.append(1.0 if orig_val == 1.0 and orig_key == 1.0 else 0.0)
            valid_evals_original += 1
        else:
            original_combined_rewards.append(None) # Mark as invalid if any component is invalid

        # Fine-tuned model combined score
        ft_val = finetuned_val_rewards[i] if i < len(finetuned_val_rewards) else None
        ft_key = finetuned_key_rewards[i] if i < len(finetuned_key_rewards) else None
        if ft_val is not None and ft_key is not None:
            finetuned_combined_rewards.append(1.0 if ft_val == 1.0 and ft_key == 1.0 else 0.0)
            valid_evals_finetuned += 1
        else:
            finetuned_combined_rewards.append(None) # Mark as invalid

    original_combined_valid = [r for r in original_combined_rewards if r is not None]
    finetuned_combined_valid = [r for r in finetuned_combined_rewards if r is not None]

    # 7. Calculate and Print Scores
    logger.info("--- Comparison Results ---")

    avg_original_score = sum(original_combined_valid) / len(original_combined_valid) if original_combined_valid else 0
    avg_finetuned_score = sum(finetuned_combined_valid) / len(finetuned_combined_valid) if finetuned_combined_valid else 0

    logger.info(f"Number of test samples: {NUM_SAMPLES}")
    logger.info(f"Number of valid evaluations (Original): {valid_evals_original}")
    logger.info(f"Number of valid evaluations (Fine-tuned): {valid_evals_finetuned}")
    logger.info("-" * 30)
    logger.info(f"Average Combined Score (Key Exclusivity AND Value Correctness)")
    logger.info(f"  Original Model:   {avg_original_score:.4f}")
    logger.info(f"  Fine-tuned Model: {avg_finetuned_score:.4f}")
    logger.info("-" * 30)

    # Optional: Print some examples with individual and combined rewards
    logger.info("\n--- Example Generations (First 5) ---")
    for i in range(min(5, NUM_SAMPLES)):
        logger.info(f"\nInput Text {i+1}: {test_texts[i]}")
        # Ensure completions list is not empty and has the expected structure
        original_output = original_completions[i][0]['content'] if i < len(original_completions) and original_completions[i] else "[NO OUTPUT]"
        finetuned_output = finetuned_completions[i][0]['content'] if i < len(finetuned_completions) and finetuned_completions[i] else "[NO OUTPUT]"

        orig_val_r = original_val_rewards[i] if i < len(original_val_rewards) else "N/A"
        orig_key_r = original_key_rewards[i] if i < len(original_key_rewards) else "N/A"
        orig_comb_r = original_combined_rewards[i] if i < len(original_combined_rewards) else "N/A"

        ft_val_r = finetuned_val_rewards[i] if i < len(finetuned_val_rewards) else "N/A"
        ft_key_r = finetuned_key_rewards[i] if i < len(finetuned_key_rewards) else "N/A"
        ft_comb_r = finetuned_combined_rewards[i] if i < len(finetuned_combined_rewards) else "N/A"

        logger.info(f"  Original Output:    {original_output}")
        logger.info(f"    Value Reward:     {orig_val_r}")
        logger.info(f"    Key Excl Reward:  {orig_key_r}")
        logger.info(f"    Combined Reward:  {orig_comb_r}")
        logger.info(f"  Fine-tuned Output:  {finetuned_output}")
        logger.info(f"    Value Reward:     {ft_val_r}")
        logger.info(f"    Key Excl Reward:  {ft_key_r}")
        logger.info(f"    Combined Reward:  {ft_comb_r}")


    logger.info("--- Test script finished ---")
