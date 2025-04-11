# scripts/test_custom_input_models.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import sys
import argparse
import os
from tqdm import tqdm

# Adjust path to import from src if needed
try:
    from src.open_r1.rewards import logger as reward_logger
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    try:
        from src.open_r1.rewards import logger as reward_logger
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
if 'reward_logger' in globals():
    reward_logger.setLevel(logging.WARNING)

# --- Constants ---
ORIGINAL_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
FINETUNED_MODEL_PATH = "data/Qwen2.5-0.5B-JSON-GRPO"  # Local path
MAX_NEW_TOKENS = 60  # Should be enough for short responses

# --- Helper Functions ---
def load_model_and_tokenizer(model_id_or_path):
    """Loads model and tokenizer."""
    logger.info(f"Loading model and tokenizer from: {model_id_or_path}")
    try:
        device = torch.device("cpu")
        logger.info(f"Attempting to load model on: {device}")

        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))
                logger.warning("Added new pad_token '[PAD]'")

        model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully on device: {model.device}")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model/tokenizer from {model_id_or_path}: {e}", exc_info=True)
        raise

def prepare_prompts(texts, tokenizer):
    """Prepares prompts in the required conversational format without system prompt."""
    logger.info("Preparing prompts...")
    formatted_prompts_for_generation = []
    for text in texts:
        messages = [{"role": "user", "content": text}]
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_prompts_for_generation.append(formatted_prompt)
        except Exception as e:
            logger.error(f"Error applying chat template for text '{text}': {e}. Using basic formatting.")
            prompt_str = f"User: {text}\nAssistant:"
            formatted_prompts_for_generation.append(prompt_str)

    logger.info(f"Prepared {len(formatted_prompts_for_generation)} prompts.")
    return formatted_prompts_for_generation

def run_inference(model, tokenizer, prompts_for_generation):
    """Runs inference for the given model and prompts."""
    model_name = model.config._name_or_path
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
                    do_sample=False  # Greedy decoding
                )
                full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                completion_text = full_text[len(prompt):].strip() if full_text.startswith(prompt) else tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                completions.append(completion_text)
            except Exception as e:
                logger.error(f"Error during generation for prompt '{prompt[:50]}...': {e}", exc_info=True)
                completions.append("[GENERATION ERROR]")

    logger.info(f"Inference completed for {model_name}.")
    return completions

def read_inputs_from_file(file_path):
    """Reads input texts from a file, one per line."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        logger.error(f"Error reading input file {file_path}: {e}")
        sys.exit(1)

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("--- Starting Custom Input Model Comparison Test ---")

    parser = argparse.ArgumentParser(description="Test models with custom input.")
    parser.add_argument("--input", type=str, help="Custom input text for testing.")
    parser.add_argument("--file", type=str, help="Path to a file with input texts, one per line.")
    args = parser.parse_args()

    if not args.input and not args.file:
        logger.error("Error: You must provide either --input or --file argument.")
        sys.exit(1)

    if args.input and args.file:
        logger.error("Error: You cannot provide both --input and --file arguments. Choose one.")
        sys.exit(1)

    if args.input:
        test_texts = [args.input]
    else:
        test_texts = read_inputs_from_file(args.file)
        logger.info(f"Loaded {len(test_texts)} inputs from {args.file}")

    # 1. Load Models and Tokenizers
    logger.info("--- Loading Original Model ---")
    original_model, original_tokenizer = load_model_and_tokenizer(ORIGINAL_MODEL_ID)
    logger.info("--- Loading Fine-tuned Model ---")
    finetuned_model, finetuned_tokenizer = load_model_and_tokenizer(FINETUNED_MODEL_PATH)

    # 2. Prepare Prompts
    logger.info("--- Preparing prompts for models ---")
    prompts_for_generation = prepare_prompts(test_texts, finetuned_tokenizer)

    # 3. Run Inference
    logger.info("--- Running Inference: Original Model ---")
    original_completions = run_inference(original_model, original_tokenizer, prompts_for_generation)

    logger.info("--- Running Inference: Fine-tuned Model ---")
    finetuned_completions = run_inference(finetuned_model, finetuned_tokenizer, prompts_for_generation)

    # 4. Display Results
    logger.info("--- Comparison Results ---")
    for i, input_text in enumerate(test_texts):
        logger.info(f"\nInput Text {i+1}: {input_text}")
        logger.info(f"  Original Output:   {original_completions[i]}")
        logger.info(f"  Fine-tuned Output: {finetuned_completions[i]}")

    logger.info("--- Test script finished ---")
