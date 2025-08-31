#!/usr/bin/env python3
"""
Convert SwissAI models to Apertus models
"""

import argparse
import sys
import os
import json
import torch
from pathlib import Path
import shutil

from transformers import AutoTokenizer, SwissAIConfig, SwissAIForCausalLM, ApertusConfig, ApertusForCausalLM

PAD_TOKEN_ID = 3
EOS_TOKEN_ID = 2
BOS_TOKEN_ID = 1
ROPE_THETA = {
    "base": 500000,
    "post-train": 12000000
}

def check_model_type(model_path):
    """Check the actual model type from config.json."""
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config.get('model_type', 'unknown'), config.get('dtype', 'float32')
    return 'unknown', 'float32'


def convert_config(swissai_config: SwissAIConfig) -> ApertusConfig:
    """Convert a SwissAI configuration to an Apertus configuration."""
    config_dict = swissai_config.to_dict()
    
    config_dict["model_type"] = "apertus"
    if "architectures" in config_dict:
        config_dict["architectures"] = [arch.replace("SwissAI", "Apertus") for arch in config_dict["architectures"]]
    
    keys_to_remove = ["_name_or_path", "_commit_hash", "transformers_version"]
    for key in keys_to_remove:
        config_dict.pop(key, None)
    
    apertus_config = ApertusConfig(**config_dict)
    
    return apertus_config


def convert_model(swissai_model_path: str, apertus_output_path: str, force_convert: bool = False, path_to_tokenizer: str = None, is_base: bool = False):
    """
    Convert a SwissAI model to an Apertus model.
    
    Args:
        swissai_model_path: Path to the SwissAI model (local directory or HuggingFace model ID)
        apertus_output_path: Path where to save the Apertus model
        force_convert: Force conversion even if the model is already the target type
        path_to_tokenizer: Path to the tokenizer and chat template
        is_base: Whether the model is the base model
    """
    print(f"Loading model from: {swissai_model_path}")
    
    actual_model_type, original_dtype = check_model_type(swissai_model_path)
    print(f"Detected model type: {actual_model_type}")
    print(f"Original dtype: {original_dtype}")
    
    torch_dtype = torch.bfloat16 if original_dtype == "bfloat16" else torch.float32
    print(f"Using torch dtype: {torch_dtype}")
    
    if actual_model_type == "apertus" and not force_convert:
        print("Warning: This appears to already be an Apertus model!")
        print("Skipping conversion...")
        return
        
    else:

        try:
            swissai_config = SwissAIConfig.from_pretrained(swissai_model_path)
        except Exception as e:
            print(f"Warning: Could not load as SwissAI config: {e}")
            print("Attempting to load with AutoConfig and convert...")
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(swissai_model_path)
            config_dict = config.to_dict()
            config_dict["model_type"] = "swissai"
            swissai_config = SwissAIConfig(**config_dict)
        
        print("Converting configuration...")
        apertus_config = convert_config(swissai_config)
        
        apertus_config.torch_dtype = original_dtype
        
        print("Loading model as SwissAIForCausalLM...")
        try:
            swissai_model = SwissAIForCausalLM.from_pretrained(
                swissai_model_path,
                config=swissai_config,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            print(f"Error loading as SwissAI: {e}")
            print("Attempting to load with AutoModelForCausalLM...")
            from transformers import AutoModelForCausalLM
            swissai_model = AutoModelForCausalLM.from_pretrained(
                swissai_model_path,
                config=swissai_config,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
        
        print("Creating ApertusForCausalLM model...")
        apertus_model = ApertusForCausalLM(apertus_config)
                        
        print("Copying model weights...")
        swissai_state_dict = swissai_model.state_dict()
        
        if torch_dtype == torch.bfloat16:
            print("Converting weights to bfloat16...")
            for key in swissai_state_dict:
                if swissai_state_dict[key].dtype == torch.float32:
                    swissai_state_dict[key] = swissai_state_dict[key].to(torch.bfloat16)
        
        missing_keys, unexpected_keys = apertus_model.load_state_dict(swissai_state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys in state dict: {missing_keys[:5]}..." if len(missing_keys) > 5 else missing_keys)
            activation_keys = [k for k in missing_keys if 'act_fn' in k or 'activation' in k]
            if activation_keys:
                print(f"Missing activation function parameters: {len(activation_keys)} keys")
                print("These will be initialized with default values.")
        
        if unexpected_keys:
            print(f"Warning: Unexpected keys in state dict: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else unexpected_keys)
        
        if torch_dtype == torch.bfloat16:
            apertus_model = apertus_model.to(torch.bfloat16)
        
        if hasattr(swissai_model, 'generation_config'):
            apertus_model.generation_config = swissai_model.generation_config
        
        print(f"Saving Apertus model to: {apertus_output_path}")
        os.makedirs(apertus_output_path, exist_ok=True)
        
        apertus_model.config.torch_dtype = original_dtype
        
        apertus_model.save_pretrained(apertus_output_path)
    
    try:
        if path_to_tokenizer:
            print(f"Copying tokenizer from: {path_to_tokenizer}")
            for file in os.listdir(path_to_tokenizer):
                shutil.copy(os.path.join(path_to_tokenizer, file), os.path.join(apertus_output_path, file))
        else:
            print("Copying tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(swissai_model_path)
            tokenizer.save_pretrained(apertus_output_path)
    except Exception as e:
        print(f"Warning: Could not copy tokenizer: {e}")
        print("You may need to copy the tokenizer files manually.")
    
    config_path = Path(apertus_output_path) / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            saved_config = json.load(f)
        print(f"Saved model dtype: {saved_config.get('torch_dtype', 'not specified')}")
        saved_config["pad_token_id"] = PAD_TOKEN_ID
        saved_config["eos_token_id"] = EOS_TOKEN_ID
        saved_config["bos_token_id"] = BOS_TOKEN_ID

        if is_base:
            saved_config["rope_theta"] = ROPE_THETA["base"]
            if os.path.exists(os.path.join(apertus_output_path, "chat_template.jinja")):
                os.remove(os.path.join(apertus_output_path, "chat_template.jinja"))
        else:
            saved_config["rope_theta"] = ROPE_THETA["post-train"]

        if saved_config.get("model_type") != "apertus":
            saved_config["model_type"] = "apertus"
            if "architectures" in saved_config:
                saved_config["architectures"] = [arch.replace("SwissAI", "Apertus") for arch in saved_config["architectures"]]
            with open(config_path, 'w') as f:
                json.dump(saved_config, f, indent=2)
    
    print("Conversion completed successfully!")

    print('Loading Apertus model to verify conversion')
    model = ApertusForCausalLM.from_pretrained(apertus_output_path)
    print(f'Successfully loaded Apertus model with model_type: {model.config.model_type}')
    print(f'Hidden size: {model.config.hidden_size}')
    print(f'Num layers: {model.config.num_hidden_layers}')
    print(f'Num attention heads: {model.config.num_attention_heads}')
    print("Apertus model loaded successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SwissAI models to Apertus models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a local SwissAI model
  python convert_swissai_to_apertus.py /path/to/swissai-model /path/to/apertus-output
  
  # Convert a HuggingFace model
  python convert_swissai_to_apertus.py swiss-ai/SwissAI-8B ./apertus-8b-converted

  
  # Force conversion even if already Apertus
  python convert_swissai_to_apertus.py /path/to/model ./output --force
        """
    )
    
    parser.add_argument(
        "swissai_model_path",
        type=str,
        help="Path to the SwissAI model (local directory or HuggingFace model ID)"
    )
    
    parser.add_argument(
        "apertus_output_path",
        type=str,
        help="Path where to save the converted Apertus model"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force conversion even if the model is already the target type"
    )

    parser.add_argument(
        "--path-to-tokenizer",
        type=str,
        default=None,
        help="Path to the tokenizer and chat template"
    )

    parser.add_argument(
        "--is-base",
        action="store_true",
        help="Whether the model is the base model"
    )
    
    args = parser.parse_args()
    
    try:
        convert_model(
            args.swissai_model_path,
            args.apertus_output_path,
            args.force,
            args.path_to_tokenizer,
            args.is_base
        )
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
