"""CLI for running inference with vLLM and specified configuration."""

import json
import os
from pathlib import Path

import click
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger
from vllm import LLM, SamplingParams

from voice_inference.config import (
    is_wandb_artifact,
    load_config,
    load_config_from_wandb_artifact,
)
from voice_inference.hf import configure_hf, get_token
from voice_inference.logging import setup_logging
from voice_inference.utils import clean_wandb_run

ROOT_DIR = Path.cwd()
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"


@click.command()
@click.argument("config_path")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--log-file", help="Log file path")
@click.option("--max-tokens", type=int, help="Override max tokens for generation")
@click.option("--n-gpus", type=int, help="Number of GPUs to use for inference")
@click.option("--wandb-run-id", help="Attach to existing wandb run ID")
@click.option("--base-model", type=str, help="Override to use base model")
def main(
    config_path: str,
    log_level: str,
    log_file: str,
    max_tokens: int,
    n_gpus: int,
    wandb_run_id: str,
    base_model: str,
) -> None:
    """
    Run inference with vLLM and specified configuration.

    :param config_path: Path to configuration file
    :param log_level: Optional override for logging level
    :param log_file: Optional override for logging file path
    :param max_tokens: Optional override for max tokens for generation
    :param n_gpus: Optional override for number of GPUs to use for inference
    :param wandb_run_id: Option to attach to an existing wandb run ID.
    :param base_model: Optional override to use base model
    :return: None
    """
    setup_logging(log_level, log_file)

    # Detect local vs W&B artifact
    if is_wandb_artifact(config_path):
        logger.info("Detected W&B artifact config: {}", config_path)
        config_file = load_config_from_wandb_artifact(config_path)
        logger.info("Downloaded config to {}", str(config_path))
    else:
        config_file = Path(config_path).expanduser()

    try:
        logger.info("Loading config from {}", config_file)
        config = load_config(str(config_file))

        # Override sampling parameters if provided via CLI
        if max_tokens is not None:
            config.max_tokens = max_tokens
            logger.info("Overriding max_tokens to: {}", max_tokens)
        if n_gpus is not None:
            config.gpus = n_gpus
            logger.info("Overriding number of GPUs to: {}", n_gpus)
        if base_model is not None:
            config.model = base_model
            logger.info(
                "Overriding model to use base model for comparison: {}",
                base_model
            )

        logger.success("Config loaded successfully!")
        print("Current configuration:")
        print(config.model_dump_json(indent=2))
        print("")

    except Exception as e:
        logger.error("Failed to load config: {}", e)
        raise

    configure_hf(config.model)
    get_token()

    # Configure W&B run
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    if wandb_run_id:
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            id=wandb_run_id,
            resume="must",
        )
    else:
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
        )

    dataset = load_dataset(config.test_data, split=config.split)

    # Prepare messages for each example (system + user only)
    chat_prompts = [
        [m for m in example["messages"] if m["role"] != "assistant"]
        for example in dataset
    ]

    # Prepare reference answers in the same order
    references = [
        next(
            (m["content"] for m in example["messages"] if m["role"] == "assistant"),
            None
        )
        for example in dataset
    ]

    quantization = "bitsandbytes" if config.quantization and not base_model else None

    logger.info("Instantiating model {}", config.model)
    llm = LLM(
        model=config.model,
        quantization=quantization,
        tensor_parallel_size=config.gpus,
        dtype="bfloat16",
        max_model_len=4096,
    )

    sampling_params = SamplingParams(
        max_tokens=config.max_tokens,
    )

    logger.info("Running batched chat inference on {} prompts", len(chat_prompts))

    # Single batched call
    responses = llm.chat(chat_prompts, sampling_params)

    # Format results
    outputs = []
    for msgs, resp, ref in zip(chat_prompts, responses, references):
        text = resp.outputs[0].text.strip()
        outputs.append(
            {
                "messages": msgs,
                "generated_response": text,
                "reference_response": ref,
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / config.output_file

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    logger.success("Results successfully written to {}", output_path)

    # Log results as wandb artifact
    if wandb_run_id:
        artifact_type = "Results" if not base_model else "BaseResults"

        artifact = wandb.Artifact(
            name=f"{run.name}-{artifact_type}",
            type=artifact_type,
        )
        artifact.add_file(str(output_path))

        run.log_artifact(artifact)
        run.finish()

        clean_wandb_run(wandb_run_id)
