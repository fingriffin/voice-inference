"""Configuration management for VOICE inference package."""

import shutil
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from wandb import Api


class InferenceConfig(BaseModel):
    """Configuration for inference."""

    model: str = Field(..., description="Model name or path")
    test_data: str = Field(..., description="Path to the input data file")
    split: str = Field("test", description="Dataset split to use for inference")

    gpus: int = Field(1, description="Number of GPUs to use")
    quantization: Optional[str] = Field(
        None, description="Quantization method to use (must be '4bit' if provided)"
    )
    max_tokens: int = Field(2048, description="Maximum tokens to generate per prompt")
    temperature: float = Field(
        0.7,
        description="Temperature for sampling during inference"
    )

    output_file: str = Field(..., description="Path to save the output results")

    @field_validator("output_file")
    def ensure_output_dir_exists(cls, v: str) -> str:
        """
        Ensure the parent directory for output_file exists.

        :param v: The value of output_file.
        :return: The value of output_file.
        """
        output_path = Path(v)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("quantization")
    def validate_quantization(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate the quantization field.

        :param v: The value of quantization.
        :raises ValueError: If the quantization field is not 4bit when provided.
        :return: The validated value.
        """
        if v is not None and v != "4bit":
            raise ValueError("Quantization must be '4bit' if provided")
        return v


def load_config(config_path: str) -> InferenceConfig:
    """Load and validate configuration from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return InferenceConfig(**config_dict)

def is_wandb_artifact(uri: str) -> bool:
    """
    Check if the given URI points to a Weights & Biases artifact.

    :param uri: URI to check
    :return: True if URI is a W&B artifact, False otherwise
    """
    # Local file exists → not an artifact
    if Path(uri).expanduser().exists():
        return False

    # Must contain "/" and ":" in typical positions
    return ("/" in uri) and (":" in uri)

def load_config_from_wandb_artifact(uri: str) -> Path:
    """
    Load InferenceConfig YAML file from wandb artifact.

    :param uri: wandb artifact URI i.e. entity/project/artifact:version
    :return: path to downloaded YAML file
    """
    load_dotenv()
    api = Api()

    artifact = api.artifact(uri, type=None)

    configs_dir = Path("configs")
    configs_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(artifact.download())

    yaml_files = list(tmp_dir.glob("*.yaml")) + list(tmp_dir.glob("*.yml"))
    if not yaml_files:
        raise RuntimeError(f"No YAML files found inside artifact: {uri}")
    if len(yaml_files) > 1:
        raise RuntimeError(f"Multiple YAML files found inside artifact: {uri}")

    yaml_src = yaml_files[0]

    # Ensure local configs directory exists
    configs_dir = Path("configs")
    configs_dir.mkdir(parents=True, exist_ok=True)

    # Construct destination filename
    artifact_name = uri.split("/")[-1].split(":")[0]
    dest_path = configs_dir / f"{artifact_name}.yaml"

    # Save YAML file
    dest_path.write_text(yaml_src.read_text())

    # Delete the downloaded artifact directory
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Delete the wandb artifacts/ folder
    artifacts_dir = Path("artifacts")
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir, ignore_errors=True)

    return dest_path
