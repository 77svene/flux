"""
Production-Ready Model Serving with vLLM Integration
Seamless export to optimized inference formats with automatic batching, continuous batching, and PagedAttention for high-throughput serving.
"""

import os
import json
import logging
import torch
from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

@dataclass
class VLLMExportConfig:
    """Configuration for vLLM model export."""
    model_name_or_path: str
    output_dir: str
    dtype: str = "auto"
    tensor_parallel_size: int = 1
    quantization: Optional[str] = None  # "awq", "gptq", "squeezellm", None
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = False
    revision: Optional[str] = None
    tokenizer_mode: str = "auto"
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256
    enable_prefix_caching: bool = False
    use_v2_block_manager: bool = False
    swap_space: int = 4  # GiB
    cpu_offload_gb: int = 0  # GiB
    enforce_eager: bool = False
    max_context_len_to_capture: int = 8192
    max_seq_len_to_capture: int = 8192
    disable_custom_all_reduce: bool = False
    enable_chunked_prefill: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VLLMExportConfig":
        return cls(**config_dict)


class VLLMModelExporter:
    """Export models to vLLM-compatible format with optimized inference configuration."""
    
    SUPPORTED_QUANTIZATION = ["awq", "gptq", "squeezellm"]
    
    def __init__(self, config: VLLMExportConfig):
        self.config = config
        self._validate_config()
        
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.config.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        
        if self.config.quantization and self.config.quantization not in self.SUPPORTED_QUANTIZATION:
            raise ValueError(f"Unsupported quantization: {self.config.quantization}. "
                           f"Supported: {self.SUPPORTED_QUANTIZATION}")
        
        if self.config.gpu_memory_utilization <= 0 or self.config.gpu_memory_utilization > 1:
            raise ValueError("gpu_memory_utilization must be between 0 and 1")
    
    def _detect_model_architecture(self, model_path: str) -> str:
        """Detect model architecture from config."""
        try:
            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=self.config.trust_remote_code,
                revision=self.config.revision
            )
            return config.architectures[0] if config.architectures else "unknown"
        except Exception as e:
            logger.warning(f"Failed to detect model architecture: {e}")
            return "unknown"
    
    def _get_optimal_tensor_parallel_size(self, model_path: str) -> int:
        """Automatically determine optimal tensor parallel size based on model size and available GPUs."""
        if self.config.tensor_parallel_size > 1:
            return self.config.tensor_parallel_size
        
        try:
            # Try to estimate model size
            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=self.config.trust_remote_code,
                revision=self.config.revision
            )
            
            # Count parameters from config
            hidden_size = getattr(config, "hidden_size", 4096)
            num_layers = getattr(config, "num_hidden_layers", 32)
            vocab_size = getattr(config, "vocab_size", 32000)
            intermediate_size = getattr(config, "intermediate_size", 11008)
            
            # Rough parameter estimation
            params = (vocab_size * hidden_size +  # Embedding
                     num_layers * (4 * hidden_size * hidden_size +  # Attention
                                  3 * hidden_size * intermediate_size))  # FFN
            
            # Convert to GB (assuming float16)
            model_size_gb = params * 2 / (1024 ** 3)
            
            # Get available GPU memory
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                # Use 80% of GPU memory for model
                available_memory = gpu_memory_gb * 0.8
                
                # Determine TP size
                if model_size_gb > available_memory:
                    tp_size = int(model_size_gb / available_memory) + 1
                    # Round to nearest power of 2
                    tp_size = 1 << (tp_size - 1).bit_length()
                    return min(tp_size, torch.cuda.device_count())
            
            return 1
            
        except Exception as e:
            logger.warning(f"Failed to auto-detect optimal TP size: {e}")
            return 1
    
    def _prepare_model_for_export(self, model_path: str) -> Tuple[Any, Any]:
        """Load and prepare model for export."""
        logger.info(f"Loading model from {model_path}")
        
        # Download if needed
        if not os.path.exists(model_path):
            logger.info("Model not found locally, downloading...")
            model_path = snapshot_download(
                repo_id=model_path,
                revision=self.config.revision,
                trust_remote_code=self.config.trust_remote_code
            )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.config.trust_remote_code,
            revision=self.config.revision,
            use_fast=True
        )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=self._get_torch_dtype(),
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=self.config.trust_remote_code,
            revision=self.config.revision,
            low_cpu_mem_usage=True
        )
        
        return model, tokenizer
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Convert dtype string to torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "auto": torch.float16 if torch.cuda.is_available() else torch.float32
        }
        return dtype_map.get(self.config.dtype, torch.float16)
    
    def _save_model_config(self, output_dir: str, model_path: str):
        """Save vLLM-specific configuration."""
        config_path = os.path.join(output_dir, "config.json")
        
        # Load original config
        original_config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=self.config.trust_remote_code,
            revision=self.config.revision
        )
        
        # Convert to dict and add vLLM-specific fields
        config_dict = original_config.to_dict()
        config_dict.update({
            "vllm_config": {
                "dtype": self.config.dtype,
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "quantization": self.config.quantization,
                "max_model_len": self.config.max_model_len,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "max_num_batched_tokens": self.config.max_num_batched_tokens,
                "max_num_seqs": self.config.max_num_seqs,
                "enable_prefix_caching": self.config.enable_prefix_caching,
                "use_v2_block_manager": self.config.use_v2_block_manager,
                "swap_space": self.config.swap_space,
                "cpu_offload_gb": self.config.cpu_offload_gb,
                "enforce_eager": self.config.enforce_eager,
                "max_context_len_to_capture": self.config.max_context_len_to_capture,
                "max_seq_len_to_capture": self.config.max_seq_len_to_capture,
                "disable_custom_all_reduce": self.config.disable_custom_all_reduce,
                "enable_chunked_prefill": self.config.enable_chunked_prefill
            }
        })
        
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Saved vLLM config to {config_path}")
    
    def _generate_vllm_launch_script(self, output_dir: str):
        """Generate a launch script for vLLM server."""
        script_content = f"""#!/bin/bash
# vLLM Server Launch Script
# Generated by flux vLLM Exporter

MODEL_DIR="{output_dir}"
TP_SIZE={self.config.tensor_parallel_size}
MAX_MODEL_LEN={self.config.max_model_len if self.config.max_model_len else '""'}
QUANTIZATION={self.config.quantization if self.config.quantization else '""'}

# Build command
CMD="python -m vllm.entrypoints.openai.api_server \\\\
    --model $MODEL_DIR \\\\
    --tensor-parallel-size $TP_SIZE \\\\
    --gpu-memory-utilization {self.config.gpu_memory_utilization} \\\\
    --max-num-seqs {self.config.max_num_seqs} \\\\
    --swap-space {self.config.swap_space} \\\\
    --disable-log-requests"

# Add optional parameters
if [ ! -z "$MAX_MODEL_LEN" ]; then
    CMD="$CMD --max-model-len $MAX_MODEL_LEN"
fi

if [ ! -z "$QUANTIZATION" ]; then
    CMD="$CMD --quantization $QUANTIZATION"
fi

if [ "{str(self.config.enable_prefix_caching).lower()}" = "true" ]; then
    CMD="$CMD --enable-prefix-caching"
fi

if [ "{str(self.config.use_v2_block_manager).lower()}" = "true" ]; then
    CMD="$CMD --use-v2-block-manager"
fi

if [ "{str(self.config.enforce_eager).lower()}" = "true" ]; then
    CMD="$CMD --enforce-eager"
fi

if [ "{str(self.config.enable_chunked_prefill).lower()}" = "true" ]; then
    CMD="$CMD --enable-chunked-prefill"
fi

echo "Starting vLLM server with command:"
echo "$CMD"
echo ""

# Execute
eval $CMD
"""
        
        script_path = os.path.join(output_dir, "launch_vllm.sh")
        with open(script_path, "w") as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        logger.info(f"Generated launch script at {script_path}")
    
    def _generate_docker_compose(self, output_dir: str):
        """Generate Docker Compose configuration for vLLM deployment."""
        docker_compose = f"""version: '3.8'

services:
  vllm-server:
    image: vllm/vllm-openai:latest
    ports:
      - "8000:8000"
    volumes:
      - ./{os.path.basename(output_dir)}:/model
    environment:
      - MODEL=/model
      - TENSOR_PARALLEL_SIZE={self.config.tensor_parallel_size}
      - GPU_MEMORY_UTILIZATION={self.config.gpu_memory_utilization}
      - MAX_MODEL_LEN={self.config.max_model_len if self.config.max_model_len else ''}
      - QUANTIZATION={self.config.quantization if self.config.quantization else ''}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: {self.config.tensor_parallel_size}
              capabilities: [gpu]
    command: >
      --model /model
      --tensor-parallel-size {self.config.tensor_parallel_size}
      --gpu-memory-utilization {self.config.gpu_memory_utilization}
      --max-num-seqs {self.config.max_num_seqs}
      --host 0.0.0.0
      --port 8000
"""
        
        compose_path = os.path.join(output_dir, "docker-compose.yml")
        with open(compose_path, "w") as f:
            f.write(docker_compose)
        
        logger.info(f"Generated Docker Compose at {compose_path}")
    
    def _generate_readme(self, output_dir: str, model_path: str):
        """Generate README with usage instructions."""
        model_name = os.path.basename(model_path)
        readme_content = f"""# vLLM Exported Model: {model_name}

## Model Information
- **Original Model**: {model_path}
- **Export Format**: vLLM-compatible
- **Tensor Parallel Size**: {self.config.tensor_parallel_size}
- **Quantization**: {self.config.quantization or 'None'}
- **Data Type**: {self.config.dtype}

## Quick Start

### Option 1: Using vLLM Python API
```python
from vllm import LLM, SamplingParams

# Initialize the LLM
llm = LLM(
    model="{output_dir}",
    tensor_parallel_size={self.config.tensor_parallel_size},
    gpu_memory_utilization={self.config.gpu_memory_utilization}
)

# Create sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=512
)

# Generate text
prompts = ["Hello, my name is"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### Option 2: Using vLLM OpenAI-Compatible Server
```bash
# Start the server
./launch_vllm.sh

# Or use Docker Compose
docker-compose up -d
```

### Option 3: Using curl
```bash
curl http://localhost:8000/v1/completions \\
  -H "Content-Type: application/json" \\
  -d '{{
    "model": "{output_dir}",
    "prompt": "Hello, my name is",
    "max_tokens": 512,
    "temperature": 0.8
  }}'
```

## Configuration
Edit `vllm_config.json` to modify inference parameters:
- `tensor_parallel_size`: Number of GPUs for tensor parallelism
- `max_model_len`: Maximum sequence length
- `gpu_memory_utilization`: GPU memory usage target
- `quantization`: Quantization method (awq, gptq, squeezellm)

## Performance Optimization
- Use `enable_prefix_caching=True` for better performance with shared prefixes
- Adjust `max_num_seqs` based on your GPU memory
- Enable `use_v2_block_manager` for improved memory management

## Deployment
The exported model includes:
- `config.json`: Model configuration with vLLM settings
- `launch_vllm.sh`: Server launch script
- `docker-compose.yml`: Docker deployment configuration
- `vllm_config.json`: vLLM-specific configuration

## Monitoring
Access the vLLM metrics endpoint:
```bash
curl http://localhost:8000/metrics
```
"""
        
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(readme_content)
        
        logger.info(f"Generated README at {readme_path}")
    
    def export(self) -> str:
        """Export model to vLLM-compatible format."""
        logger.info(f"Starting vLLM export for {self.config.model_name_or_path}")
        
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect optimal tensor parallel size
        if self.config.tensor_parallel_size == 1:
            self.config.tensor_parallel_size = self._get_optimal_tensor_parallel_size(
                self.config.model_name_or_path
            )
            logger.info(f"Auto-detected tensor_parallel_size: {self.config.tensor_parallel_size}")
        
        # Load and prepare model
        model, tokenizer = self._prepare_model_for_export(self.config.model_name_or_path)
        
        # Save model and tokenizer
        logger.info(f"Saving model to {output_dir}")
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        
        # Save configuration files
        self._save_model_config(str(output_dir), self.config.model_name_or_path)
        
        # Save vLLM export configuration
        export_config_path = output_dir / "vllm_export_config.json"
        self.config.save(str(export_config_path))
        
        # Generate deployment files
        self._generate_vllm_launch_script(str(output_dir))
        self._generate_docker_compose(str(output_dir))
        self._generate_readme(str(output_dir), self.config.model_name_or_path)
        
        logger.info(f"Successfully exported model to {output_dir}")
        logger.info(f"Tensor parallel size: {self.config.tensor_parallel_size}")
        logger.info(f"Quantization: {self.config.quantization or 'None'}")
        
        return str(output_dir)


def export_to_vllm(
    model_name_or_path: str,
    output_dir: str,
    dtype: str = "auto",
    tensor_parallel_size: int = 1,
    quantization: Optional[str] = None,
    max_model_len: Optional[int] = None,
    gpu_memory_utilization: float = 0.9,
    trust_remote_code: bool = False,
    **kwargs
) -> str:
    """
    One-click export to vLLM-compatible format.
    
    Args:
        model_name_or_path: HuggingFace model name or path
        output_dir: Output directory for exported model
        dtype: Data type (auto, float16, bfloat16, float32)
        tensor_parallel_size: Number of GPUs for tensor parallelism (1 for auto-detect)
        quantization: Quantization method (awq, gptq, squeezellm)
        max_model_len: Maximum sequence length
        gpu_memory_utilization: Target GPU memory utilization (0-1)
        trust_remote_code: Trust remote code in model
        **kwargs: Additional vLLM configuration parameters
    
    Returns:
        Path to exported model directory
    """
    config = VLLMExportConfig(
        model_name_or_path=model_name_or_path,
        output_dir=output_dir,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        quantization=quantization,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=trust_remote_code,
        **kwargs
    )
    
    exporter = VLLMModelExporter(config)
    return exporter.export()


def batch_export_to_vllm(
    model_configs: List[Dict[str, Any]],
    base_output_dir: str = "./vllm_exports"
) -> List[str]:
    """
    Batch export multiple models to vLLM format.
    
    Args:
        model_configs: List of model configuration dictionaries
        base_output_dir: Base output directory
    
    Returns:
        List of exported model paths
    """
    exported_paths = []
    
    for i, config_dict in enumerate(model_configs):
        model_name = config_dict.get("model_name_or_path", f"model_{i}")
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        output_dir = os.path.join(base_output_dir, safe_name)
        
        config_dict["output_dir"] = output_dir
        config = VLLMExportConfig.from_dict(config_dict)
        
        try:
            exporter = VLLMModelExporter(config)
            path = exporter.export()
            exported_paths.append(path)
            logger.info(f"Successfully exported {model_name}")
        except Exception as e:
            logger.error(f"Failed to export {model_name}: {e}")
    
    return exported_paths


def estimate_model_size(model_name_or_path: str) -> Dict[str, Any]:
    """
    Estimate model size and recommend vLLM configuration.
    
    Args:
        model_name_or_path: Model name or path
    
    Returns:
        Dictionary with size estimates and recommendations
    """
    try:
        config = AutoConfig.from_pretrained(model_name_or_path)
        
        # Extract model dimensions
        hidden_size = getattr(config, "hidden_size", 4096)
        num_layers = getattr(config, "num_hidden_layers", 32)
        vocab_size = getattr(config, "vocab_size", 32000)
        intermediate_size = getattr(config, "intermediate_size", 11008)
        num_attention_heads = getattr(config, "num_attention_heads", 32)
        
        # Estimate parameters
        embedding_params = vocab_size * hidden_size
        attention_params = num_layers * (4 * hidden_size * hidden_size)
        ffn_params = num_layers * (3 * hidden_size * intermediate_size)
        total_params = embedding_params + attention_params + ffn_params
        
        # Memory estimates (in GB)
        fp16_memory = total_params * 2 / (1024 ** 3)
        int8_memory = total_params * 1 / (1024 ** 3)
        int4_memory = total_params * 0.5 / (1024 ** 3)
        
        # Get GPU info
        gpu_info = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    "id": i,
                    "name": gpu_props.name,
                    "memory_gb": gpu_props.total_memory / (1024 ** 3)
                })
        
        # Recommend configuration
        recommendations = {
            "model_size_gb": fp16_memory,
            "recommended_quantization": None,
            "recommended_tp_size": 1,
            "estimated_max_batch_size": 32
        }
        
        if gpu_info:
            max_gpu_memory = max(gpu["memory_gb"] for gpu in gpu_info)
            
            if fp16_memory > max_gpu_memory * 0.8:
                if fp16_memory > max_gpu_memory * 2:
                    recommendations["recommended_quantization"] = "awq"
                    recommendations["recommended_tp_size"] = min(
                        int(fp16_memory / (max_gpu_memory * 0.8)) + 1,
                        len(gpu_info)
                    )
                else:
                    recommendations["recommended_tp_size"] = 2 if len(gpu_info) >= 2 else 1
            
            # Estimate batch size based on available memory
            available_memory = max_gpu_memory * 0.7  # Leave 30% for activations
            if recommendations["recommended_quantization"] == "awq":
                model_memory = int4_memory
            elif recommendations["recommended_quantization"] == "gptq":
                model_memory = int4_memory
            else:
                model_memory = fp16_memory
            
            # Each sequence in batch uses memory proportional to sequence length
            # Assuming average sequence length of 512
            seq_memory = 512 * hidden_size * 2 / (1024 ** 3)  # Rough estimate
            recommendations["estimated_max_batch_size"] = int(
                (available_memory - model_memory) / seq_memory
            )
        
        return {
            "architecture": config.architectures[0] if hasattr(config, "architectures") else "unknown",
            "parameters": total_params,
            "parameters_human": f"{total_params / 1e9:.2f}B",
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "memory_estimates": {
                "fp16_gb": fp16_memory,
                "int8_gb": int8_memory,
                "int4_gb": int4_memory
            },
            "gpu_info": gpu_info,
            "recommendations": recommendations
        }
        
    except Exception as e:
        return {"error": str(e)}


# Integration with existing flux modules
def integrate_with_training_pipeline(
    checkpoint_path: str,
    training_args: Dict[str, Any],
    export_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Integrate vLLM export with flux training pipeline.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        training_args: Training arguments from flux
        export_config: Optional export configuration overrides
    
    Returns:
        Path to exported vLLM model
    """
    # Extract relevant training args for vLLM config
    vllm_config = {
        "model_name_or_path": checkpoint_path,
        "output_dir": os.path.join(os.path.dirname(checkpoint_path), "vllm_export"),
        "trust_remote_code": training_args.get("trust_remote_code", False),
        "dtype": "auto"
    }
    
    # Apply export config overrides
    if export_config:
        vllm_config.update(export_config)
    
    # Export
    return export_to_vllm(**vllm_config)


# CLI interface for standalone usage
def main():
    """Command-line interface for vLLM export."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export models to vLLM-compatible format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export with auto-configuration
  python vllm_export.py --model meta-llama/Llama-2-7b-hf --output ./llama2-7b-vllm
  
  # Export with specific tensor parallel size and quantization
  python vllm_export.py --model meta-llama/Llama-2-70b-hf --output ./llama2-70b-vllm \\
    --tensor-parallel-size 4 --quantization awq
  
  # Batch export from config file
  python vllm_export.py --batch-config models.json --base-output ./vllm_exports
  
  # Estimate model size and get recommendations
  python vllm_export.py --estimate meta-llama/Llama-2-7b-hf
        """
    )
    
    parser.add_argument("--model", type=str, help="Model name or path")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--dtype", type=str, default="auto", 
                       choices=["auto", "float16", "bfloat16", "float32"],
                       help="Data type")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                       help="Tensor parallel size (1 for auto-detect)")
    parser.add_argument("--quantization", type=str, default=None,
                       choices=["awq", "gptq", "squeezellm"],
                       help="Quantization method")
    parser.add_argument("--max-model-len", type=int, default=None,
                       help="Maximum model length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                       help="GPU memory utilization (0-1)")
    parser.add_argument("--trust-remote-code", action="store_true",
                       help="Trust remote code")
    parser.add_argument("--batch-config", type=str,
                       help="JSON file with batch export configurations")
    parser.add_argument("--base-output", type=str, default="./vllm_exports",
                       help="Base output directory for batch export")
    parser.add_argument("--estimate", type=str,
                       help="Estimate model size and get recommendations")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    if args.estimate:
        # Estimate model size
        info = estimate_model_size(args.estimate)
        print(json.dumps(info, indent=2))
        return
    
    if args.batch_config:
        # Batch export
        with open(args.batch_config, "r") as f:
            model_configs = json.load(f)
        
        paths = batch_export_to_vllm(model_configs, args.base_output)
        print(f"Exported {len(paths)} models to {args.base_output}")
        return
    
    if not args.model or not args.output:
        parser.error("--model and --output are required for single export")
    
    # Single export
    path = export_to_vllm(
        model_name_or_path=args.model,
        output_dir=args.output,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        quantization=args.quantization,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code
    )
    
    print(f"Successfully exported model to {path}")
    print(f"Launch server with: ./launch_vllm.sh")


if __name__ == "__main__":
    main()