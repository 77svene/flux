"""
Production-Ready Model Serving with vLLM Integration

Seamless export to optimized inference formats with automatic batching,
continuous batching, and PagedAttention for high-throughput serving.
"""

import os
import json
import logging
import asyncio
import argparse
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

try:
    from vllm import LLM, SamplingParams
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.openai.api_server import create_openai_api_server
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None

from flux.hparams import ModelArguments, DataArguments, TrainingArguments
from flux.model import load_tokenizer
from flux.extras.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ServingArguments:
    """Arguments for model serving configuration."""
    # Model configuration
    model_name_or_path: str = field(
        default="",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to adapter weights (LoRA, QLoRA, etc.)"}
    )
    
    # vLLM configuration
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of GPUs for tensor parallelism"}
    )
    gpu_memory_utilization: float = field(
        default=0.9,
        metadata={"help": "GPU memory utilization ratio (0-1)"}
    )
    max_model_len: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum model context length"}
    )
    quantization: Optional[str] = field(
        default=None,
        metadata={"help": "Quantization method (awq, gptq, squeezellm, etc.)"}
    )
    dtype: str = field(
        default="auto",
        metadata={"help": "Model data type (auto, half, float16, bfloat16, float32)"}
    )
    
    # Serving configuration
    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host to bind the server to"}
    )
    port: int = field(
        default=8000,
        metadata={"help": "Port to bind the server to"}
    )
    api_keys: Optional[List[str]] = field(
        default=None,
        metadata={"help": "API keys for authentication"}
    )
    cors_allow_origins: List[str] = field(
        default_factory=lambda: ["*"],
        metadata={"help": "Allowed CORS origins"}
    )
    cors_allow_methods: List[str] = field(
        default_factory=lambda: ["*"],
        metadata={"help": "Allowed CORS methods"}
    )
    cors_allow_headers: List[str] = field(
        default_factory=lambda: ["*"],
        metadata={"help": "Allowed CORS headers"}
    )
    
    # Inference configuration
    max_num_seqs: int = field(
        default=256,
        metadata={"help": "Maximum number of sequences per iteration"}
    )
    max_num_batched_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of batched tokens"}
    )
    scheduler_policy: str = field(
        default="fcfs",
        metadata={"help": "Scheduler policy (fcfs, priority)"}
    )
    
    # Export configuration
    export_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to export optimized model"}
    )
    export_format: str = field(
        default="vllm",
        metadata={"help": "Export format (vllm, hf, awq, gptq)"}
    )
    export_quantization: Optional[str] = field(
        default=None,
        metadata={"help": "Quantization method for export"}
    )
    export_max_shard_size: str = field(
        default="2GB",
        metadata={"help": "Maximum shard size for exported model"}
    )


class InferenceServer:
    """Production-ready inference server with vLLM integration."""
    
    def __init__(
        self,
        model_args: ModelArguments,
        serving_args: ServingArguments,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        """
        Initialize the inference server.
        
        Args:
            model_args: Model configuration arguments
            serving_args: Serving configuration arguments
            tokenizer: Optional tokenizer instance
        """
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. Please install vLLM to use the inference server: "
                "pip install vllm"
            )
        
        self.model_args = model_args
        self.serving_args = serving_args
        self.tokenizer = tokenizer
        self.llm_engine = None
        self.async_engine = None
        
        # Initialize tokenizer if not provided
        if self.tokenizer is None:
            self._load_tokenizer()
    
    def _load_tokenizer(self) -> None:
        """Load tokenizer from model path."""
        logger.info(f"Loading tokenizer from {self.model_args.model_name_or_path}")
        
        tokenizer_path = self.model_args.model_name_or_path
        if self.model_args.adapter_name_or_path:
            # For adapter models, use the base model tokenizer
            tokenizer_path = self.model_args.model_name_or_path
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=self.model_args.use_fast_tokenizer,
            padding_side="left",
            model_max_length=self.serving_args.max_model_len,
            **self.model_args.tokenizer_kwargs,
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _get_model_path(self) -> str:
        """Get the actual model path, handling adapters."""
        if self.model_args.adapter_name_or_path:
            # For adapter models, we need to merge or use the adapter
            # vLLM supports LoRA, so we can pass the adapter path
            return self.model_args.adapter_name_or_path
        return self.model_args.model_name_or_path
    
    def _create_engine_args(self) -> AsyncEngineArgs:
        """Create vLLM engine arguments from configuration."""
        engine_args = AsyncEngineArgs(
            model=self._get_model_path(),
            tokenizer=self.model_args.model_name_or_path,
            tensor_parallel_size=self.serving_args.tensor_parallel_size,
            gpu_memory_utilization=self.serving_args.gpu_memory_utilization,
            max_model_len=self.serving_args.max_model_len,
            quantization=self.serving_args.quantization,
            dtype=self.serving_args.dtype,
            max_num_seqs=self.serving_args.max_num_seqs,
            max_num_batched_tokens=self.serving_args.max_num_batched_tokens,
            scheduler_policy=self.serving_args.scheduler_policy,
            trust_remote_code=self.model_args.trust_remote_code,
            revision=self.model_args.model_revision,
        )
        return engine_args
    
    def load_model(self) -> None:
        """Load the model into vLLM engine."""
        logger.info("Loading model with vLLM...")
        
        engine_args = self._create_engine_args()
        
        # Create synchronous LLM engine for batch inference
        self.llm_engine = LLM(
            model=engine_args.model,
            tokenizer=engine_args.tokenizer,
            tensor_parallel_size=engine_args.tensor_parallel_size,
            gpu_memory_utilization=engine_args.gpu_memory_utilization,
            max_model_len=engine_args.max_model_len,
            quantization=engine_args.quantization,
            dtype=engine_args.dtype,
            trust_remote_code=engine_args.trust_remote_code,
            revision=engine_args.revision,
        )
        
        logger.info("Model loaded successfully with vLLM")
    
    async def load_async_model(self) -> None:
        """Load the model into async vLLM engine for serving."""
        logger.info("Loading model with async vLLM engine...")
        
        engine_args = self._create_engine_args()
        
        # Create async engine for serving
        self.async_engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        logger.info("Async model loaded successfully")
    
    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
        **kwargs,
    ) -> List[str]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts: List of input prompts
            sampling_params: vLLM sampling parameters
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        if self.llm_engine is None:
            self.load_model()
        
        if sampling_params is None:
            sampling_params = SamplingParams(**kwargs)
        
        outputs = self.llm_engine.generate(prompts, sampling_params)
        
        responses = []
        for output in outputs:
            responses.append(output.outputs[0].text)
        
        return responses
    
    async def generate_async(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate response asynchronously.
        
        Args:
            prompt: Input prompt
            sampling_params: vLLM sampling parameters
            request_id: Optional request ID for tracking
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        if self.async_engine is None:
            await self.load_async_model()
        
        if sampling_params is None:
            sampling_params = SamplingParams(**kwargs)
        
        results_generator = self.async_engine.generate(
            prompt, sampling_params, request_id
        )
        
        # Collect the final result
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
        
        if final_output is None:
            return ""
        
        return final_output.outputs[0].text
    
    def export_model(
        self,
        export_dir: Optional[str] = None,
        export_format: Optional[str] = None,
        quantization: Optional[str] = None,
    ) -> None:
        """
        Export model to optimized format for serving.
        
        Args:
            export_dir: Directory to export model to
            export_format: Export format (vllm, hf, awq, gptq)
            quantization: Quantization method
        """
        export_dir = export_dir or self.serving_args.export_dir
        export_format = export_format or self.serving_args.export_format
        quantization = quantization or self.serving_args.export_quantization
        
        if not export_dir:
            raise ValueError("export_dir must be specified for model export")
        
        logger.info(f"Exporting model to {export_dir} in {export_format} format")
        
        Path(export_dir).mkdir(parents=True, exist_ok=True)
        
        if export_format == "vllm":
            self._export_vllm_format(export_dir, quantization)
        elif export_format == "hf":
            self._export_hf_format(export_dir, quantization)
        elif export_format in ["awq", "gptq"]:
            self._export_quantized_format(export_dir, export_format, quantization)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        logger.info(f"Model exported successfully to {export_dir}")
    
    def _export_vllm_format(self, export_dir: str, quantization: Optional[str] = None) -> None:
        """Export model in vLLM-compatible format."""
        # vLLM can load models directly from HuggingFace format
        # We just need to save the model in a format vLLM can use
        
        from transformers import AutoModelForCausalLM
        
        logger.info("Loading model for vLLM export...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name_or_path,
            torch_dtype=torch.float16 if self.serving_args.dtype == "float16" else None,
            trust_remote_code=self.model_args.trust_remote_code,
            revision=self.model_args.model_revision,
        )
        
        # Save model in HuggingFace format (vLLM compatible)
        model.save_pretrained(
            export_dir,
            max_shard_size=self.serving_args.export_max_shard_size,
        )
        
        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(export_dir)
        
        # Save vLLM configuration
        vllm_config = {
            "model_type": "vllm",
            "tensor_parallel_size": self.serving_args.tensor_parallel_size,
            "quantization": quantization,
            "dtype": self.serving_args.dtype,
            "max_model_len": self.serving_args.max_model_len,
        }
        
        with open(os.path.join(export_dir, "vllm_config.json"), "w") as f:
            json.dump(vllm_config, f, indent=2)
    
    def _export_hf_format(self, export_dir: str, quantization: Optional[str] = None) -> None:
        """Export model in standard HuggingFace format."""
        from transformers import AutoModelForCausalLM
        
        logger.info("Loading model for HuggingFace export...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_args.model_name_or_path,
            torch_dtype=torch.float16 if self.serving_args.dtype == "float16" else None,
            trust_remote_code=self.model_args.trust_remote_code,
            revision=self.model_args.model_revision,
        )
        
        # Apply quantization if specified
        if quantization:
            model = self._apply_quantization(model, quantization)
        
        # Save model
        model.save_pretrained(
            export_dir,
            max_shard_size=self.serving_args.export_max_shard_size,
        )
        
        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(export_dir)
    
    def _export_quantized_format(
        self,
        export_dir: str,
        format_type: str,
        quantization: Optional[str] = None,
    ) -> None:
        """Export model in quantized format (AWQ, GPTQ, etc.)."""
        try:
            if format_type == "awq":
                from awq import AutoAWQForCausalLM
                
                logger.info("Loading model for AWQ quantization...")
                model = AutoAWQForCausalLM.from_pretrained(
                    self.model_args.model_name_or_path,
                    trust_remote_code=self.model_args.trust_remote_code,
                )
                
                # Quantize and save
                model.quantize(self.tokenizer, quant_config={
                    "zero_point": True,
                    "q_group_size": 128,
                    "w_bit": 4,
                    "version": "GEMM",
                })
                model.save_quantized(export_dir)
                
            elif format_type == "gptq":
                from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
                
                logger.info("Loading model for GPTQ quantization...")
                quantize_config = BaseQuantizeConfig(
                    bits=4,
                    group_size=128,
                    desc_act=False,
                )
                
                model = AutoGPTQForCausalLM.from_pretrained(
                    self.model_args.model_name_or_path,
                    quantize_config=quantize_config,
                    trust_remote_code=self.model_args.trust_remote_code,
                )
                
                # Quantize and save
                model.quantize(self.tokenizer)
                model.save_quantized(export_dir)
                
        except ImportError as e:
            logger.error(f"Required library not installed for {format_type}: {e}")
            raise
    
    def _apply_quantization(self, model, quantization: str):
        """Apply quantization to model."""
        if quantization == "8bit":
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
            model.quantization_config = quantization_config
            
        elif quantization == "4bit":
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model.quantization_config = quantization_config
        
        return model
    
    def create_openai_server(self):
        """Create OpenAI-compatible API server."""
        if self.async_engine is None:
            asyncio.run(self.load_async_model())
        
        return create_openai_api_server(
            engine=self.async_engine,
            served_model_name=self.model_args.model_name_or_path,
            api_keys=self.serving_args.api_keys,
            cors_allow_origins=self.serving_args.cors_allow_origins,
            cors_allow_methods=self.serving_args.cors_allow_methods,
            cors_allow_headers=self.serving_args.cors_allow_headers,
        )
    
    def start_server(self) -> None:
        """Start the inference server."""
        import uvicorn
        
        app = self.create_openai_server()
        
        logger.info(f"Starting inference server on {self.serving_args.host}:{self.serving_args.port}")
        
        uvicorn.run(
            app,
            host=self.serving_args.host,
            port=self.serving_args.port,
            log_level="info",
        )


def parse_serving_args() -> argparse.Namespace:
    """Parse command line arguments for serving."""
    parser = argparse.ArgumentParser(description="flux Inference Server")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pretrained model or model identifier")
    parser.add_argument("--adapter_name_or_path", type=str, default=None,
                        help="Path to adapter weights")
    
    # vLLM arguments
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="GPU memory utilization ratio")
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="Maximum model context length")
    parser.add_argument("--quantization", type=str, default=None,
                        help="Quantization method")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "half", "float16", "bfloat16", "float32"],
                        help="Model data type")
    
    # Serving arguments
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind the server to")
    parser.add_argument("--api_keys", type=str, nargs="+", default=None,
                        help="API keys for authentication")
    
    # Export arguments
    parser.add_argument("--export_dir", type=str, default=None,
                        help="Directory to export optimized model")
    parser.add_argument("--export_format", type=str, default="vllm",
                        choices=["vllm", "hf", "awq", "gptq"],
                        help="Export format")
    parser.add_argument("--export_quantization", type=str, default=None,
                        help="Quantization method for export")
    
    return parser.parse_args()


def main():
    """Main entry point for inference server."""
    args = parse_serving_args()
    
    # Create model arguments
    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        adapter_name_or_path=args.adapter_name_or_path,
        trust_remote_code=True,
    )
    
    # Create serving arguments
    serving_args = ServingArguments(
        model_name_or_path=args.model_name_or_path,
        adapter_name_or_path=args.adapter_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        quantization=args.quantization,
        dtype=args.dtype,
        host=args.host,
        port=args.port,
        api_keys=args.api_keys,
        export_dir=args.export_dir,
        export_format=args.export_format,
        export_quantization=args.export_quantization,
    )
    
    # Create inference server
    server = InferenceServer(model_args, serving_args)
    
    # Export model if requested
    if args.export_dir:
        server.export_model()
        logger.info("Model export completed")
        return
    
    # Start server
    server.start_server()


if __name__ == "__main__":
    main()