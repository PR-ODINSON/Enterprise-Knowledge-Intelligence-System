"""
Local LLM Module
Loads and manages a locally-hosted HuggingFace causal language model
(Mistral-7B-Instruct by default) for text generation.

Hardware notes:
  - GPU (CUDA):   Full float16 inference; optional 4-bit quantisation via
                  bitsandbytes to reduce VRAM to ~5 GB.
  - CPU only:     float32 inference; slower but functional for evaluation.
                  Consider a smaller model (e.g. phi-2, TinyLlama) for
                  practical CPU-only deployments.

The pipeline is initialised lazily on the first call to ``generate()`` to
avoid paying the model-loading cost at import time or during app startup.
"""

from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from app.config import (
    LLM_DEVICE,
    LLM_MAX_NEW_TOKENS,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_USE_4BIT,
)
from llm.prompt_templates import build_rag_prompt
from utils.logger import get_logger

logger = get_logger(__name__)


class LocalLLM:
    """
    Manages a locally-hosted causal LLM for answer generation in the RAG pipeline.

    Attributes:
        model_name:     HuggingFace model identifier.
        max_new_tokens: Maximum number of tokens to generate per call.
        temperature:    Sampling temperature (lower → more deterministic).
        use_4bit:       Enable 4-bit NF4 quantisation (requires CUDA +
                        bitsandbytes).
    """

    def __init__(
        self,
        model_name: str = LLM_MODEL_NAME,
        max_new_tokens: int = LLM_MAX_NEW_TOKENS,
        temperature: float = LLM_TEMPERATURE,
        use_4bit: bool = LLM_USE_4BIT,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_4bit = use_4bit
        self._pipeline = None   # Lazy-loaded

        logger.info(
            f"LocalLLM configured — model: {model_name}, "
            f"max_new_tokens: {max_new_tokens}, temperature: {temperature}, "
            f"4-bit: {use_4bit}"
        )

    # ------------------------------------------------------------------
    # Pipeline access (lazy loading)
    # ------------------------------------------------------------------

    @property
    def pipe(self):
        """Load the HuggingFace text-generation pipeline on first access."""
        if self._pipeline is None:
            self._load_pipeline()
        return self._pipeline

    def _load_pipeline(self) -> None:
        """
        Initialise the tokenizer, model, and text-generation pipeline.

        Steps:
          1. Resolve the compute device (auto-detects CUDA).
          2. Optionally build a 4-bit quantisation config.
          3. Load tokenizer and model.
          4. Wrap in a ``transformers.pipeline`` for convenient inference.
        """
        logger.info(
            f"Loading model '{self.model_name}' — "
            "this may take several minutes on the first run…"
        )

        device = self._resolve_device()
        logger.info(f"Compute device: {device}")

        if device != "cpu" and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            free_vram = (torch.cuda.get_device_properties(0).total_memory
                         - torch.cuda.memory_allocated(0)) / (1024 ** 3)
            logger.info(
                f"GPU: {gpu_name} | Total VRAM: {total_vram:.1f} GB "
                f"| Free VRAM: {free_vram:.1f} GB"
            )

        # --- Optional 4-bit quantisation --------------------------------
        quantization_config: Optional[BitsAndBytesConfig] = None
        if self.use_4bit and device != "cpu":
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                logger.info("4-bit NF4 quantisation enabled")
            except Exception as exc:
                logger.warning(f"4-bit quantisation unavailable: {exc}")

        # --- Tokenizer --------------------------------------------------
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True,
        )
        # Mistral tokenizer may lack a pad token; use eos_token as fallback
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # --- Model loading ----------------------------------------------
        model_kwargs: dict = {
            "torch_dtype": (
                torch.float16 if device != "cpu" else torch.float32
            ),
        }
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        # device_map="auto" lets Accelerate distribute across available GPUs/CPU
        if device != "cpu":
            model_kwargs["device_map"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        # --- Build pipeline ---------------------------------------------
        self._pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature if self.temperature > 0 else None,
            do_sample=self.temperature > 0,
            repetition_penalty=1.1,
            # Return only the newly generated text, not the echoed prompt
            return_full_text=False,
        )

        logger.info("Model pipeline ready for inference")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str) -> str:
        """
        Run a forward pass and return the generated text.

        Args:
            prompt: Fully formatted prompt string.

        Returns:
            Generated text with leading/trailing whitespace stripped.
        """
        logger.debug(f"Generating response — prompt length: {len(prompt)} chars")

        outputs = self.pipe(prompt)
        generated_text = outputs[0]["generated_text"].strip()

        logger.debug(f"Generated {len(generated_text)} chars")
        return generated_text

    def answer_question(self, context: str, question: str) -> str:
        """
        High-level helper: build a RAG prompt and generate an answer.

        Args:
            context:  Retrieved document context string.
            question: User's natural-language question.

        Returns:
            Generated answer string.
        """
        prompt = build_rag_prompt(context, question, use_instruction_format=True)
        return self.generate(prompt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device() -> str:
        """
        Determine the compute device to use for model inference.

        Returns:
            ``"cuda"`` when a CUDA GPU is configured and available,
            or ``"cpu"`` otherwise.

        Raises:
            RuntimeError: If LLM_DEVICE is set to ``"cuda"`` but no
                CUDA-capable GPU is detected.
        """
        if LLM_DEVICE == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "LLM_DEVICE is set to 'cuda' but torch.cuda.is_available() "
                    "returned False. Install a CUDA-enabled PyTorch wheel or "
                    "set LLM_DEVICE='cpu' in config.py."
                )
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA GPU detected: {gpu_name}")
            return "cuda"
        if LLM_DEVICE == "cpu":
            logger.info("Device explicitly set to CPU")
            return "cpu"
        # Legacy 'auto' fallback
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA GPU auto-detected: {gpu_name}")
            return "cuda"
        logger.info("No CUDA GPU detected — using CPU")
        return "cpu"
