"""
Embedding extraction pipeline for base and LoRA-finetuned MiniLM.

Produces frozen 384-dim sentence embeddings using mean pooling for both
models so that downstream comparisons are on equal footing.

Usage:
    extractor = EmbeddingExtractor("base")   # or "lora"
    embeddings = extractor.extract(["Hello world", "Foo bar"], batch_size=64)
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

from ..utils.config import MODEL_CONFIG, PROJECT_ROOT
from ..utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_MODEL_NAME = MODEL_CONFIG["base_model"]
HIDDEN_SIZE = MODEL_CONFIG["hidden_size"]
MAX_LENGTH = MODEL_CONFIG["max_length"]

FROZEN_MODEL_DIR = PROJECT_ROOT / "models" / "frozen_minilm"
LORA_MODEL_DIR = PROJECT_ROOT / "models" / "lora_minilm"


# ---------------------------------------------------------------------------
# Helper: mean-pool last hidden states
# ---------------------------------------------------------------------------
def _mean_pool(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool over token embeddings, respecting the attention mask."""
    mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_embeddings = (last_hidden_state * mask_expanded).sum(dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class EmbeddingExtractor:
    """
    Extract frozen sentence embeddings from either the base or LoRA MiniLM.

    Both variants use **mean pooling** over the last hidden states so that
    the resulting 384-d vectors are directly comparable.

    Parameters
    ----------
    model_type : str
        ``"base"`` for the frozen MiniLM or ``"lora"`` for the LoRA
        fine-tuned variant.
    device : str, optional
        PyTorch device string.  Defaults to ``"cuda"`` if available.
    """

    VALID_TYPES = ("base", "lora")

    def __init__(self, model_type: str, device: Optional[str] = None):
        if model_type not in self.VALID_TYPES:
            raise ValueError(
                f"model_type must be one of {self.VALID_TYPES}, got '{model_type}'"
            )
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading %s model on %s …", model_type, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        self._load_model()
        logger.info("Model ready.")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        if self.model_type == "base":
            self._load_base()
        else:
            self._load_lora()

    def _load_base(self) -> None:
        """Load frozen base MiniLM from the saved full state-dict."""
        base = AutoModel.from_pretrained(BASE_MODEL_NAME)

        # The saved model.pt is a *full* FrozenMiniLMClassifier state-dict
        # that has keys like ``encoder.…`` and ``classifier.…``.
        # We only need the encoder weights.
        state_dict = torch.load(
            FROZEN_MODEL_DIR / "model.pt",
            map_location=self.device,
            weights_only=True,
        )
        encoder_state = {
            k.replace("encoder.", "", 1): v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }
        base.load_state_dict(encoder_state)

        self.model = base.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _load_lora(self) -> None:
        """Load LoRA-adapted MiniLM from the PEFT adapter directory."""
        base = AutoModel.from_pretrained(BASE_MODEL_NAME)
        lora_model = PeftModel.from_pretrained(
            base, str(LORA_MODEL_DIR / "adapter")
        )
        lora_model = lora_model.merge_and_unload()  # merge for faster inference

        self.model = lora_model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Extract embeddings for a list of texts.

        Parameters
        ----------
        texts : list[str]
            Raw text strings.
        batch_size : int
            Batch size for GPU processing.
        show_progress : bool
            Whether to display a tqdm progress bar.

        Returns
        -------
        np.ndarray
            Array of shape ``(len(texts), 384)``.
        """
        all_embeddings: List[np.ndarray] = []
        batches = range(0, len(texts), batch_size)
        if show_progress:
            batches = tqdm(batches, desc=f"Extracting ({self.model_type})")

        for start in batches:
            batch_texts = texts[start : start + batch_size]
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            pooled = _mean_pool(outputs.last_hidden_state, attention_mask)
            all_embeddings.append(pooled.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def __repr__(self) -> str:
        return f"EmbeddingExtractor(model_type='{self.model_type}', device='{self.device}')"
