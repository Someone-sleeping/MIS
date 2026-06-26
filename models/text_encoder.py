import hashlib
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HashNGramTextEncoder(nn.Module):
    """A trainable CLIP-style text tower that does not require external packages.

    It hashes UTF-8 character n-grams into an embedding table, pools them, and
    projects to the model hidden dimension. The interface intentionally mirrors a
    CLIP text encoder so a pretrained backend can be swapped in later.
    """

    def __init__(
        self,
        output_dim: int,
        vocab_size: int = 8192,
        embedding_dim: int = 256,
        min_n: int = 1,
        max_n: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.min_n = min_n
        self.max_n = max_n
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def _hash(self, text: str) -> int:
        return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16) % self.vocab_size

    def _token_ids(self, text: str) -> List[int]:
        text = (text or "object").strip().lower()
        chars = list(text)
        ids = [self._hash("<bos>"), self._hash("<eos>")]
        if len(chars) == 0:
            return ids + [self._hash("object")]
        for n in range(self.min_n, self.max_n + 1):
            if len(chars) < n:
                continue
            for start in range(0, len(chars) - n + 1):
                ids.append(self._hash("".join(chars[start : start + n])))
        return ids

    def forward(self, texts: Iterable[str], device: Optional[torch.device] = None) -> torch.Tensor:
        text_list = list(texts)
        if len(text_list) == 0:
            raise ValueError("HashNGramTextEncoder received an empty text list.")
        device = device or self.embedding.weight.device
        pooled = []
        for text in text_list:
            token_ids = torch.as_tensor(self._token_ids(text), dtype=torch.long, device=device)
            pooled.append(self.embedding(token_ids).mean(dim=0))
        return self.proj(torch.stack(pooled, dim=0))


class TextEncoder(nn.Module):
    """Text encoder wrapper.

    backend="hash_ngram" is self-contained and trainable.
    backend="hf_clip" uses a local HuggingFace CLIP text model when transformers
    is installed and text_encoder.model_name_or_path is provided.
    """

    def __init__(
        self,
        output_dim: int,
        backend: str = "hash_ngram",
        model_name_or_path: Optional[str] = None,
        freeze: bool = True,
        vocab_size: int = 8192,
        embedding_dim: int = 256,
    ):
        super().__init__()
        self.backend = backend
        self.output_dim = output_dim
        self.freeze = freeze

        if backend == "hf_clip":
            try:
                from transformers import CLIPTextModel, CLIPTokenizerFast
            except ImportError as exc:
                raise ImportError("backend='hf_clip' requires transformers to be installed.") from exc
            if not model_name_or_path:
                raise ValueError("backend='hf_clip' requires model_name_or_path.")
            self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name_or_path)
            self.text_model = CLIPTextModel.from_pretrained(model_name_or_path)
            if freeze:
                for param in self.text_model.parameters():
                    param.requires_grad = False
            clip_dim = self.text_model.config.hidden_size
            self.proj = nn.Linear(clip_dim, output_dim)
        elif backend == "hash_ngram":
            self.text_model = HashNGramTextEncoder(
                output_dim=output_dim,
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
            )
        else:
            raise ValueError(f"Unknown text encoder backend: {backend}")

    def forward(self, texts: Iterable[str], device: Optional[torch.device] = None) -> torch.Tensor:
        if self.backend == "hf_clip":
            text_list = list(texts)
            device = device or next(self.parameters()).device
            encoded = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            if self.freeze:
                with torch.no_grad():
                    output = self.text_model(**encoded)
            else:
                output = self.text_model(**encoded)
            pooled = output.pooler_output
            return self.proj(pooled)
        return self.text_model(texts, device=device)

    def normalized(self, texts: Iterable[str], device: Optional[torch.device] = None) -> torch.Tensor:
        return F.normalize(self.forward(texts, device=device), dim=-1)
