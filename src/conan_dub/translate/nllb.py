"""NLLB translation adapter for Japanese to Italian conversion."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TranslationConfig:
    """Configuration for the NLLB translator."""

    model_id: str = "facebook/nllb-200-distilled-600M"
    device: str | int = "cpu"
    max_length: int = 512
    source_lang: str = "jpn_Jpan"
    target_lang: str = "ita_Latn"


class NLLBTranslator:
    """Translate text using Hugging Face transformers."""

    def __init__(self, config: TranslationConfig):
        self.config = config
        self._pipeline = None

    def _lazy_load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline  # type: ignore
        except ModuleNotFoundError:
            logger.warning("transformers not installed; translations will be echoed.")
            self._pipeline = None
            return

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_id)
        self._pipeline = pipeline(
            task="translation",
            model=model,
            tokenizer=tokenizer,
            device=self.config.device,
            src_lang=self.config.source_lang,
            tgt_lang=self.config.target_lang,
            max_length=self.config.max_length,
        )

    def translate_texts(
        self,
        texts: Iterable[str],
        dry_run: bool = False,
    ) -> List[str]:
        """Translate sentences into Italian; echoes originals during dry-run."""

        sentences = list(texts)
        if dry_run:
            logger.info("Dry-run translation for %d sentences", len(sentences))
            return sentences

        self._lazy_load()
        if self._pipeline is None:
            return self.translate_texts(sentences, dry_run=True)

        outputs = self._pipeline(sentences)
        return [entry["translation_text"] for entry in outputs]

