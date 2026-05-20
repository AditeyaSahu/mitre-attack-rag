"""
src/generator.py

LLM generator wrapper around the Groq API.

Uses the open-source Llama 3.3 70B model hosted on Groq's free-tier
inference endpoint. The model is open-source (Meta), Groq is just the
inference provider — this satisfies the assignment's "open-source LLM"
requirement.
"""

import logging
import os
import time
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
logger = logging.getLogger(__name__)


class GroqGenerator:
    """Thin, retry-capable wrapper around the Groq chat completions API."""

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to your .env file."
            )
        self.client = Groq(api_key=api_key)
        self.model = model or os.getenv(
            "GROQ_GENERATION_MODEL", "llama-3.3-70b-versatile"
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a chat completion request. Retries on transient failures
        with exponential backoff.
        """
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_error = e
                wait = self.retry_delay * (2 ** attempt)
                logger.warning(
                    f"Groq call failed (attempt {attempt + 1}/{self.max_retries}): "
                    f"{e}. Retrying in {wait:.1f}s..."
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Groq generation failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )