"""
LLM-based title generator for analysis summaries.

Uses Ollama to generate concise, descriptive titles for bridge simulator
session analyses. Falls back to first-sentence extraction if unavailable.
"""

import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.llm.llm_client import LLMClient, get_default_client

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
TITLE_MAX_WORDS = int(os.getenv('TITLE_MAX_WORDS', '10'))
TITLE_MIN_WORDS = int(os.getenv('TITLE_MIN_WORDS', '4'))


class TitleGenerator:
    """
    Generates descriptive titles for analysis sessions using LLM.

    Uses Ollama for intelligent title generation with fallback to
    simple text extraction when LLM is unavailable.
    """

    def __init__(
        self,
        ollama_host: Optional[str] = None,
        ollama_model: Optional[str] = None,
        timeout: Optional[float] = None
    ):
        """
        Initialize title generator.

        Args:
            ollama_host: LLM API host URL (kept for backward compat)
            ollama_model: Model to use for generation (kept for backward compat)
            timeout: Request timeout in seconds
        """
        # Build LLMClient — only pass overrides when explicitly provided
        llm_kwargs: Dict[str, Any] = {}
        if ollama_host:
            llm_kwargs['base_url'] = f"{ollama_host.rstrip('/')}/v1"
        if ollama_model:
            llm_kwargs['model'] = ollama_model
        # Title generation uses a short timeout by default
        llm_kwargs['timeout'] = int(timeout) if timeout else 30

        self._llm = LLMClient(**llm_kwargs)
        self.model = self._llm.model

    async def check_llm_available(self) -> bool:
        """
        Check if LLM backend is available.

        Returns:
            True if backend is responding, False otherwise
        """
        return await self._llm.acheck_available()

    # Backward-compat alias
    check_ollama_available = check_llm_available

    def _build_prompt(
        self,
        full_text: str,
        speakers: List[Dict[str, Any]],
        duration_seconds: float
    ) -> str:
        """
        Build the LLM prompt for title generation.

        Args:
            full_text: Full transcript text
            speakers: List of speaker info dicts
            duration_seconds: Audio duration

        Returns:
            Formatted prompt string
        """
        # Truncate text if too long (keep first and last portions)
        max_chars = 2000
        if len(full_text) > max_chars:
            half = max_chars // 2
            full_text = full_text[:half] + " ... " + full_text[-half:]

        # Extract speaker roles if available
        roles = []
        for s in speakers:
            role = s.get('role') or s.get('inferred_role')
            if role and role not in roles:
                roles.append(role)

        roles_str = ", ".join(roles) if roles else "bridge crew"
        duration_str = f"{int(duration_seconds // 60)} minutes" if duration_seconds >= 60 else f"{int(duration_seconds)} seconds"

        prompt = f"""Generate a brief, descriptive title (4-10 words) for this bridge simulator session transcript.

Session info:
- Duration: {duration_str}
- Speakers: {len(speakers)} ({roles_str})

Transcript excerpt:
{full_text}

Generate ONLY the title, no quotes, no explanation. Focus on the main activity or scenario.
Examples of good titles:
- "Navigation drill with sensor calibration exercise"
- "Combat scenario against Kralien vessels"
- "Crew coordination practice docking sequence"
- "Bridge communication assessment training"

Title:"""

        return prompt

    async def generate_title_with_llm(
        self,
        full_text: str,
        speakers: List[Dict[str, Any]],
        duration_seconds: float
    ) -> Optional[str]:
        """
        Generate title using LLM.

        Args:
            full_text: Full transcript text
            speakers: List of speaker info
            duration_seconds: Audio duration

        Returns:
            Generated title or None if failed
        """
        try:
            prompt = self._build_prompt(full_text, speakers, duration_seconds)

            result = await self._llm.agenerate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=30,
                stop=["\n", ".", "Title:"],
            )

            if result is None:
                return None

            title = result.text.strip()

            # Clean up the title
            title = self._clean_title(title)

            if title and TITLE_MIN_WORDS <= len(title.split()) <= TITLE_MAX_WORDS:
                logger.info(f"Generated title: {title}")
                return title
            else:
                logger.warning(f"Title validation failed: '{title}'")
                return None

        except Exception as e:
            logger.warning(f"LLM title generation failed: {e}")
            return None

    def _clean_title(self, title: str) -> str:
        """
        Clean and normalize generated title.

        Args:
            title: Raw generated title

        Returns:
            Cleaned title
        """
        # Remove quotes
        title = title.strip('"\'')

        # Remove "Title:" prefix if present
        title = re.sub(r'^Title:\s*', '', title, flags=re.IGNORECASE)

        # Remove numbering like "1." or "1:"
        title = re.sub(r'^\d+[\.:]\s*', '', title)

        # Remove non-ASCII characters (multilingual model artifacts)
        title = re.sub(r'[^\x00-\x7F]+', '', title)

        # Remove trailing punctuation
        title = title.rstrip('.,;:')

        # Clean up extra whitespace
        title = ' '.join(title.split())

        # Capitalize first letter
        if title:
            title = title[0].upper() + title[1:]

        return title.strip()

    def extract_first_sentence(self, full_text: str) -> str:
        """
        Extract first meaningful sentence as fallback title.

        Args:
            full_text: Full transcript text

        Returns:
            First sentence or truncated text
        """
        if not full_text:
            return "Bridge session recording"

        # Clean up text
        text = full_text.strip()

        # Find first sentence end
        sentence_ends = ['.', '!', '?']
        first_end = len(text)
        for end in sentence_ends:
            pos = text.find(end)
            if pos > 0 and pos < first_end:
                first_end = pos

        # Extract first sentence
        sentence = text[:first_end].strip()

        # Truncate if too long
        words = sentence.split()
        if len(words) > TITLE_MAX_WORDS:
            sentence = ' '.join(words[:TITLE_MAX_WORDS]) + '...'
        elif len(words) < TITLE_MIN_WORDS and len(words) > 0:
            # Too short, add context
            sentence = f"Bridge session: {sentence}"

        # Default if empty
        if not sentence or len(sentence) < 5:
            sentence = "Bridge session recording"

        return sentence

    async def generate_title(
        self,
        full_text: str,
        speakers: List[Dict[str, Any]] = None,
        duration_seconds: float = 0
    ) -> str:
        """
        Generate a title for the analysis, with fallback.

        Attempts LLM generation first, falls back to first sentence
        extraction if Ollama is unavailable.

        Args:
            full_text: Full transcript text
            speakers: List of speaker info (optional)
            duration_seconds: Audio duration (optional)

        Returns:
            Generated title string
        """
        speakers = speakers or []

        # Try LLM first
        if await self.check_llm_available():
            llm_title = await self.generate_title_with_llm(
                full_text, speakers, duration_seconds
            )
            if llm_title:
                return llm_title

        # Fallback to first sentence extraction
        logger.info("Using fallback title extraction")
        return self.extract_first_sentence(full_text)


# Synchronous wrapper for non-async contexts
def generate_title_sync(
    full_text: str,
    speakers: List[Dict[str, Any]] = None,
    duration_seconds: float = 0
) -> str:
    """
    Synchronous wrapper for title generation.

    Args:
        full_text: Full transcript text
        speakers: List of speaker info
        duration_seconds: Audio duration

    Returns:
        Generated title string
    """
    async def _generate():
        generator = TitleGenerator()
        return await generator.generate_title(full_text, speakers, duration_seconds)

    return asyncio.run(_generate())
