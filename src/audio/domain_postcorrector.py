"""
Domain-aware post-correction for transcription segments.

Layer 3: Dictionary-based phrase fixes for known Whisper errors.
Layer 5: LLM-powered semantic transcript cleanup via Ollama.

These corrections run AFTER TranscriptPostProcessor (merge/cleanup)
and BEFORE segment formatting, so fixes flow through to all
downstream analysis, narrative, and story generation.
"""

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv

from src.llm.llm_client import LLMClient

load_dotenv()

logger = logging.getLogger(__name__)


# ============================================================================
# Layer 3: Known phrase-level corrections
# ============================================================================

# Format: (regex_pattern, replacement, case_insensitive)
# These fix multi-word Whisper errors identified from accuracy analysis.
# Single-word domain vocabulary corrections are in whisper_transcriber.py
# (DOMAIN_CORRECTIONS) and run during transcription. These phrase-level
# corrections catch errors that slip through or require multi-word context.
PHRASE_CORRECTIONS: List[Tuple[str, Union[str, Callable], bool]] = [
    # Meaning-inverting errors from accuracy analysis
    (r'\bwith intolerance\b', 'within tolerance', True),
    (r'\bnod unkindly\b', 'not unkindly', True),
    (r'\bshit breeze\b', 'ship breathe', True),
    (r'\bship breathes\b', 'ship breathe', True),

    # Context-dependent single-word fixes (require preceding context)
    # "contact directed" → "contact detected", but "directed the crew" stays
    (r'\b(contact|anomaly|signal|vessel|object|threat|target|energy)\s+directed\b',
     lambda m: m.group().replace('directed', 'detected'), True),

    # Common bridge command mishearings
    (r'\ball head\b', 'all ahead', True),
    (r'\bfull head\b', 'full ahead', True),
    (r'\bhalf head\b', 'half ahead', True),
    (r'\bcome about face\b', 'come about', True),

    # Navigation terms
    (r'\bbaring\b', 'bearing', True),
    (r'\bbare ring\b', 'bearing', True),

    # Status report phrases
    (r'\bhull integrity\s+is\s+holding\b', 'hull integrity is holding', True),
    (r'\bshe\'?s\s+holding\b', "she's holding", True),
]


class DomainPostCorrector:
    """
    Layer 3: Domain-aware post-correction for known transcription errors.

    Applies regex-based phrase corrections to transcript segments,
    fixing multi-word errors that Whisper consistently gets wrong
    in the Starship Horizons context.

    Attributes:
        corrections: List of (pattern, replacement, case_insensitive) tuples
    """

    def __init__(
        self,
        extra_corrections: Optional[List[Tuple[str, Union[str, Callable], bool]]] = None
    ):
        """
        Initialize post-corrector.

        Args:
            extra_corrections: Additional corrections to append to defaults
        """
        self.corrections = list(PHRASE_CORRECTIONS)
        if extra_corrections:
            self.corrections.extend(extra_corrections)

        # Pre-compile patterns for performance
        self._compiled: List[Tuple[re.Pattern, Union[str, Callable], bool]] = []
        for pattern, replacement, case_insensitive in self.corrections:
            flags = re.IGNORECASE if case_insensitive else 0
            try:
                compiled = re.compile(pattern, flags)
                self._compiled.append((compiled, replacement, case_insensitive))
            except re.error as e:
                logger.warning(f"Invalid correction pattern '{pattern}': {e}")

    def correct_segments(
        self,
        segments: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Apply phrase corrections to all segments.

        Args:
            segments: List of transcription segments with 'text' field

        Returns:
            Tuple of (corrected_segments, stats_dict) where stats contains:
                - corrections_count: total corrections made
                - corrected_segments: number of segments that were modified
                - corrections_log: list of (original, corrected, pattern) entries
        """
        if not segments:
            return segments, {
                'corrections_count': 0,
                'corrected_segments': 0,
                'corrections_log': []
            }

        corrections_count = 0
        corrected_segments = 0
        corrections_log: List[Dict[str, str]] = []

        for seg in segments:
            text = seg.get('text', '')
            if not text:
                continue

            original_text = text
            segment_corrected = False

            for compiled_pattern, replacement, _ in self._compiled:
                if compiled_pattern.search(text):
                    new_text = compiled_pattern.sub(replacement, text)
                    if new_text != text:
                        corrections_log.append({
                            'original': text,
                            'corrected': new_text,
                            'pattern': compiled_pattern.pattern
                        })
                        text = new_text
                        corrections_count += 1
                        segment_corrected = True

            if segment_corrected:
                seg['text'] = text
                corrected_segments += 1
                logger.debug(
                    f"Post-correction: '{original_text}' -> '{text}'"
                )

        stats = {
            'corrections_count': corrections_count,
            'corrected_segments': corrected_segments,
            'corrections_log': corrections_log
        }

        if corrections_count > 0:
            logger.info(
                f"Domain post-correction applied {corrections_count} fixes "
                f"across {corrected_segments} segments"
            )

        return segments, stats


# ============================================================================
# Layer 5: LLM-powered semantic transcript cleanup
# ============================================================================

# Default prompt template for transcript correction
TRANSCRIPT_CORRECTION_PROMPT = """You are a transcript correction assistant for a Starship Horizons bridge simulator session.
Fix any transcription errors. Common issues: phonetic substitutions, wrong homophones,
nonsensical phrases that should be bridge/naval commands.

Rules:
- Only fix clear errors. Do not rephrase or rewrite correct text.
- Preserve the speaker's intent and meaning.
- Common terms: shields, warp, helm, tactical, bearing, aye, captain, sir.
- Only output corrections. Format: LINE_NUMBER|corrected text
- If a line is correct, skip it (do not output it).

{lines}"""


class TranscriptLLMCleaner:
    """
    Layer 5: LLM-powered semantic transcript cleanup.

    Sends batches of transcript segments to Ollama for contextual
    error correction. Catches errors that dictionary-based corrections
    miss, especially novel phonetic substitutions and context-dependent
    fixes.

    Attributes:
        host: Ollama server URL
        model: Model name to use
        timeout: Request timeout in seconds
        enabled: Whether LLM cleanup is enabled
    """

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None
    ):
        """
        Initialize LLM transcript cleaner.

        Args:
            host: Ollama server URL (default from OLLAMA_HOST env var)
            model: Model name (default from OLLAMA_MODEL env var)
            timeout: Request timeout in seconds (default from OLLAMA_TIMEOUT env var)
        """
        self.enabled = os.getenv(
            'LLM_TRANSCRIPT_CLEANUP', 'true'
        ).lower() == 'true'

        # Build LLMClient — only pass overrides when explicitly provided
        llm_kwargs: Dict[str, Any] = {}
        if host:
            llm_kwargs['base_url'] = f"{host.rstrip('/')}/v1"
        if model:
            llm_kwargs['model'] = model
        if timeout:
            llm_kwargs['timeout'] = timeout

        self._llm = LLMClient(**llm_kwargs) if llm_kwargs else LLMClient()
        self.host = self._llm.base_url.replace('/v1', '')
        self.model = self._llm.model
        self.timeout = self._llm.timeout

    def clean_segments(
        self,
        segments: List[Dict[str, Any]],
        batch_size: int = 15
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Clean transcript segments using LLM.

        Batches segments and sends them to Ollama for contextual correction.
        Gracefully degrades if Ollama is unavailable.

        Args:
            segments: List of transcription segments with 'text' field
            batch_size: Number of segments per LLM batch

        Returns:
            Tuple of (cleaned_segments, stats_dict) where stats contains:
                - batches_sent: number of batches processed
                - corrections_made: total corrections applied
                - time_seconds: total processing time
                - skipped_reason: reason if cleanup was skipped (optional)
        """
        empty_stats: Dict[str, Any] = {
            'batches_sent': 0,
            'corrections_made': 0,
            'time_seconds': 0.0
        }

        if not segments:
            return segments, empty_stats

        if not self.enabled:
            empty_stats['skipped_reason'] = 'disabled by LLM_TRANSCRIPT_CLEANUP env var'
            logger.info("LLM transcript cleanup disabled by environment variable")
            return segments, empty_stats

        # Check LLM backend availability
        if not self._llm.check_available():
            empty_stats['skipped_reason'] = 'LLM backend not available'
            logger.warning("LLM transcript cleanup skipped: backend not available")
            return segments, empty_stats

        start_time = time.time()
        batches_sent = 0
        corrections_made = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0

        # Build all batches upfront
        batch_items: List[Tuple[int, List[Dict[str, Any]]]] = []
        for batch_start in range(0, len(segments), batch_size):
            batch_end = min(batch_start + batch_size, len(segments))
            batch_items.append((batch_start, segments[batch_start:batch_end]))

        max_workers = int(os.getenv('LLM_CLEANUP_WORKERS', '3'))

        def _process_batch(
            item: Tuple[int, List[Dict[str, Any]]]
        ) -> Tuple[int, Optional[str], Dict[str, Any]]:
            """Process a single batch: build prompt, call Ollama, return result."""
            b_start, batch = item
            prompt = self._build_batch_prompt(batch)
            metrics: Dict[str, Any] = {}
            text = self._call_ollama(prompt, metrics_out=metrics)
            return b_start, text, metrics

        # Process batches in parallel (I/O-bound HTTP calls)
        results_list: List[Tuple[int, Optional[str], Dict[str, Any]]] = []
        with ThreadPoolExecutor(
            max_workers=min(max_workers, len(batch_items)),
            thread_name_prefix="llm-cleanup"
        ) as executor:
            futures = {
                executor.submit(_process_batch, item): item
                for item in batch_items
            }
            for future in as_completed(futures):
                try:
                    results_list.append(future.result())
                except Exception as e:
                    logger.warning(f"LLM cleanup batch failed: {e}")

        # Apply corrections in segment order
        for batch_start, response_text, batch_metrics in sorted(
            results_list, key=lambda x: x[0]
        ):
            if response_text:
                batches_sent += 1
                corrections = self._parse_corrections(response_text)

                total_prompt_tokens += batch_metrics.get('prompt_eval_count', 0)
                total_completion_tokens += batch_metrics.get('eval_count', 0)

                for line_num, corrected_text in corrections.items():
                    seg_idx = batch_start + line_num - 1  # 1-indexed to 0-indexed
                    if 0 <= seg_idx < len(segments):
                        original = segments[seg_idx].get('text', '')
                        if corrected_text.strip() and corrected_text.strip() != original.strip():
                            logger.debug(
                                f"LLM correction [{seg_idx}]: "
                                f"'{original}' -> '{corrected_text}'"
                            )
                            segments[seg_idx]['text'] = corrected_text.strip()
                            corrections_made += 1

        elapsed = time.time() - start_time
        stats: Dict[str, Any] = {
            'batches_sent': batches_sent,
            'corrections_made': corrections_made,
            'time_seconds': round(elapsed, 2),
            'total_prompt_tokens': total_prompt_tokens,
            'total_completion_tokens': total_completion_tokens,
            'total_tokens': total_prompt_tokens + total_completion_tokens,
        }

        if corrections_made > 0:
            logger.info(
                f"LLM transcript cleanup: {corrections_made} corrections "
                f"in {batches_sent} batches ({elapsed:.1f}s, "
                f"{min(max_workers, len(batch_items))} workers)"
            )

        return segments, stats

    def _build_batch_prompt(
        self,
        batch: List[Dict[str, Any]]
    ) -> str:
        """
        Build prompt for a batch of segments.

        Args:
            batch: List of segments to include in prompt

        Returns:
            Formatted prompt string
        """
        lines = []
        for i, seg in enumerate(batch, start=1):
            text = seg.get('text', '').strip()
            if text:
                lines.append(f"{i}: {text}")

        lines_text = '\n'.join(lines)
        return TRANSCRIPT_CORRECTION_PROMPT.format(lines=lines_text)

    def _parse_corrections(
        self,
        response_text: str
    ) -> Dict[int, str]:
        """
        Parse LLM response into line number -> corrected text mapping.

        Expected format per line: "LINE_NUMBER|corrected text"

        Args:
            response_text: Raw LLM output

        Returns:
            Dict mapping 1-indexed line numbers to corrected text
        """
        corrections: Dict[int, str] = {}

        for line in response_text.strip().split('\n'):
            line = line.strip()
            if not line or '|' not in line:
                continue

            parts = line.split('|', 1)
            if len(parts) != 2:
                continue

            try:
                # Handle formats like "1|text", "1.|text", "1:|text"
                num_str = parts[0].strip().rstrip('.').rstrip(':')
                line_num = int(num_str)
                corrected = parts[1].strip()
                # LLM sometimes outputs "original text|corrected text" after the
                # line number. Take only the last pipe-separated segment.
                if '|' in corrected:
                    corrected = corrected.split('|')[-1].strip()
                if corrected and line_num > 0:
                    corrections[line_num] = corrected
            except ValueError:
                continue

        return corrections

    def _call_ollama(
        self,
        prompt: str,
        metrics_out: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Send prompt to LLM backend and return response.

        Args:
            prompt: The prompt to send
            metrics_out: Optional dict to populate with response metrics

        Returns:
            Generated text or None if request failed
        """
        result = self._llm.generate(
            prompt=prompt,
            temperature=0.1,
            max_tokens=500,
            top_k=30,
        )

        if result is None:
            return None

        if metrics_out is not None:
            metrics_out['model'] = result.model
            metrics_out['prompt_eval_count'] = result.prompt_tokens
            metrics_out['eval_count'] = result.completion_tokens
            metrics_out['total_duration'] = 0
            metrics_out['load_duration'] = 0
            metrics_out['prompt_eval_duration'] = 0
            metrics_out['eval_duration'] = 0

        return result.text
