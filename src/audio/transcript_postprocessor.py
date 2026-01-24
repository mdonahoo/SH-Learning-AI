"""
Transcript post-processing for improved quality.

This module provides text cleanup, segment merging, and word-level
timestamp optimization for Whisper transcription output.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TranscriptPostProcessor:
    """
    Post-processes transcription segments for improved quality.

    Features:
    1. Text cleanup (extra spaces, hyphenation artifacts)
    2. Merge adjacent short segments from same speaker
    3. Remove empty segments
    4. Word-level timestamp optimization for better boundaries
    """

    # Minimum words for a segment to stand alone
    MIN_STANDALONE_WORDS = 3

    # Maximum gap (seconds) to merge segments
    MAX_MERGE_GAP = 2.0

    # Patterns to fix
    HYPHEN_ARTIFACT_PATTERN = re.compile(r'\s*-\s*')
    MULTI_SPACE_PATTERN = re.compile(r'\s{2,}')

    def __init__(
        self,
        min_standalone_words: int = 3,
        max_merge_gap: float = 2.0
    ):
        """
        Initialize post-processor.

        Args:
            min_standalone_words: Minimum words for segment to stand alone
            max_merge_gap: Maximum time gap (seconds) to merge segments
        """
        self.min_standalone_words = min_standalone_words
        self.max_merge_gap = max_merge_gap

    def process(
        self,
        segments: List[Dict[str, Any]],
        speaker_aware: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Apply all post-processing steps to segments.

        Args:
            segments: List of transcription segments
            speaker_aware: Only merge segments from same speaker

        Returns:
            Processed segments
        """
        if not segments:
            return segments

        original_count = len(segments)

        # Step 1: Clean text in all segments
        segments = self._clean_all_text(segments)

        # Step 2: Remove empty segments
        segments = self._remove_empty_segments(segments)

        # Step 3: Merge short adjacent segments
        segments = self._merge_short_segments(segments, speaker_aware)

        # Step 4: Word-level boundary optimization (if word data available)
        segments = self._optimize_boundaries(segments)

        # Step 5: Final cleanup pass
        segments = self._remove_empty_segments(segments)

        final_count = len(segments)
        if final_count != original_count:
            logger.info(
                f"Transcript post-processing: {original_count} -> {final_count} segments "
                f"({original_count - final_count} merged/removed)"
            )

        return segments

    def _clean_all_text(
        self,
        segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Clean text in all segments."""
        for seg in segments:
            if 'text' in seg:
                seg['text'] = self._clean_text(seg['text'])
            # Also clean words if present
            if 'words' in seg:
                for word in seg['words']:
                    if 'word' in word:
                        word['word'] = self._clean_text(word['word'])
        return segments

    def _clean_text(self, text: str) -> str:
        """
        Clean a single text string.

        Fixes:
        - Multiple spaces -> single space
        - Hyphenation artifacts (e.g., "deploy -the -shuttle")
        - Leading/trailing whitespace
        """
        if not text:
            return text

        # Fix hyphenation artifacts: "deploy -the -shuttle" -> "deploy the shuttle"
        text = self.HYPHEN_ARTIFACT_PATTERN.sub(' ', text)

        # Collapse multiple spaces
        text = self.MULTI_SPACE_PATTERN.sub(' ', text)

        # Strip whitespace
        text = text.strip()

        return text

    def _remove_empty_segments(
        self,
        segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove segments with empty or whitespace-only text."""
        return [
            seg for seg in segments
            if seg.get('text', '').strip()
        ]

    def _merge_short_segments(
        self,
        segments: List[Dict[str, Any]],
        speaker_aware: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Merge short adjacent segments.

        A segment is considered "short" if it has fewer than
        min_standalone_words words.
        """
        if len(segments) < 2:
            return segments

        # Sort by start time
        segments = sorted(segments, key=lambda s: s.get('start', s.get('start_time', 0)))

        merged = []
        i = 0

        while i < len(segments):
            current = segments[i].copy()
            current_text = current.get('text', '').strip()
            current_words = len(current_text.split()) if current_text else 0
            current_speaker = current.get('speaker_id', current.get('speaker'))

            # Look ahead and merge if current is short
            while (current_words < self.min_standalone_words and
                   i + 1 < len(segments)):

                next_seg = segments[i + 1]
                next_text = next_seg.get('text', '').strip()
                next_speaker = next_seg.get('speaker_id', next_seg.get('speaker'))

                # Check speaker constraint
                if speaker_aware and current_speaker != next_speaker:
                    break

                # Check time gap
                current_end = current.get('end', current.get('end_time', 0))
                next_start = next_seg.get('start', next_seg.get('start_time', 0))
                gap = next_start - current_end

                if gap > self.max_merge_gap:
                    break

                # Merge the segments
                current['text'] = f"{current_text} {next_text}".strip()
                current['end'] = next_seg.get('end', next_seg.get('end_time', current_end))
                if 'end_time' in current:
                    current['end_time'] = current['end']

                # Merge words if present
                if 'words' in current and 'words' in next_seg:
                    current['words'] = current['words'] + next_seg['words']

                # Average confidence
                if 'confidence' in current and 'confidence' in next_seg:
                    current['confidence'] = (
                        current['confidence'] + next_seg['confidence']
                    ) / 2

                # Update loop variables
                current_text = current['text']
                current_words = len(current_text.split()) if current_text else 0
                i += 1

                logger.debug(f"Merged short segment: '{next_text}' into previous")

            merged.append(current)
            i += 1

        return merged

    def _optimize_boundaries(
        self,
        segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Optimize segment boundaries using word-level timestamps.

        This creates better segment breaks at natural sentence boundaries
        rather than arbitrary time splits.
        """
        # Check if we have word-level data
        has_words = any('words' in seg for seg in segments)
        if not has_words:
            return segments

        optimized = []

        for seg in segments:
            words = seg.get('words', [])
            if not words:
                optimized.append(seg)
                continue

            # Try to split at sentence boundaries if segment is long
            text = seg.get('text', '')
            if len(text.split()) > 15:  # Only split long segments
                sub_segments = self._split_at_sentence_boundaries(seg, words)
                optimized.extend(sub_segments)
            else:
                optimized.append(seg)

        return optimized

    def _split_at_sentence_boundaries(
        self,
        segment: Dict[str, Any],
        words: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Split a long segment at sentence boundaries.

        Uses word-level timestamps to create accurate boundaries.
        """
        if not words:
            return [segment]

        # Find sentence-ending punctuation positions
        sentence_ends = []
        for i, word in enumerate(words):
            word_text = word.get('word', '')
            if word_text.rstrip().endswith(('.', '?', '!')):
                sentence_ends.append(i)

        if not sentence_ends:
            return [segment]

        # Create sub-segments at sentence boundaries
        sub_segments = []
        prev_end = 0

        for end_idx in sentence_ends:
            if end_idx == prev_end:
                continue

            # Get words for this sub-segment
            sub_words = words[prev_end:end_idx + 1]
            if not sub_words:
                continue

            # Build sub-segment
            sub_text = ' '.join(w.get('word', '') for w in sub_words)
            sub_text = self._clean_text(sub_text)

            if not sub_text:
                continue

            sub_seg = {
                'start': sub_words[0].get('start', segment.get('start', 0)),
                'end': sub_words[-1].get('end', segment.get('end', 0)),
                'text': sub_text,
                'words': sub_words,
                'speaker_id': segment.get('speaker_id'),
                'speaker': segment.get('speaker'),
                'confidence': segment.get('confidence', 0.0)
            }

            # Copy timing fields
            if 'start_time' in segment:
                sub_seg['start_time'] = sub_seg['start']
                sub_seg['end_time'] = sub_seg['end']

            sub_segments.append(sub_seg)
            prev_end = end_idx + 1

        # Add remaining words as final segment
        if prev_end < len(words):
            remaining_words = words[prev_end:]
            remaining_text = ' '.join(w.get('word', '') for w in remaining_words)
            remaining_text = self._clean_text(remaining_text)

            if remaining_text:
                final_seg = {
                    'start': remaining_words[0].get('start', segment.get('start', 0)),
                    'end': remaining_words[-1].get('end', segment.get('end', 0)),
                    'text': remaining_text,
                    'words': remaining_words,
                    'speaker_id': segment.get('speaker_id'),
                    'speaker': segment.get('speaker'),
                    'confidence': segment.get('confidence', 0.0)
                }
                if 'start_time' in segment:
                    final_seg['start_time'] = final_seg['start']
                    final_seg['end_time'] = final_seg['end']
                sub_segments.append(final_seg)

        return sub_segments if sub_segments else [segment]


def merge_adjacent_fragments(
    segments: List[Dict[str, Any]],
    max_gap: float = 1.5
) -> List[Dict[str, Any]]:
    """
    Merge adjacent segments that appear to be fragments of the same utterance.

    Specifically targets patterns like:
    - "stand" + "by" -> "standby"
    - "deploy" + "the shuttle" -> "deploy the shuttle"

    Args:
        segments: List of segments with speaker_id, text, start/end times
        max_gap: Maximum gap in seconds to consider for merging

    Returns:
        Merged segments
    """
    if len(segments) < 2:
        return segments

    # Sort by start time
    segments = sorted(segments, key=lambda s: s.get('start', s.get('start_time', 0)))

    merged = []
    i = 0

    while i < len(segments):
        current = segments[i].copy()
        current_text = current.get('text', '').strip()

        # Check if current ends without punctuation (fragment indicator)
        ends_without_punct = (
            current_text and
            not current_text[-1] in '.?!,;:'
        )

        # Look for continuation
        if ends_without_punct and i + 1 < len(segments):
            next_seg = segments[i + 1]
            next_text = next_seg.get('text', '').strip()

            # Same speaker check
            same_speaker = (
                current.get('speaker_id') == next_seg.get('speaker_id') or
                current.get('speaker') == next_seg.get('speaker')
            )

            # Time gap check
            current_end = current.get('end', current.get('end_time', 0))
            next_start = next_seg.get('start', next_seg.get('start_time', 0))
            gap = next_start - current_end

            # Check if next starts with lowercase (continuation indicator)
            starts_lowercase = (
                next_text and
                next_text[0].islower()
            )

            # Merge if looks like continuation
            if same_speaker and gap <= max_gap and (starts_lowercase or len(current_text.split()) <= 2):
                current['text'] = f"{current_text} {next_text}".strip()
                current['end'] = next_seg.get('end', next_seg.get('end_time', current_end))
                if 'end_time' in current:
                    current['end_time'] = current['end']
                if 'words' in current and 'words' in next_seg:
                    current['words'] = current['words'] + next_seg['words']
                i += 1  # Skip next segment since we merged it

        merged.append(current)
        i += 1

    return merged
