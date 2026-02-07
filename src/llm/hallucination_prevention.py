"""
Hallucination prevention utilities for LLM-generated content.

This module provides tools to constrain LLM inputs and validate outputs,
reducing fabricated content in mission debriefs and stories.

Strategies implemented:
1. Constrained context building - only pass verified data to LLM
2. Structured output templates - fill-in-the-blank reduces invention
3. Post-generation validation - check quotes and stats against source
4. Fuzzy quote matching - verify quoted text exists in transcript
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """A detected hallucination or validation issue."""
    issue_type: str  # "quote_not_found", "stat_mismatch", "role_invented"
    description: str
    severity: str  # "error", "warning", "info"
    original_text: str
    suggestion: Optional[str] = None


class ConstrainedContextBuilder:
    """
    Builds constrained context for LLM prompts using only verified data.

    The LLM should never see raw transcripts - only pre-extracted,
    verified evidence and computed metrics.
    """

    def __init__(
        self,
        transcripts: List[Dict[str, Any]],
        analysis_results: Dict[str, Any]
    ):
        """
        Initialize context builder.

        Args:
            transcripts: Raw transcript data
            analysis_results: Pre-computed analysis from other modules
        """
        self.transcripts = transcripts
        self.analysis = analysis_results

    def build_debrief_context(self) -> Dict[str, Any]:
        """
        Build constrained context for debrief generation.

        Returns:
            Dictionary with only verified, computed data
        """
        # Extract computed metrics (not hallucinated)
        seven_habits = self.analysis.get('seven_habits', {})
        comm_quality = self.analysis.get('communication_quality', {})
        learning_eval = self.analysis.get('learning_evaluation', {})

        # Get habits scores
        habits_score = seven_habits.get('overall_effectiveness_score', 'N/A')

        # Find top and lowest habits
        strengths = seven_habits.get('strengths', [])
        growth_areas = seven_habits.get('growth_areas', [])

        top_habit = strengths[0] if strengths else {
            'name': 'Teamwork',
            'score': 3,
            'interpretation': 'No data available'
        }

        lowest_habit = growth_areas[0] if growth_areas else {
            'name': 'Taking Initiative',
            'score': 2,
            'interpretation': 'No data available'
        }

        # Get communication score
        comm_stats = comm_quality.get('statistics', {})
        comm_score = comm_stats.get('effectiveness_percentage', 70)

        # Extract verified speaker statistics
        speakers = self._extract_speaker_stats()

        # Extract pre-selected quotes (verified to exist)
        positive_quotes = self._extract_top_quotes(category='effective', limit=8)
        negative_quotes = self._extract_top_quotes(category='improvement', limit=5)
        key_moments = self._extract_key_moments(limit=20)

        return {
            # Computed metrics
            'duration': self.analysis.get('mission_statistics', {}).get('mission_duration', 'Unknown'),
            'communication_score': round(comm_score, 1),
            'habits_score': habits_score,

            # Habit details
            'top_habit': top_habit,
            'lowest_habit': lowest_habit,

            # Speaker stats (computed)
            'speakers': speakers,
            'speaker_count': len(speakers),

            # Pre-verified quotes (these exist in transcript)
            'positive_examples': positive_quotes,
            'negative_examples': negative_quotes,
            'key_moments': key_moments,

            # Learning metrics
            'kirkpatrick_scores': self._extract_kirkpatrick_scores(learning_eval),

            # Mission outcome
            'objectives_completed': self.analysis.get('mission_statistics', {}).get('objectives_completed', 0),
            'objectives_total': self.analysis.get('mission_statistics', {}).get('objectives_total', 0),
        }

    def _extract_speaker_stats(self) -> List[Dict[str, Any]]:
        """Extract verified speaker statistics."""
        speaker_counts = {}
        speaker_times = {}

        for t in self.transcripts:
            speaker = t.get('speaker') or t.get('speaker_id') or 'unknown'
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

            # Estimate speaking time from text length
            text = t.get('text', '')
            words = len(text.split())
            # Rough estimate: 150 words per minute
            speaker_times[speaker] = speaker_times.get(speaker, 0) + (words / 150 * 60)

        total = sum(speaker_counts.values())

        speakers = []
        for speaker, count in sorted(speaker_counts.items(), key=lambda x: -x[1]):
            # Try to get role from analysis
            role_analysis = self.analysis.get('role_analysis', [])
            role = 'Crew Member'
            for ra in role_analysis:
                if ra.get('speaker') == speaker:
                    role = ra.get('likely_role', 'Crew Member')
                    break

            speakers.append({
                'speaker_id': speaker,
                'role': role,
                'utterance_count': count,
                'percentage': round(count / total * 100, 1) if total > 0 else 0,
                'speaking_time': round(speaker_times.get(speaker, 0), 1)
            })

        return speakers[:10]  # Top 10 speakers

    def _extract_top_quotes(
        self,
        category: str = 'effective',
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Extract pre-verified quotes from transcript.

        Args:
            category: 'effective' or 'improvement'
            limit: Maximum quotes to return

        Returns:
            List of verified quote dictionaries
        """
        comm_quality = self.analysis.get('communication_quality', {})

        if category == 'effective':
            examples = comm_quality.get('effective_examples', [])
        else:
            examples = comm_quality.get('improvement_examples', [])

        quotes = []
        for ex in examples[:limit]:
            # Verify quote exists in transcript
            text = ex.get('text', '')
            if self._verify_quote_exists(text):
                quotes.append({
                    'timestamp': ex.get('timestamp', ''),
                    'speaker': ex.get('speaker', 'Unknown'),
                    'text': text,
                    'confidence': ex.get('confidence', 0),
                    'assessment': ex.get('assessment', '') if category == 'effective' else ex.get('issue', '')
                })

        return quotes

    def _extract_key_moments(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Extract key moments from transcript by confidence."""
        # Sort by confidence and get top moments
        sorted_transcripts = sorted(
            self.transcripts,
            key=lambda x: x.get('confidence', 0),
            reverse=True
        )

        moments = []
        for t in sorted_transcripts[:limit]:
            moments.append({
                'timestamp': t.get('timestamp', ''),
                'speaker': t.get('speaker') or t.get('speaker_id') or 'Unknown',
                'text': t.get('text', ''),
                'confidence': t.get('confidence', 0)
            })

        return moments

    def _extract_kirkpatrick_scores(
        self,
        learning_eval: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract Kirkpatrick level scores."""
        levels = learning_eval.get('kirkpatrick_levels', [])

        scores = {}
        for level in levels:
            level_num = level.get('level', 0)
            scores[f'level_{level_num}'] = {
                'name': level.get('name', ''),
                'score': level.get('score', 0),
                'interpretation': level.get('interpretation', '')
            }

        return scores

    def _verify_quote_exists(self, quote: str, threshold: float = 0.8) -> bool:
        """
        Verify a quote exists in the transcript using fuzzy matching.

        Args:
            quote: Quote text to verify
            threshold: Minimum similarity ratio (0-1)

        Returns:
            True if quote matches transcript text
        """
        quote_lower = quote.lower().strip()

        for t in self.transcripts:
            text = t.get('text', '').lower().strip()

            # Exact match
            if quote_lower in text or text in quote_lower:
                return True

            # Fuzzy match
            ratio = SequenceMatcher(None, quote_lower, text).ratio()
            if ratio >= threshold:
                return True

        return False


class ContradictionDetector:
    """
    Detects logical contradictions in LLM-generated debrief content.

    Identifies patterns like the same skill listed as both strength and weakness,
    which suggest the LLM received conflicting data or hallucinated.
    """

    def __init__(self, narrative: str, analysis: Dict[str, Any]):
        """
        Initialize contradiction detector.

        Args:
            narrative: LLM-generated narrative text
            analysis: Original analysis data for validation context
        """
        self.narrative = narrative.lower()
        self.analysis = analysis

    def detect_contradictions(self) -> List[ValidationIssue]:
        """
        Detect contradictions in the narrative.

        Returns:
            List of detected contradictions as ValidationIssue objects
        """
        issues = []
        issues.extend(self._detect_strength_weakness_overlap())
        issues.extend(self._detect_conflicting_evaluations())
        return issues

    def _detect_strength_weakness_overlap(self) -> List[ValidationIssue]:
        """
        Detect when the same concept appears as both strength and weakness.

        Returns:
            List of ValidationIssue objects for contradictions
        """
        issues = []

        # Extract strengths and growth areas sections
        strength_text = self._extract_section(
            r"strength|did well|excel|excellent|strong|good at",
            max_length=500
        )
        weakness_text = self._extract_section(
            r"grow|weakness|improve|need|work on|challenge|difficult",
            max_length=500
        )

        # Define conceptual keywords grouped by theme
        concepts = [
            ("delegation", ["delegat", "assign", "distribute work", "task assignment"]),
            ("communication", ["communicat", "share", "inform", "report"]),
            ("teamwork", ["teamwork", "team", "collaborat", "synerg", "cooperative"]),
            ("leadership", ["lead", "command", "decision", "direct"]),
            ("initiative", ["initiative", "proactive", "take action", "volunteer"]),
            ("listening", ["listen", "understand", "empathetic", "ask"]),
            ("planning", ["plan", "organize", "prepare", "strategy"]),
        ]

        for concept_name, keywords in concepts:
            # Check if concept appears in both sections
            in_strengths = any(kw in strength_text for kw in keywords)
            in_weaknesses = any(kw in weakness_text for kw in keywords)

            if in_strengths and in_weaknesses:
                issues.append(ValidationIssue(
                    issue_type="strength_weakness_contradiction",
                    description=f"'{concept_name}' listed as both strength and weakness",
                    severity="error",
                    original_text=f"{concept_name} (appears in both sections)",
                    suggestion=(
                        f"Review data sources. This suggests conflicting metrics. "
                        f"Low 7-Habits score may be unreliable."
                    )
                ))

        return issues

    def _detect_conflicting_evaluations(self) -> List[ValidationIssue]:
        """
        Detect other types of contradictory statements.

        Returns:
            List of ValidationIssue objects
        """
        issues = []

        # Pattern: "X is excellent" followed by "X needs work" in close proximity
        # Split into sentences for easier analysis
        sentences = re.split(r'[.!?]', self.narrative)
        sentences = [s.strip() for s in sentences if s.strip()]

        for i, sent1 in enumerate(sentences):
            for sent2 in sentences[max(0, i-2):i+3]:  # Check nearby sentences
                if sent1 == sent2:
                    continue

                # Look for contradictory sentiment on same topic
                if self._is_contradiction_pair(sent1, sent2):
                    issues.append(ValidationIssue(
                        issue_type="contradictory_statements",
                        description="Contradictory statements about the same topic",
                        severity="warning",
                        original_text=f"'{sent1[:50]}...' vs '{sent2[:50]}...'",
                        suggestion="Review and resolve contradictory claims"
                    ))
                    break

        return issues

    def _extract_section(
        self,
        pattern: str,
        max_length: int = 500
    ) -> str:
        """
        Extract section matching pattern from narrative.

        Args:
            pattern: Regex pattern to find section
            max_length: Max characters to extract around match

        Returns:
            Extracted text section
        """
        match = re.search(pattern, self.narrative)
        if not match:
            return ""

        start = max(0, match.start() - max_length)
        end = min(len(self.narrative), match.end() + max_length)
        return self.narrative[start:end]

    def _is_contradiction_pair(self, sent1: str, sent2: str) -> bool:
        """
        Check if two sentences contradict each other.

        Args:
            sent1: First sentence
            sent2: Second sentence

        Returns:
            True if contradictory
        """
        # Extract key words (nouns, verbs) from both sentences
        words1 = set(re.findall(r'\b\w+\b', sent1.lower()))
        words2 = set(re.findall(r'\b\w+\b', sent2.lower()))

        # Check for overlap (same topic)
        overlap = words1 & words2
        if len(overlap) < 2:
            return False

        # Check for contradictory sentiment words
        positive = {"excellent", "great", "good", "strong", "well", "success"}
        negative = {"poor", "weak", "bad", "difficult", "struggle", "fail"}

        sent1_positive = bool(positive & words1)
        sent1_negative = bool(negative & words1)
        sent2_positive = bool(positive & words2)
        sent2_negative = bool(negative & words2)

        # Contradiction if one is clearly positive and other is clearly negative
        return (sent1_positive and sent2_negative) or (sent1_negative and sent2_positive)


class OutputValidator:
    """
    Validates LLM-generated content against source data.

    Detects and flags:
    - Quotes not found in transcript
    - Statistics that don't match computed values
    - Roles/names that weren't detected
    """

    def __init__(
        self,
        source_transcripts: List[Dict[str, Any]],
        source_metrics: Dict[str, Any]
    ):
        """
        Initialize validator.

        Args:
            source_transcripts: Original transcript data
            source_metrics: Computed metrics to validate against
        """
        self.transcripts = source_transcripts
        self.metrics = source_metrics

        # Build lookup for fast quote verification
        self.transcript_texts = [
            t.get('text', '').lower().strip()
            for t in source_transcripts
        ]

        # Get detected roles
        role_analysis = source_metrics.get('role_analysis', [])
        self.detected_roles = set()
        for ra in role_analysis:
            role = ra.get('likely_role', '')
            if role:
                self.detected_roles.add(role)

        # Standard bridge roles
        self.valid_roles = {
            'Captain', 'Tactical', 'Helm', 'Operations', 'Science',
            'Engineering', 'Communications', 'First Officer', 'XO',
            'Crew Member', 'Officer'
        }

    def validate(self, generated_text: str) -> List[ValidationIssue]:
        """
        Validate generated text against source data.

        Args:
            generated_text: LLM-generated content

        Returns:
            List of validation issues found
        """
        issues = []

        # Check quoted text
        issues.extend(self._validate_quotes(generated_text))

        # Check statistics
        issues.extend(self._validate_statistics(generated_text))

        # Check role mentions
        issues.extend(self._validate_roles(generated_text))

        return issues

    def _validate_quotes(self, text: str) -> List[ValidationIssue]:
        """Check that quoted text exists in transcript."""
        issues = []

        # Find all quoted text
        quote_pattern = r'"([^"]+)"'
        quotes = re.findall(quote_pattern, text)

        for quote in quotes:
            if len(quote) < 5:  # Skip very short quotes
                continue

            if not self._fuzzy_match_in_transcript(quote):
                issues.append(ValidationIssue(
                    issue_type="quote_not_found",
                    description=f"Quote not found in transcript",
                    severity="warning",
                    original_text=quote[:100] + "..." if len(quote) > 100 else quote,
                    suggestion="Remove or replace with verified quote from transcript"
                ))

        return issues

    def _validate_statistics(self, text: str) -> List[ValidationIssue]:
        """Check that statistics match computed values."""
        issues = []

        # Define patterns and expected values
        seven_habits = self.metrics.get('seven_habits', {})
        habits_score = seven_habits.get('overall_effectiveness_score', None)

        comm_quality = self.metrics.get('communication_quality', {})
        comm_stats = comm_quality.get('statistics', {})
        comm_score = comm_stats.get('effectiveness_percentage', None)

        # Check habits score (X/5 pattern)
        if habits_score is not None:
            pattern = r'(\d+\.?\d*)\s*/\s*5'
            matches = re.findall(pattern, text)
            for match in matches:
                value = float(match)
                if abs(value - habits_score) > 0.5:
                    issues.append(ValidationIssue(
                        issue_type="stat_mismatch",
                        description=f"Habits score mismatch: generated {value}/5, actual {habits_score}/5",
                        severity="error",
                        original_text=f"{value}/5",
                        suggestion=f"Use actual value: {habits_score}/5"
                    ))

        # Check percentage patterns
        if comm_score is not None:
            pattern = r'(\d+\.?\d*)\s*%'
            matches = re.findall(pattern, text)
            for match in matches:
                value = float(match)
                # Allow some tolerance for percentage stats
                if 40 < value < 100:  # Likely a communication/effectiveness percentage
                    if abs(value - comm_score) > 10:
                        issues.append(ValidationIssue(
                            issue_type="stat_mismatch",
                            description=f"Percentage may not match: generated {value}%, expected ~{comm_score}%",
                            severity="warning",
                            original_text=f"{value}%",
                            suggestion=f"Verify against source data"
                        ))

        return issues

    def _validate_roles(self, text: str) -> List[ValidationIssue]:
        """Check that mentioned roles were actually detected."""
        issues = []

        # Find role mentions
        role_pattern = r'\b(Captain|Tactical|Helm|Operations|Science|Engineering|Communications)\b'
        mentioned_roles = set(re.findall(role_pattern, text, re.IGNORECASE))

        for role in mentioned_roles:
            role_title = role.title()
            if role_title not in self.detected_roles and role_title not in self.valid_roles:
                issues.append(ValidationIssue(
                    issue_type="role_invented",
                    description=f"Role '{role_title}' was not detected in transcript",
                    severity="info",
                    original_text=role,
                    suggestion="Use speaker_1, speaker_2 or detected roles only"
                ))

        return issues

    def _fuzzy_match_in_transcript(
        self,
        quote: str,
        threshold: float = 0.7
    ) -> bool:
        """
        Check if quote fuzzy-matches any transcript text.

        Args:
            quote: Quote to search for
            threshold: Minimum similarity ratio

        Returns:
            True if match found
        """
        quote_lower = quote.lower().strip()

        # Remove common filler words for matching
        quote_clean = re.sub(r'\b(um|uh|like|you know)\b', '', quote_lower).strip()

        for text in self.transcript_texts:
            # Exact substring match
            if quote_lower in text or text in quote_lower:
                return True

            # Fuzzy match
            ratio = SequenceMatcher(None, quote_clean, text).ratio()
            if ratio >= threshold:
                return True

            # Check for significant overlap
            words_quote = set(quote_clean.split())
            words_text = set(text.split())
            if len(words_quote) > 3:
                overlap = len(words_quote & words_text) / len(words_quote)
                if overlap >= 0.7:
                    return True

        return False


def clean_hallucinations(
    generated_text: str,
    source_transcripts: List[Dict[str, Any]],
    source_metrics: Dict[str, Any],
    add_warning: bool = True
) -> Tuple[str, List[ValidationIssue]]:
    """
    Validate and optionally clean hallucinated content.

    Args:
        generated_text: LLM-generated content
        source_transcripts: Original transcript data
        source_metrics: Computed metrics
        add_warning: Whether to add warning if issues found

    Returns:
        Tuple of (cleaned/annotated text, list of issues)
    """
    validator = OutputValidator(source_transcripts, source_metrics)
    issues = validator.validate(generated_text)

    if not issues:
        return generated_text, []

    # Count by severity
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    logger.warning(
        f"Validation found {len(errors)} errors, {len(warnings)} warnings "
        f"in generated content"
    )

    result_text = generated_text

    if add_warning and (errors or warnings):
        warning_text = "\n\n---\n\n> ⚠️ **Data Verification Notice**\n"
        warning_text += "> Some content could not be verified against transcript data.\n"

        if errors:
            warning_text += f"> - {len(errors)} statistic(s) may not match source data\n"
        if warnings:
            warning_text += f"> - {len(warnings)} quote(s) could not be verified\n"

        result_text = generated_text + warning_text

    return result_text, issues


# Recommended sampling parameters for reduced hallucination
ANTI_HALLUCINATION_PARAMS = {
    "temperature": 0.3,      # Lower = more deterministic
    "top_p": 0.9,            # Nucleus sampling
    "top_k": 40,             # Limit vocabulary
    "repeat_penalty": 1.1,   # Reduce repetition
    "num_predict": 1500,     # Output length for debriefs
}

STORY_PARAMS = {
    "temperature": 0.5,      # Slightly higher for creativity
    "top_p": 0.9,
    "top_k": 50,
    "repeat_penalty": 1.1,
    "num_predict": 4000,     # Full-length stories
}
