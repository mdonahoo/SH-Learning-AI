"""
Transcription confidence distribution analyzer.

Analyzes the distribution of transcription confidence scores to identify
audio quality issues and communication clarity problems.
"""

import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceRange:
    """Definition of a confidence range bucket."""
    name: str
    min_value: float
    max_value: float
    label: str
    interpretation: str


# Standard confidence ranges matching the example report
CONFIDENCE_RANGES = [
    ConfidenceRange(
        name="excellent",
        min_value=0.90,
        max_value=1.01,
        label="90% and above",
        interpretation="Clear speech, high reliability"
    ),
    ConfidenceRange(
        name="good",
        min_value=0.80,
        max_value=0.90,
        label="80% to 89%",
        interpretation="Good clarity, reliable transcription"
    ),
    ConfidenceRange(
        name="acceptable",
        min_value=0.70,
        max_value=0.80,
        label="70% to 79%",
        interpretation="Acceptable, some uncertainty"
    ),
    ConfidenceRange(
        name="marginal",
        min_value=0.60,
        max_value=0.70,
        label="60% to 69%",
        interpretation="Marginal quality, may have errors"
    ),
    ConfidenceRange(
        name="poor",
        min_value=0.0,
        max_value=0.60,
        label="Below 60%",
        interpretation="Poor quality, likely transcription errors"
    ),
]


class ConfidenceAnalyzer:
    """
    Analyzes transcription confidence distribution.

    Provides detailed breakdown of confidence scores across transcripts
    to identify audio quality issues and communication clarity problems.
    """

    def __init__(
        self,
        transcripts: List[Dict[str, Any]],
        ranges: List[ConfidenceRange] = None
    ):
        """
        Initialize the confidence analyzer.

        Args:
            transcripts: List of transcript dictionaries with 'confidence' field
            ranges: Optional custom confidence ranges
        """
        self.transcripts = transcripts
        self.ranges = ranges or CONFIDENCE_RANGES

    def analyze_distribution(self) -> Dict[str, Any]:
        """
        Analyze the distribution of confidence scores.

        Returns:
            Dictionary with distribution statistics
        """
        if not self.transcripts:
            return self._empty_results()

        # Initialize counters
        range_counts = {r.name: 0 for r in self.ranges}
        range_examples = {r.name: [] for r in self.ranges}
        speaker_confidence = defaultdict(list)

        total = len(self.transcripts)
        all_confidences = []

        for transcript in self.transcripts:
            confidence = transcript.get('confidence', 0)
            # Handle confidence as decimal (0-1) or percentage (0-100)
            if confidence > 1:
                confidence = confidence / 100

            all_confidences.append(confidence)
            speaker = transcript.get('speaker') or transcript.get('speaker_id') or 'unknown'
            speaker_confidence[speaker].append(confidence)

            # Categorize into range
            for r in self.ranges:
                if r.min_value <= confidence < r.max_value:
                    range_counts[r.name] += 1
                    if len(range_examples[r.name]) < 3:
                        range_examples[r.name].append({
                            'timestamp': transcript.get('timestamp', ''),
                            'speaker': speaker,
                            'text': transcript.get('text', ''),
                            'confidence': round(confidence, 3)
                        })
                    break

        # Calculate statistics
        avg_confidence = sum(all_confidences) / total if total > 0 else 0
        min_confidence = min(all_confidences) if all_confidences else 0
        max_confidence = max(all_confidences) if all_confidences else 0

        # Calculate standard deviation
        if total > 1:
            variance = sum((c - avg_confidence) ** 2 for c in all_confidences) / total
            std_dev = variance ** 0.5
        else:
            std_dev = 0

        # Calculate per-speaker averages
        speaker_stats = {}
        for speaker, confidences in speaker_confidence.items():
            speaker_avg = sum(confidences) / len(confidences) if confidences else 0
            speaker_stats[speaker] = {
                'average_confidence': round(speaker_avg, 3),
                'utterance_count': len(confidences),
                'min_confidence': round(min(confidences), 3) if confidences else 0,
                'max_confidence': round(max(confidences), 3) if confidences else 0,
            }

        # Build distribution results
        distribution = []
        for r in self.ranges:
            count = range_counts[r.name]
            percentage = (count / total * 100) if total > 0 else 0
            distribution.append({
                'range': r.label,
                'count': count,
                'percentage': round(percentage, 1),
                'interpretation': r.interpretation,
                'examples': range_examples[r.name]
            })

        # Generate training implications
        poor_percentage = range_counts['poor'] / total * 100 if total > 0 else 0
        marginal_percentage = range_counts['marginal'] / total * 100 if total > 0 else 0

        training_implications = self._generate_training_implications(
            poor_percentage, marginal_percentage, speaker_stats
        )

        return {
            'total_utterances': total,
            'average_confidence': round(avg_confidence, 3),
            'min_confidence': round(min_confidence, 3),
            'max_confidence': round(max_confidence, 3),
            'std_deviation': round(std_dev, 3),
            'distribution': distribution,
            'speaker_stats': speaker_stats,
            'training_implications': training_implications,
            'quality_assessment': self._assess_overall_quality(avg_confidence, poor_percentage)
        }

    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure."""
        return {
            'total_utterances': 0,
            'average_confidence': 0,
            'min_confidence': 0,
            'max_confidence': 0,
            'std_deviation': 0,
            'distribution': [],
            'speaker_stats': {},
            'training_implications': [],
            'quality_assessment': 'No data available'
        }

    def _generate_training_implications(
        self,
        poor_pct: float,
        marginal_pct: float,
        speaker_stats: Dict[str, Dict]
    ) -> List[str]:
        """Generate training implications based on confidence analysis."""
        implications = []

        if poor_pct > 25:
            implications.append(
                f"High rate of low-confidence utterances ({poor_pct:.1f}% below 60%) suggests "
                "environmental noise issues, overlapping speech, or unclear diction that should "
                "be addressed in crew training."
            )
        elif poor_pct > 15:
            implications.append(
                f"Moderate rate of low-confidence utterances ({poor_pct:.1f}% below 60%) "
                "indicates some communication clarity issues to address."
            )

        if marginal_pct > 30:
            implications.append(
                f"Significant portion of marginal quality ({marginal_pct:.1f}% in 60-69% range) "
                "suggests crew should practice clearer enunciation and avoid speaking over each other."
            )

        # Check for speakers with consistently low confidence
        low_confidence_speakers = [
            speaker for speaker, stats in speaker_stats.items()
            if stats['average_confidence'] < 0.65
        ]
        if low_confidence_speakers:
            implications.append(
                f"Speakers with below-average clarity: {', '.join(low_confidence_speakers)}. "
                "Individual coaching on voice projection and clear speech patterns recommended."
            )

        if not implications:
            implications.append(
                "Overall transcription confidence is acceptable. Continue monitoring for "
                "any degradation in communication clarity during high-stress scenarios."
            )

        return implications

    def _assess_overall_quality(self, avg_confidence: float, poor_pct: float) -> str:
        """Assess overall audio/transcription quality."""
        if avg_confidence >= 0.85 and poor_pct < 10:
            return "Excellent - High confidence across most communications"
        elif avg_confidence >= 0.75 and poor_pct < 20:
            return "Good - Generally clear communications with minor issues"
        elif avg_confidence >= 0.65 and poor_pct < 30:
            return "Acceptable - Some clarity issues to address"
        elif avg_confidence >= 0.55:
            return "Marginal - Significant clarity issues affecting transcription"
        else:
            return "Poor - Major audio quality or speech clarity problems"

    def generate_distribution_table(self) -> str:
        """
        Generate a markdown table of confidence distribution.

        Returns:
            Markdown formatted table string
        """
        results = self.analyze_distribution()

        lines = [
            "| Confidence Range | Utterance Count | Percentage |",
            "| --- | --- | --- |"
        ]

        for dist in results['distribution']:
            lines.append(
                f"| {dist['range']} | {dist['count']} | {dist['percentage']}% |"
            )

        return "\n".join(lines)

    def generate_analysis_section(self) -> str:
        """
        Generate the full confidence analysis section for reports.

        Returns:
            Markdown formatted analysis section
        """
        results = self.analyze_distribution()

        lines = [
            "### Transcription Confidence Distribution",
            "",
            self.generate_distribution_table(),
            "",
            f"**Average Confidence:** {results['average_confidence']:.1%}",
            f"**Quality Assessment:** {results['quality_assessment']}",
            "",
        ]

        if results['training_implications']:
            lines.append("**Training Implications:**")
            for impl in results['training_implications']:
                lines.append(f"- {impl}")
            lines.append("")

        return "\n".join(lines)

    def get_structured_results(self) -> Dict[str, Any]:
        """
        Get structured results suitable for LLM prompts.

        Returns:
            Dictionary with all confidence analysis data
        """
        results = self.analyze_distribution()

        return {
            'distribution_table': self.generate_distribution_table(),
            'analysis_section': self.generate_analysis_section(),
            'statistics': {
                'total_utterances': results['total_utterances'],
                'average_confidence': results['average_confidence'],
                'min_confidence': results['min_confidence'],
                'max_confidence': results['max_confidence'],
                'std_deviation': results['std_deviation'],
            },
            'distribution': results['distribution'],
            'speaker_stats': results['speaker_stats'],
            'quality_assessment': results['quality_assessment'],
            'training_implications': results['training_implications']
        }
