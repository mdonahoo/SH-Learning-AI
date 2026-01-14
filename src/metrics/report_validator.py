"""
Report validation module for verifying LLM-generated reports.

Compares LLM output against pre-computed facts to detect hallucinations
and ensure data integrity in generated reports.
"""

import logging
import re
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"      # Critical mismatch - data is wrong
    WARNING = "warning"  # Potential issue - should be reviewed
    INFO = "info"        # Minor discrepancy - likely acceptable


@dataclass
class ValidationIssue:
    """Represents a single validation issue found in a report."""
    severity: ValidationSeverity
    category: str
    message: str
    expected: Any
    found: Any
    location: Optional[str] = None

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.category}: {self.message}"


class ReportValidator:
    """
    Validates LLM-generated reports against pre-computed facts.

    This class extracts claims from generated reports and compares them
    against the structured data that was computed by Python, catching
    cases where the LLM invented or miscounted data.
    """

    def __init__(
        self,
        structured_data: Dict[str, Any],
        tolerance_percent: float = 5.0
    ):
        """
        Initialize validator with ground truth data.

        Args:
            structured_data: Pre-computed facts from LearningEvaluator
            tolerance_percent: Acceptable percentage deviation for numeric values
        """
        self.structured_data = structured_data
        self.tolerance_percent = tolerance_percent
        self.issues: List[ValidationIssue] = []

    def validate_report(self, report_text: str) -> List[ValidationIssue]:
        """
        Validate a generated report against known facts.

        Args:
            report_text: The LLM-generated report markdown

        Returns:
            List of validation issues found
        """
        self.issues = []

        # Run all validation checks
        self._validate_speaker_counts(report_text)
        self._validate_total_communications(report_text)
        self._validate_mission_metrics(report_text)
        self._validate_objective_status(report_text)
        self._validate_framework_scores(report_text)
        self._validate_quotes_exist(report_text)

        # Log summary
        error_count = sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)

        if error_count > 0:
            logger.error(f"Report validation failed: {error_count} errors, {warning_count} warnings")
        elif warning_count > 0:
            logger.warning(f"Report validation passed with warnings: {warning_count} warnings")
        else:
            logger.info("Report validation passed: No issues found")

        return self.issues

    def _validate_speaker_counts(self, report_text: str) -> None:
        """Validate speaker utterance counts mentioned in report."""
        expected_stats = {
            stat['speaker']: stat['utterances']
            for stat in self.structured_data.get('speaker_statistics', [])
        }

        # Pattern to match speaker counts like "speaker_1: 54 utterances" or "speaker_1 (54)"
        patterns = [
            r'(speaker_\d+)[:\s]+(\d+)\s*utterances?',
            r'(speaker_\d+)\s*\((\d+)\)',
            r'(speaker_\d+)[:\s]+(\d+)\s*\(',
            r'\|\s*(speaker_\d+)\s*\|\s*(\d+)\s*\|',  # Table format
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, report_text, re.IGNORECASE):
                speaker = match.group(1).lower()
                claimed_count = int(match.group(2))

                if speaker in expected_stats:
                    expected_count = expected_stats[speaker]
                    if claimed_count != expected_count:
                        deviation = abs(claimed_count - expected_count) / expected_count * 100
                        severity = (
                            ValidationSeverity.ERROR if deviation > 20
                            else ValidationSeverity.WARNING if deviation > self.tolerance_percent
                            else ValidationSeverity.INFO
                        )
                        self.issues.append(ValidationIssue(
                            severity=severity,
                            category="speaker_count",
                            message=f"{speaker} utterance count mismatch",
                            expected=expected_count,
                            found=claimed_count,
                            location=match.group(0)
                        ))

    def _validate_total_communications(self, report_text: str) -> None:
        """Validate total communication count."""
        expected_total = self.structured_data.get('metadata', {}).get('total_communications', 0)

        # Pattern to match total communications
        patterns = [
            r'total\s+communications?[:\s]+(\d+)',
            r'(\d+)\s+(?:total\s+)?communications?',
            r'(\d+)\s+utterances?',
            r'crew\s+communications?[:\s]+(\d+)',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, report_text, re.IGNORECASE):
                claimed_total = int(match.group(1))

                # Skip if this is clearly a per-speaker count (small number)
                if claimed_total < 20:
                    continue

                if claimed_total != expected_total:
                    deviation = abs(claimed_total - expected_total) / expected_total * 100 if expected_total > 0 else 100
                    if deviation > self.tolerance_percent:
                        self.issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR if deviation > 20 else ValidationSeverity.WARNING,
                            category="total_communications",
                            message="Total communications count mismatch",
                            expected=expected_total,
                            found=claimed_total,
                            location=match.group(0)
                        ))
                        break  # Only report once

    def _validate_mission_metrics(self, report_text: str) -> None:
        """Validate mission-specific metrics."""
        evaluations = self.structured_data.get('evaluations', {})
        mission_specific = evaluations.get('mission_specific', {})

        # Validate duration format if mentioned
        expected_duration = mission_specific.get('mission_duration', '')
        if expected_duration and expected_duration != "Unknown":
            # Look for duration mentions
            duration_pattern = r'duration[:\s]+([0-9:]+)'
            for match in re.finditer(duration_pattern, report_text, re.IGNORECASE):
                claimed_duration = match.group(1)
                if claimed_duration != expected_duration:
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="duration",
                        message="Mission duration mismatch",
                        expected=expected_duration,
                        found=claimed_duration,
                        location=match.group(0)
                    ))

    def _validate_objective_status(self, report_text: str) -> None:
        """Validate objective completion claims."""
        objectives = self.structured_data.get('objectives', {})
        obj_details = objectives.get('details', {})

        expected_completed = objectives.get('completed', 0)
        expected_total = objectives.get('total', 0)

        # Pattern to match objective completion like "5/10 objectives" or "50% completion"
        patterns = [
            r'(\d+)\s*/\s*(\d+)\s*objectives?',
            r'(\d+)\s+of\s+(\d+)\s+objectives?',
            r'completed?\s*[:=]?\s*(\d+)',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, report_text, re.IGNORECASE):
                groups = match.groups()
                if len(groups) >= 2:
                    claimed_completed = int(groups[0])
                    claimed_total = int(groups[1])

                    if claimed_total != expected_total:
                        self.issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category="objectives_total",
                            message="Total objectives count mismatch",
                            expected=expected_total,
                            found=claimed_total,
                            location=match.group(0)
                        ))

                    if claimed_completed != expected_completed:
                        self.issues.append(ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category="objectives_completed",
                            message="Completed objectives count mismatch",
                            expected=expected_completed,
                            found=claimed_completed,
                            location=match.group(0)
                        ))
                    break

    def _validate_framework_scores(self, report_text: str) -> None:
        """Validate framework scores (Kirkpatrick, NASA, Bloom's)."""
        evaluations = self.structured_data.get('evaluations', {})

        # Validate NASA teamwork score
        nasa = evaluations.get('nasa_teamwork', {})
        expected_nasa_score = nasa.get('overall_teamwork_score')

        if expected_nasa_score is not None:
            pattern = r'(?:overall\s+)?teamwork\s+(?:score)?[:\s]+(\d+(?:\.\d+)?)\s*/?\s*100'
            for match in re.finditer(pattern, report_text, re.IGNORECASE):
                claimed_score = float(match.group(1))
                if abs(claimed_score - expected_nasa_score) > self.tolerance_percent:
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="nasa_teamwork_score",
                        message="NASA teamwork score mismatch",
                        expected=expected_nasa_score,
                        found=claimed_score,
                        location=match.group(0)
                    ))

    def _validate_quotes_exist(self, report_text: str) -> None:
        """Validate that quoted text exists in available quotes."""
        top_comms = self.structured_data.get('top_communications', [])
        available_quotes = {comm['text'].lower().strip() for comm in top_comms}

        # Find all quoted text in the report
        quote_pattern = r'"([^"]+)"'
        for match in re.finditer(quote_pattern, report_text):
            quoted_text = match.group(1).lower().strip()

            # Skip very short quotes or common phrases
            if len(quoted_text) < 10:
                continue

            # Skip known non-dialogue quotes (section headers, etc.)
            skip_phrases = ['confidence:', 'score:', 'level', 'assessment', 'recommendation']
            if any(phrase in quoted_text for phrase in skip_phrases):
                continue

            # Check if this quote exists in available quotes
            quote_found = any(
                quoted_text in avail or avail in quoted_text
                for avail in available_quotes
            )

            if not quote_found and len(available_quotes) > 0:
                # This might be a hallucinated quote
                self.issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category="quote_verification",
                    message="Quote not found in available communications",
                    expected="Quote from provided data",
                    found=match.group(1)[:50] + "..." if len(match.group(1)) > 50 else match.group(1),
                    location=None
                ))

    def get_summary(self) -> Dict[str, Any]:
        """
        Get validation summary.

        Returns:
            Summary dictionary with counts and issues
        """
        return {
            'total_issues': len(self.issues),
            'errors': sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR),
            'warnings': sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING),
            'info': sum(1 for i in self.issues if i.severity == ValidationSeverity.INFO),
            'passed': all(i.severity != ValidationSeverity.ERROR for i in self.issues),
            'issues': [
                {
                    'severity': i.severity.value,
                    'category': i.category,
                    'message': i.message,
                    'expected': i.expected,
                    'found': i.found,
                    'location': i.location
                }
                for i in self.issues
            ]
        }

    def format_report(self) -> str:
        """
        Format validation results as a readable report.

        Returns:
            Formatted validation report string
        """
        lines = ["=" * 60, "REPORT VALIDATION RESULTS", "=" * 60, ""]

        summary = self.get_summary()
        status = "✓ PASSED" if summary['passed'] else "✗ FAILED"

        lines.append(f"Status: {status}")
        lines.append(f"Total Issues: {summary['total_issues']}")
        lines.append(f"  Errors: {summary['errors']}")
        lines.append(f"  Warnings: {summary['warnings']}")
        lines.append(f"  Info: {summary['info']}")
        lines.append("")

        if self.issues:
            lines.append("Issues Found:")
            lines.append("-" * 40)

            for issue in sorted(self.issues, key=lambda x: x.severity.value):
                lines.append(f"\n[{issue.severity.value.upper()}] {issue.category}")
                lines.append(f"  Message: {issue.message}")
                lines.append(f"  Expected: {issue.expected}")
                lines.append(f"  Found: {issue.found}")
                if issue.location:
                    lines.append(f"  Location: {issue.location}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


def validate_report(
    report_text: str,
    structured_data: Dict[str, Any],
    tolerance_percent: float = 5.0
) -> Tuple[bool, List[ValidationIssue]]:
    """
    Convenience function to validate a report.

    Args:
        report_text: The LLM-generated report
        structured_data: Pre-computed facts from LearningEvaluator
        tolerance_percent: Acceptable percentage deviation

    Returns:
        Tuple of (passed: bool, issues: List[ValidationIssue])
    """
    validator = ReportValidator(structured_data, tolerance_percent)
    issues = validator.validate_report(report_text)
    passed = all(i.severity != ValidationSeverity.ERROR for i in issues)
    return passed, issues
