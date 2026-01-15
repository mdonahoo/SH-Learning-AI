"""
Enhanced report builder that integrates all analysis components.

This module provides a unified interface for generating comprehensive
mission debrief reports matching the quality of professional examples.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import Counter

from src.metrics.role_inference import RoleInferenceEngine
from src.metrics.confidence_analyzer import ConfidenceAnalyzer
from src.metrics.phase_analyzer import MissionPhaseAnalyzer
from src.metrics.quality_verifier import QualityVerifier
from src.metrics.speaker_scorecard import SpeakerScorecardGenerator
from src.metrics.communication_quality import CommunicationQualityAnalyzer
from src.metrics.learning_evaluator import LearningEvaluator

logger = logging.getLogger(__name__)


class EnhancedReportBuilder:
    """
    Builds comprehensive mission reports using all analysis components.

    Integrates role inference, confidence analysis, phase detection,
    quality verification, speaker scorecards, and communication quality
    into a unified report structure.
    """

    def __init__(
        self,
        transcripts: List[Dict[str, Any]],
        events: List[Dict[str, Any]] = None,
        mission_data: Dict[str, Any] = None
    ):
        """
        Initialize the enhanced report builder.

        Args:
            transcripts: List of transcript dictionaries
            events: List of game event dictionaries
            mission_data: Additional mission metadata
        """
        self.transcripts = transcripts
        self.events = events or []
        self.mission_data = mission_data or {}

        # Initialize all analyzers
        self.role_engine = RoleInferenceEngine(transcripts)
        self.confidence_analyzer = ConfidenceAnalyzer(transcripts)
        self.phase_analyzer = MissionPhaseAnalyzer(transcripts, events)
        self.quality_verifier = QualityVerifier(transcripts, events, mission_data)
        self.communication_analyzer = CommunicationQualityAnalyzer(transcripts)
        self.learning_evaluator = LearningEvaluator(events, transcripts)

        # Role assignments will be computed and shared
        self._role_results = None

    def _get_role_assignments(self) -> Dict[str, str]:
        """Get role assignments for all speakers."""
        if self._role_results is None:
            self._role_results = self.role_engine.analyze_all_speakers()

        return {
            speaker: analysis.inferred_role.value
            for speaker, analysis in self._role_results.items()
        }

    def build_all_analyses(self) -> Dict[str, Any]:
        """
        Run all analyses and collect results.

        Returns:
            Dictionary containing all analysis results
        """
        role_results = self.role_engine.get_structured_results()
        confidence_results = self.confidence_analyzer.get_structured_results()
        phase_results = self.phase_analyzer.get_structured_results()
        quality_results = self.quality_verifier.get_structured_results()
        communication_results = self.communication_analyzer.get_structured_results()

        # Speaker scorecards need role assignments
        role_assignments = self._get_role_assignments()
        scorecard_generator = SpeakerScorecardGenerator(
            self.transcripts,
            role_assignments=role_assignments
        )
        scorecard_results = scorecard_generator.get_structured_results()

        # Learning evaluator results
        learning_results = self.learning_evaluator.evaluate_all()

        return {
            'role_analysis': role_results,
            'confidence_analysis': confidence_results,
            'phase_analysis': phase_results,
            'quality_verification': quality_results,
            'communication_quality': communication_results,
            'speaker_scorecards': scorecard_results,
            'learning_evaluation': learning_results,
        }

    def build_mission_statistics(self) -> Dict[str, Any]:
        """
        Build mission statistics section.

        Returns:
            Dictionary with mission statistics
        """
        total_utterances = len(self.transcripts)
        speakers = Counter(
            t.get('speaker') or t.get('speaker_id') or 'unknown'
            for t in self.transcripts
        )

        # Calculate duration
        duration_str = "Unknown"
        timestamps = []
        for t in self.transcripts:
            ts = t.get('timestamp', '')
            try:
                if isinstance(ts, datetime):
                    timestamps.append(ts)
                elif isinstance(ts, str) and 'T' in ts:
                    timestamps.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
            except (ValueError, TypeError):
                pass

        if timestamps:
            duration = max(timestamps) - min(timestamps)
            total_seconds = duration.total_seconds()
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)
            duration_str = f"{minutes} minutes {seconds} seconds"

        # Extract objectives
        objectives = self._extract_objectives()
        completed = sum(1 for obj in objectives.values() if obj.get('complete', False))
        total_objectives = len(objectives)

        # Get mission grade
        mission_grade = None
        for event in reversed(self.events):
            if event.get('event_type') == 'mission_update':
                data = event.get('data', {})
                if 'Grade' in data and data['Grade'] is not None:
                    mission_grade = data['Grade']
                    break

        return {
            'mission_duration': duration_str,
            'total_voice_communications': total_utterances,
            'unique_speakers': len(speakers),
            'total_game_events': len(self.events),
            'objectives_total': total_objectives,
            'objectives_completed': completed,
            'completion_rate': round(completed / total_objectives * 100, 1) if total_objectives > 0 else 0,
            'mission_grade': mission_grade,
        }

    def _extract_objectives(self) -> Dict[str, Dict[str, Any]]:
        """Extract mission objectives from events."""
        objectives = {}

        for event in self.events:
            event_type = event.get('event_type') or event.get('type', '')
            data = event.get('data', {})

            if event_type == 'mission_update' and 'Objectives' in data:
                obj_data = data.get('Objectives', {})
                for obj_name, obj_details in obj_data.items():
                    if isinstance(obj_details, dict):
                        objectives[obj_name] = {
                            'description': obj_details.get('Description', obj_name),
                            'complete': obj_details.get('Complete', False),
                            'rank': obj_details.get('Rank', 'Unknown'),
                            'group': obj_details.get('Group', 'Main'),
                        }

        return objectives

    def generate_statistics_table(self) -> str:
        """
        Generate mission statistics markdown table.

        Returns:
            Markdown formatted statistics table
        """
        stats = self.build_mission_statistics()

        lines = [
            "## Mission Statistics",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Mission Duration | {stats['mission_duration']} |",
            f"| Total Voice Communications | {stats['total_voice_communications']} utterances |",
            f"| Unique Speakers Detected | {stats['unique_speakers']} |",
            f"| Total Game Events | {stats['total_game_events']} |",
            f"| Mission Objectives | {stats['objectives_total']} total, {stats['objectives_completed']} completed ({stats['completion_rate']}%) |",
        ]

        if stats['mission_grade'] is not None:
            lines.append(f"| Mission Grade | {stats['mission_grade']:.2f} |")

        return "\n".join(lines)

    def generate_objectives_section(self) -> str:
        """
        Generate mission objectives status section.

        Returns:
            Markdown formatted objectives section
        """
        objectives = self._extract_objectives()

        if not objectives:
            return "## Mission Objectives Status\n\nNo objective data available."

        # Separate completed and incomplete
        completed = {k: v for k, v in objectives.items() if v.get('complete', False)}
        incomplete = {k: v for k, v in objectives.items() if not v.get('complete', False)}

        lines = [
            "## Mission Objectives Status",
            "",
        ]

        if completed:
            lines.append(f"### Completed Objectives ({len(completed)})")
            lines.append("")
            lines.append("| Objective | Group | Evidence |")
            lines.append("| --- | --- | --- |")
            for name, obj in completed.items():
                lines.append(f"| {obj['description']} | {obj.get('rank', 'Unknown')}/{obj.get('group', 'Main')} | Completed |")
            lines.append("")

        if incomplete:
            lines.append(f"### Incomplete Objectives ({len(incomplete)})")
            lines.append("")
            lines.append("| Objective | Group | Status |")
            lines.append("| --- | --- | --- |")
            for name, obj in incomplete.items():
                lines.append(f"| {obj['description']} | {obj.get('rank', 'Unknown')}/{obj.get('group', 'Main')} | Not completed |")
            lines.append("")

        stats = self.build_mission_statistics()
        lines.append(f"**Mission Completion:** {stats['objectives_completed']} of {stats['objectives_total']} objectives ({stats['completion_rate']}%)")

        if stats['mission_grade'] is not None:
            lines.append(f"**Mission Grade from Game System:** {stats['mission_grade']:.2f}")

        return "\n".join(lines)

    def generate_executive_summary(self) -> str:
        """
        Generate executive summary section.

        Returns:
            Markdown formatted executive summary
        """
        stats = self.build_mission_statistics()
        role_results = self.role_engine.get_structured_results()
        confidence_results = self.confidence_analyzer.get_structured_results()
        learning_results = self.learning_evaluator.evaluate_all()

        # Get mission name
        mission_name = self.mission_data.get('mission_name', 'Unknown Mission')

        # Build summary paragraphs
        lines = [
            "## Executive Summary",
            "",
            f"The crew conducted a {stats['mission_duration']} mission recording that captured "
            f"{stats['total_voice_communications']} voice communications from {stats['unique_speakers']} "
            f"distinct speakers.",
            "",
        ]

        # Performance summary
        if stats['objectives_total'] > 0:
            lines.append(
                f"The crew completed {stats['objectives_completed']} of {stats['objectives_total']} "
                f"mission objectives ({stats['completion_rate']}%)."
            )
            if stats['mission_grade'] is not None:
                lines.append(f" achieving a mission grade of {stats['mission_grade']:.2f}.")
            lines.append("")

        # Quality assessment
        quality_assessment = confidence_results.get('quality_assessment', '')
        if quality_assessment:
            lines.append(f"**Audio Quality Assessment:** {quality_assessment}")
            lines.append("")

        # Training priorities
        lines.append("**Training Priorities:**")
        implications = confidence_results.get('training_implications', [])
        for impl in implications[:3]:
            lines.append(f"- {impl}")
        lines.append("")

        return "\n".join(lines)

    def generate_full_report(self) -> str:
        """
        Generate the complete enhanced mission report.

        Returns:
            Complete markdown formatted report
        """
        mission_name = self.mission_data.get('mission_name', 'Mission Debrief')

        # Get all analysis results
        role_results = self.role_engine.get_structured_results()
        confidence_results = self.confidence_analyzer.get_structured_results()
        phase_results = self.phase_analyzer.get_structured_results()
        quality_results = self.quality_verifier.get_structured_results()
        communication_results = self.communication_analyzer.get_structured_results()

        # Get role assignments for scorecards
        role_assignments = self._get_role_assignments()
        scorecard_generator = SpeakerScorecardGenerator(
            self.transcripts,
            role_assignments=role_assignments
        )

        sections = [
            f"# Mission Debrief: {mission_name}",
            "",
            self.generate_executive_summary(),
            self.generate_statistics_table(),
            "",
            "## Role Analysis",
            "",
            role_results['role_table'],
            "",
            role_results['methodology'],
            "",
            communication_results['command_control_section'],
            "",
            confidence_results['analysis_section'],
            "",
            phase_results['phase_analysis_section'],
            "",
            scorecard_generator.generate_all_scorecards_section(),
            "",
            self.generate_objectives_section(),
            "",
            self._generate_training_recommendations(),
            "",
            quality_results['verification_section'],
            "",
            self._generate_report_metadata(),
        ]

        return "\n".join(sections)

    def _generate_training_recommendations(self) -> str:
        """Generate training recommendations section."""
        communication_results = self.communication_analyzer.get_structured_results()
        confidence_results = self.confidence_analyzer.get_structured_results()

        lines = [
            "## Training Recommendations",
            "",
            "### Immediate Actions for This Crew",
            "",
        ]

        # Based on communication quality
        improvement_patterns = communication_results.get('pattern_counts', {}).get('needs_improvement', {})

        if improvement_patterns.get('filler_words', 0) > 5:
            lines.append(
                "1. **Communication Clarity Drill:** Practice completing full sentences "
                "without filler words (uh, um). The high rate of filler words disrupts information flow."
            )
            lines.append("")

        if improvement_patterns.get('single_word_response', 0) > 10:
            lines.append(
                "2. **Acknowledgment Protocol Training:** Establish standard three-part "
                "acknowledgments (repeat order, confirm understanding, report completion) "
                "instead of single-word responses."
            )
            lines.append("")

        if improvement_patterns.get('incomplete_sentence', 0) > 5:
            lines.append(
                "3. **Complete Communication Practice:** Practice completing thoughts "
                "before transitioning to new topics. Trailing sentences disrupt crew coordination."
            )
            lines.append("")

        # Based on confidence analysis
        if confidence_results.get('statistics', {}).get('average_confidence', 1) < 0.7:
            lines.append(
                "4. **Voice Projection Training:** Practice clear enunciation and voice "
                "projection to improve transcription accuracy and crew understanding."
            )
            lines.append("")

        lines.extend([
            "### Protocol Improvements",
            "",
            "**Standard Bridge Callouts:**",
            "- Distance reports should include \"Distance to [target], [number] kilometers\"",
            "- Status reports should include \"[System] at [percentage], [status]\"",
            "- Navigation should include \"Course set for [destination], ETA [time]\"",
            "",
            "**Acknowledgment Procedures:**",
            "- Replace single word responses with \"[Order], acknowledged, executing\"",
            "- Status changes should be proactively reported without prompting",
            "",
            "### Team Exercises",
            "",
            "1. **Closed-Loop Communication Drill:** Practice the full command cycle with "
            "mandatory readback of all orders before execution.",
            "",
            "2. **Role Isolation Exercise:** Run scenarios where each station must handle "
            "their specialty without command intervention to build role confidence.",
            "",
            "3. **Time Pressure Scenarios:** Practice operations under time constraints "
            "to improve response times and decision-making under stress.",
            "",
        ])

        return "\n".join(lines)

    def _generate_report_metadata(self) -> str:
        """Generate report metadata section."""
        lines = [
            "---",
            "",
            f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "**Data Sources:** transcripts.json, game_events.json",
            "**Analysis Method:** Complete dataset analysis with keyword frequency role inference",
        ]

        return "\n".join(lines)

    def get_all_structured_data(self) -> Dict[str, Any]:
        """
        Get all analysis data in structured format for LLM prompts.

        Returns:
            Dictionary with all pre-computed analysis data
        """
        analyses = self.build_all_analyses()
        statistics = self.build_mission_statistics()

        return {
            'mission_statistics': statistics,
            'full_report': self.generate_full_report(),
            **analyses
        }
