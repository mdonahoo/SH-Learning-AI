"""
Tests for enhanced analysis modules.

Tests cover:
- RoleInferenceEngine
- ConfidenceAnalyzer
- MissionPhaseAnalyzer
- QualityVerifier
- SpeakerScorecardGenerator
- CommunicationQualityAnalyzer
- EnhancedReportBuilder
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.metrics.role_inference import (
    RoleInferenceEngine,
    BridgeRole,
    RolePatterns,
)
from src.metrics.confidence_analyzer import (
    ConfidenceAnalyzer,
    CONFIDENCE_RANGES,
)
from src.metrics.phase_analyzer import (
    MissionPhaseAnalyzer,
    PHASE_DEFINITIONS,
)
from src.metrics.quality_verifier import (
    QualityVerifier,
    VerificationCheck,
)
from src.metrics.speaker_scorecard import (
    SpeakerScorecardGenerator,
    SCORE_METRICS,
)
from src.metrics.communication_quality import (
    CommunicationQualityAnalyzer,
    EFFECTIVE_PATTERNS,
    IMPROVEMENT_PATTERNS,
)
from src.metrics.enhanced_report_builder import EnhancedReportBuilder


# Test fixtures
@pytest.fixture
def sample_transcripts():
    """Sample transcripts for testing."""
    base_time = datetime(2025, 12, 20, 21, 0, 0)
    return [
        {
            'timestamp': (base_time + timedelta(seconds=0)).isoformat(),
            'speaker': 'speaker_1',
            'text': 'Set course for sector 7, engage.',
            'confidence': 0.92
        },
        {
            'timestamp': (base_time + timedelta(seconds=5)).isoformat(),
            'speaker': 'speaker_2',
            'text': 'Aye captain, course laid in.',
            'confidence': 0.88
        },
        {
            'timestamp': (base_time + timedelta(seconds=10)).isoformat(),
            'speaker': 'speaker_1',
            'text': 'Tactical, what is our shield status?',
            'confidence': 0.85
        },
        {
            'timestamp': (base_time + timedelta(seconds=15)).isoformat(),
            'speaker': 'speaker_3',
            'text': 'Shields at 100%, all systems nominal.',
            'confidence': 0.90
        },
        {
            'timestamp': (base_time + timedelta(seconds=20)).isoformat(),
            'speaker': 'speaker_1',
            'text': 'Good work everyone. Stand by.',
            'confidence': 0.87
        },
        {
            'timestamp': (base_time + timedelta(seconds=25)).isoformat(),
            'speaker': 'speaker_4',
            'text': 'Uh, where is the, um...',
            'confidence': 0.45
        },
        {
            'timestamp': (base_time + timedelta(seconds=30)).isoformat(),
            'speaker': 'speaker_1',
            'text': 'Stop us within 20 kilometers.',
            'confidence': 0.91
        },
        {
            'timestamp': (base_time + timedelta(seconds=35)).isoformat(),
            'speaker': 'speaker_2',
            'text': 'Okay.',
            'confidence': 0.95
        },
    ]


@pytest.fixture
def sample_events():
    """Sample game events for testing."""
    base_time = datetime(2025, 12, 20, 21, 0, 0)
    return [
        {
            'timestamp': base_time.isoformat(),
            'event_type': 'mission_update',
            'data': {
                'State': 'Running',
                'Grade': 0.75,
                'Objectives': {
                    'Primary_1': {
                        'Description': 'Complete the mission',
                        'Complete': True,
                        'Rank': 'Primary'
                    },
                    'Secondary_1': {
                        'Description': 'Collect data',
                        'Complete': False,
                        'Rank': 'Secondary'
                    }
                }
            }
        },
        {
            'timestamp': (base_time + timedelta(minutes=30)).isoformat(),
            'event_type': 'navigation_update',
            'data': {'position': [100, 200, 300]}
        }
    ]


class TestRoleInferenceEngine:
    """Tests for RoleInferenceEngine."""

    def test_initialization(self, sample_transcripts):
        """Test engine initializes correctly."""
        engine = RoleInferenceEngine(sample_transcripts)
        assert engine.transcripts == sample_transcripts
        assert engine.patterns is not None

    def test_analyze_all_speakers(self, sample_transcripts):
        """Test analyzing all speakers."""
        engine = RoleInferenceEngine(sample_transcripts)
        results = engine.analyze_all_speakers()

        assert 'speaker_1' in results
        assert 'speaker_2' in results
        assert 'speaker_3' in results

        # Speaker 1 should be identified as command
        assert results['speaker_1'].inferred_role in [
            BridgeRole.CAPTAIN,
            BridgeRole.EXECUTIVE_OFFICER
        ]

    def test_generate_role_table(self, sample_transcripts):
        """Test markdown table generation."""
        engine = RoleInferenceEngine(sample_transcripts)
        table = engine.generate_role_analysis_table()

        assert '| Speaker |' in table
        assert 'speaker_1' in table
        assert '---' in table

    def test_generate_methodology_section(self, sample_transcripts):
        """Test methodology explanation generation."""
        engine = RoleInferenceEngine(sample_transcripts)
        methodology = engine.generate_methodology_section()

        assert 'Role Assignment Methodology' in methodology
        assert 'keyword frequency' in methodology.lower()

    def test_structured_results(self, sample_transcripts):
        """Test structured results format."""
        engine = RoleInferenceEngine(sample_transcripts)
        results = engine.get_structured_results()

        assert 'role_table' in results
        assert 'methodology' in results
        assert 'speaker_roles' in results
        assert 'speaker_1' in results['speaker_roles']


class TestConfidenceAnalyzer:
    """Tests for ConfidenceAnalyzer."""

    def test_initialization(self, sample_transcripts):
        """Test analyzer initializes correctly."""
        analyzer = ConfidenceAnalyzer(sample_transcripts)
        assert analyzer.transcripts == sample_transcripts

    def test_analyze_distribution(self, sample_transcripts):
        """Test confidence distribution analysis."""
        analyzer = ConfidenceAnalyzer(sample_transcripts)
        results = analyzer.analyze_distribution()

        assert 'total_utterances' in results
        assert results['total_utterances'] == len(sample_transcripts)
        assert 'distribution' in results
        assert len(results['distribution']) == len(CONFIDENCE_RANGES)

    def test_training_implications(self, sample_transcripts):
        """Test training implications generation."""
        analyzer = ConfidenceAnalyzer(sample_transcripts)
        results = analyzer.analyze_distribution()

        assert 'training_implications' in results
        assert isinstance(results['training_implications'], list)

    def test_distribution_table(self, sample_transcripts):
        """Test markdown table generation."""
        analyzer = ConfidenceAnalyzer(sample_transcripts)
        table = analyzer.generate_distribution_table()

        assert '| Confidence Range |' in table
        assert '90% and above' in table

    def test_quality_assessment(self, sample_transcripts):
        """Test quality assessment."""
        analyzer = ConfidenceAnalyzer(sample_transcripts)
        results = analyzer.analyze_distribution()

        assert 'quality_assessment' in results
        assert results['quality_assessment'] != ''


class TestMissionPhaseAnalyzer:
    """Tests for MissionPhaseAnalyzer."""

    def test_initialization(self, sample_transcripts, sample_events):
        """Test analyzer initializes correctly."""
        analyzer = MissionPhaseAnalyzer(sample_transcripts, sample_events)
        assert len(analyzer.transcripts) == len(sample_transcripts)

    def test_analyze_phases(self, sample_transcripts, sample_events):
        """Test phase detection."""
        analyzer = MissionPhaseAnalyzer(
            sample_transcripts,
            sample_events,
            min_phase_utterances=2
        )
        phases = analyzer.analyze_phases()

        assert len(phases) >= 1
        assert all(hasattr(p, 'phase_number') for p in phases)
        assert all(hasattr(p, 'utterance_count') for p in phases)

    def test_phase_analysis_section(self, sample_transcripts, sample_events):
        """Test phase analysis section generation."""
        analyzer = MissionPhaseAnalyzer(
            sample_transcripts,
            sample_events,
            min_phase_utterances=2
        )
        section = analyzer.generate_phase_analysis_section()

        assert '## Mission Phase Analysis' in section
        assert 'Phase' in section

    def test_structured_results(self, sample_transcripts, sample_events):
        """Test structured results format."""
        analyzer = MissionPhaseAnalyzer(
            sample_transcripts,
            sample_events,
            min_phase_utterances=2
        )
        results = analyzer.get_structured_results()

        assert 'phase_analysis_section' in results
        assert 'total_phases' in results
        assert 'phases' in results


class TestQualityVerifier:
    """Tests for QualityVerifier."""

    def test_initialization(self, sample_transcripts, sample_events):
        """Test verifier initializes correctly."""
        verifier = QualityVerifier(sample_transcripts, sample_events)
        assert verifier.transcripts == sample_transcripts

    def test_run_all_checks(self, sample_transcripts, sample_events):
        """Test running all verification checks."""
        verifier = QualityVerifier(sample_transcripts, sample_events)
        checks = verifier.run_all_checks()

        assert len(checks) > 0
        assert all(isinstance(c, VerificationCheck) for c in checks)
        assert all(hasattr(c, 'status') for c in checks)

    def test_verification_table(self, sample_transcripts, sample_events):
        """Test verification table generation."""
        verifier = QualityVerifier(sample_transcripts, sample_events)
        table = verifier.generate_verification_table()

        assert '| Check |' in table
        assert 'Status' in table

    def test_data_capture_gaps(self, sample_transcripts, sample_events):
        """Test data capture gap identification."""
        verifier = QualityVerifier(sample_transcripts, sample_events)
        gaps = verifier.get_data_capture_gaps()

        assert isinstance(gaps, list)
        assert len(gaps) > 0

    def test_verification_section(self, sample_transcripts, sample_events):
        """Test complete verification section."""
        verifier = QualityVerifier(sample_transcripts, sample_events)
        section = verifier.generate_verification_section()

        assert '## Quality Verification' in section
        assert 'Data Accuracy Checks' in section


class TestSpeakerScorecardGenerator:
    """Tests for SpeakerScorecardGenerator."""

    def test_initialization(self, sample_transcripts):
        """Test generator initializes correctly."""
        generator = SpeakerScorecardGenerator(sample_transcripts)
        assert len(generator.speaker_utterances) > 0

    def test_generate_all_scorecards(self, sample_transcripts):
        """Test scorecard generation for all speakers."""
        generator = SpeakerScorecardGenerator(sample_transcripts)
        scorecards = generator.generate_all_scorecards()

        assert 'speaker_1' in scorecards
        assert all(hasattr(sc, 'scores') for sc in scorecards.values())
        assert all(hasattr(sc, 'overall_score') for sc in scorecards.values())

    def test_scorecard_metrics(self, sample_transcripts):
        """Test that all metrics are calculated."""
        generator = SpeakerScorecardGenerator(sample_transcripts)
        scorecards = generator.generate_all_scorecards()

        sc = scorecards['speaker_1']
        metric_names = [s.metric_name for s in sc.scores]

        assert 'protocol_adherence' in metric_names
        assert 'communication_clarity' in metric_names
        assert 'response_time' in metric_names

    def test_scorecards_section(self, sample_transcripts):
        """Test scorecards section generation."""
        generator = SpeakerScorecardGenerator(sample_transcripts)
        section = generator.generate_all_scorecards_section()

        assert '## Crew Performance Scorecards' in section
        assert 'speaker_1' in section
        assert '/5' in section

    def test_with_role_assignments(self, sample_transcripts):
        """Test with provided role assignments."""
        roles = {'speaker_1': 'Captain', 'speaker_2': 'Helm'}
        generator = SpeakerScorecardGenerator(sample_transcripts, role_assignments=roles)
        scorecards = generator.generate_all_scorecards()

        assert scorecards['speaker_1'].inferred_role == 'Captain'
        assert scorecards['speaker_2'].inferred_role == 'Helm'


class TestCommunicationQualityAnalyzer:
    """Tests for CommunicationQualityAnalyzer."""

    def test_initialization(self, sample_transcripts):
        """Test analyzer initializes correctly."""
        analyzer = CommunicationQualityAnalyzer(sample_transcripts)
        assert analyzer.transcripts == sample_transcripts

    def test_analyze_all(self, sample_transcripts):
        """Test full analysis."""
        analyzer = CommunicationQualityAnalyzer(sample_transcripts)
        results = analyzer.analyze_all()

        assert 'effective' in results
        assert 'needs_improvement' in results
        assert 'statistics' in results

    def test_detect_effective_patterns(self, sample_transcripts):
        """Test detection of effective communication patterns."""
        analyzer = CommunicationQualityAnalyzer(sample_transcripts)
        results = analyzer.analyze_all()

        # "Set course for sector 7, engage" should be effective
        effective_texts = [e.text for e in results['effective']]
        assert any('engage' in t.lower() for t in effective_texts)

    def test_detect_improvement_patterns(self, sample_transcripts):
        """Test detection of patterns needing improvement."""
        analyzer = CommunicationQualityAnalyzer(sample_transcripts)
        results = analyzer.analyze_all()

        # "Uh, where is the, um..." should need improvement
        improvement_texts = [e.text for e in results['needs_improvement']]
        assert any('uh' in t.lower() for t in improvement_texts)

    def test_command_control_section(self, sample_transcripts):
        """Test command/control section generation."""
        analyzer = CommunicationQualityAnalyzer(sample_transcripts)
        section = analyzer.generate_command_control_section()

        assert '## Command and Control Assessment' in section
        assert 'Effective Command Examples' in section
        assert 'Communications Requiring Improvement' in section


class TestEnhancedReportBuilder:
    """Tests for EnhancedReportBuilder."""

    def test_initialization(self, sample_transcripts, sample_events):
        """Test builder initializes correctly."""
        builder = EnhancedReportBuilder(sample_transcripts, sample_events)
        assert builder.transcripts == sample_transcripts
        assert builder.events == sample_events

    def test_build_all_analyses(self, sample_transcripts, sample_events):
        """Test building all analyses."""
        builder = EnhancedReportBuilder(sample_transcripts, sample_events)
        analyses = builder.build_all_analyses()

        assert 'role_analysis' in analyses
        assert 'confidence_analysis' in analyses
        assert 'phase_analysis' in analyses
        assert 'quality_verification' in analyses
        assert 'communication_quality' in analyses
        assert 'speaker_scorecards' in analyses

    def test_build_mission_statistics(self, sample_transcripts, sample_events):
        """Test mission statistics building."""
        builder = EnhancedReportBuilder(sample_transcripts, sample_events)
        stats = builder.build_mission_statistics()

        assert 'total_voice_communications' in stats
        assert stats['total_voice_communications'] == len(sample_transcripts)
        assert 'unique_speakers' in stats

    def test_generate_statistics_table(self, sample_transcripts, sample_events):
        """Test statistics table generation."""
        builder = EnhancedReportBuilder(sample_transcripts, sample_events)
        table = builder.generate_statistics_table()

        assert '## Mission Statistics' in table
        assert '| Metric | Value |' in table

    def test_generate_full_report(self, sample_transcripts, sample_events):
        """Test full report generation."""
        builder = EnhancedReportBuilder(
            sample_transcripts,
            sample_events,
            mission_data={'mission_name': 'Test Mission'}
        )
        report = builder.generate_full_report()

        assert '# Mission Debrief: Test Mission' in report
        assert '## Executive Summary' in report
        assert '## Role Analysis' in report
        assert '## Quality Verification' in report

    def test_get_all_structured_data(self, sample_transcripts, sample_events):
        """Test getting all structured data."""
        builder = EnhancedReportBuilder(sample_transcripts, sample_events)
        data = builder.get_all_structured_data()

        assert 'mission_statistics' in data
        assert 'full_report' in data
        assert 'role_analysis' in data


class TestIntegration:
    """Integration tests for the complete analysis pipeline."""

    def test_full_pipeline(self, sample_transcripts, sample_events):
        """Test the complete analysis pipeline."""
        builder = EnhancedReportBuilder(
            sample_transcripts,
            sample_events,
            mission_data={'mission_name': 'Integration Test Mission'}
        )

        # Generate full report
        report = builder.generate_full_report()

        # Verify all major sections present
        assert '# Mission Debrief' in report
        assert '## Executive Summary' in report
        assert '## Mission Statistics' in report
        assert '## Role Analysis' in report
        assert '## Command and Control Assessment' in report
        assert '## Quality Verification' in report

        # Verify data integrity
        assert 'speaker_1' in report
        assert 'speaker_2' in report

    def test_empty_transcripts(self):
        """Test handling of empty transcripts."""
        builder = EnhancedReportBuilder([], [])
        stats = builder.build_mission_statistics()

        assert stats['total_voice_communications'] == 0
        assert stats['unique_speakers'] == 0

    def test_missing_confidence(self):
        """Test handling of missing confidence values."""
        transcripts = [
            {'speaker': 'speaker_1', 'text': 'Hello', 'timestamp': '2025-01-01T00:00:00'}
        ]
        analyzer = ConfidenceAnalyzer(transcripts)
        results = analyzer.analyze_distribution()

        assert results['total_utterances'] == 1
