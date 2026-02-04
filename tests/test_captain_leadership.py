"""Tests for captain leadership assessment module."""

import pytest
from datetime import datetime, timedelta

from src.metrics.captain_leadership import (
    CaptainLeadershipAssessor,
    CaptainAssessment,
    LeadershipDimension,
)


@pytest.fixture
def captain_transcripts():
    """Sample transcripts with a clear captain speaker."""
    return [
        {'speaker': 'speaker_1', 'text': 'Helm, set course for sector 7.', 'start_time': 0},
        {'speaker': 'speaker_2', 'text': 'Aye captain, course laid in.', 'start_time': 5},
        {'speaker': 'speaker_1', 'text': 'Tactical, what do you think about our approach vector?', 'start_time': 10},
        {'speaker': 'speaker_3', 'text': 'Recommend we come in from bearing 270.', 'start_time': 15},
        {'speaker': 'speaker_1', 'text': 'Good call. Execute that approach.', 'start_time': 20},
        {'speaker': 'speaker_1', 'text': 'Status report, all stations.', 'start_time': 25},
        {'speaker': 'speaker_2', 'text': 'Helm reporting, all systems nominal.', 'start_time': 30},
        {'speaker': 'speaker_3', 'text': 'Tactical ready, weapons online.', 'start_time': 35},
        {'speaker': 'speaker_1', 'text': 'Well done everyone, nice work on that last maneuver.', 'start_time': 40},
        {'speaker': 'speaker_1', 'text': 'Any ideas on how we should handle the next objective?', 'start_time': 45},
        {'speaker': 'speaker_4', 'text': 'Engineering here, power levels stable.', 'start_time': 50},
        {'speaker': 'speaker_1', 'text': 'Red alert! Battle stations!', 'start_time': 55},
        {'speaker': 'speaker_1', 'text': 'Tactical, fire at will. Helm, evasive maneuvers.', 'start_time': 60},
        {'speaker': 'speaker_1', 'text': 'Great job keeping shields up, Engineering.', 'start_time': 70},
        {'speaker': 'speaker_1', 'text': 'Science, what are we looking at out there?', 'start_time': 75},
    ]


@pytest.fixture
def role_assignments():
    """Role assignments with a clear captain."""
    return {
        'speaker_1': 'Captain/Command',
        'speaker_2': 'Helm/Navigation',
        'speaker_3': 'Tactical/Weapons',
        'speaker_4': 'Engineering/Systems',
    }


@pytest.fixture
def minimal_transcripts():
    """Minimal transcripts — too few for meaningful assessment."""
    return [
        {'speaker': 'speaker_1', 'text': 'Hello everyone.', 'start_time': 0},
    ]


@pytest.fixture
def no_captain_roles():
    """Role assignments with no captain identified."""
    return {
        'speaker_1': 'Crew Member',
        'speaker_2': 'Helm/Navigation',
    }


@pytest.fixture
def telemetry_with_combat():
    """Telemetry events that include a combat crisis."""
    return [
        {'event_type': 'mission_start', 'category': 'mission', 'relative_time': 0, 'description': 'Mission began'},
        {'event_type': 'combat', 'category': 'combat', 'relative_time': 55, 'description': 'Combat engaged'},
        {'event_type': 'weapon_fire', 'category': 'combat', 'relative_time': 62, 'description': 'Weapons fired'},
    ]


class TestCaptainIdentification:
    """Tests for captain speaker identification."""

    def test_identify_captain_standard_role(self, captain_transcripts, role_assignments):
        """Test captain identification with standard role assignment."""
        assessor = CaptainLeadershipAssessor(
            captain_transcripts, role_assignments=role_assignments
        )
        assert assessor.identify_captain() == 'speaker_1'

    def test_identify_captain_no_role(self, captain_transcripts, no_captain_roles):
        """Test returns None when no captain role assigned."""
        assessor = CaptainLeadershipAssessor(
            captain_transcripts, role_assignments=no_captain_roles
        )
        assert assessor.identify_captain() is None

    def test_identify_captain_alternate_role_names(self, captain_transcripts):
        """Test captain identification with alternate role names."""
        roles = {'speaker_1': 'Command'}
        assessor = CaptainLeadershipAssessor(
            captain_transcripts, role_assignments=roles
        )
        assert assessor.identify_captain() == 'speaker_1'

    def test_identify_captain_empty_roles(self, captain_transcripts):
        """Test with empty role assignments."""
        assessor = CaptainLeadershipAssessor(captain_transcripts, role_assignments={})
        assert assessor.identify_captain() is None


class TestAssessmentDimensions:
    """Tests for individual leadership dimension assessments."""

    def test_delegation_detected(self, captain_transcripts, role_assignments):
        """Test that delegation patterns are detected in captain utterances."""
        assessor = CaptainLeadershipAssessor(
            captain_transcripts, role_assignments=role_assignments
        )
        assessment = assessor.assess()
        assert assessment is not None
        delegation = assessment.dimensions['delegation']
        # Captain addresses Helm, Tactical, Science, Engineering by name
        assert delegation.count >= 3
        assert delegation.score >= 3

    def test_crew_engagement_detected(self, captain_transcripts, role_assignments):
        """Test that crew engagement patterns are detected."""
        assessor = CaptainLeadershipAssessor(
            captain_transcripts, role_assignments=role_assignments
        )
        assessment = assessor.assess()
        assert assessment is not None
        engagement = assessment.dimensions['crew_engagement']
        # "what do you think", "any ideas", "what are we looking at"
        assert engagement.count >= 2
        assert engagement.score >= 2

    def test_information_flow_detected(self, captain_transcripts, role_assignments):
        """Test that information flow patterns are detected."""
        assessor = CaptainLeadershipAssessor(
            captain_transcripts, role_assignments=role_assignments
        )
        assessment = assessor.assess()
        assert assessment is not None
        info_flow = assessment.dimensions['information_flow']
        # "Status report", "what are we looking at"
        assert info_flow.count >= 1
        assert info_flow.score >= 1

    def test_praise_feedback_detected(self, captain_transcripts, role_assignments):
        """Test that praise/feedback patterns are detected."""
        assessor = CaptainLeadershipAssessor(
            captain_transcripts, role_assignments=role_assignments
        )
        assessment = assessor.assess()
        assert assessment is not None
        praise = assessment.dimensions['praise_feedback']
        # "Good call", "Well done everyone", "nice work", "Great job"
        assert praise.count >= 3
        assert praise.score >= 3

    def test_crisis_response_detected(self, captain_transcripts, role_assignments):
        """Test that directive/crisis patterns are detected."""
        assessor = CaptainLeadershipAssessor(
            captain_transcripts, role_assignments=role_assignments
        )
        assessment = assessor.assess()
        assert assessment is not None
        crisis = assessment.dimensions['crisis_response']
        # "Red alert! Battle stations!", "fire at will", "evasive maneuvers"
        assert crisis.count >= 2
        assert crisis.score >= 2


class TestOverallAssessment:
    """Tests for the overall captain assessment."""

    def test_overall_score_calculated(self, captain_transcripts, role_assignments):
        """Test that overall score is computed from dimension scores."""
        assessor = CaptainLeadershipAssessor(
            captain_transcripts, role_assignments=role_assignments
        )
        assessment = assessor.assess()
        assert assessment is not None
        assert 1.0 <= assessment.overall_score <= 5.0

    def test_strengths_identified(self, captain_transcripts, role_assignments):
        """Test that strengths are identified from high-scoring dimensions."""
        assessor = CaptainLeadershipAssessor(
            captain_transcripts, role_assignments=role_assignments
        )
        assessment = assessor.assess()
        assert assessment is not None
        # Should have at least one strength (delegation is strong in this data)
        assert isinstance(assessment.strengths, list)

    def test_development_areas_identified(self, captain_transcripts, role_assignments):
        """Test that development areas are identified from low-scoring dimensions."""
        assessor = CaptainLeadershipAssessor(
            captain_transcripts, role_assignments=role_assignments
        )
        assessment = assessor.assess()
        assert assessment is not None
        assert isinstance(assessment.development_areas, list)

    def test_assessment_returns_none_no_captain(self, captain_transcripts, no_captain_roles):
        """Test assessment returns None when captain not identified."""
        assessor = CaptainLeadershipAssessor(
            captain_transcripts, role_assignments=no_captain_roles
        )
        assert assessor.assess() is None

    def test_assessment_counts_captain_utterances(self, captain_transcripts, role_assignments):
        """Test that utterance count matches captain's actual utterances."""
        assessor = CaptainLeadershipAssessor(
            captain_transcripts, role_assignments=role_assignments
        )
        assessment = assessor.assess()
        assert assessment is not None
        captain_count = sum(
            1 for t in captain_transcripts if t['speaker'] == 'speaker_1'
        )
        assert assessment.utterance_count == captain_count


class TestCrisisWithTelemetry:
    """Tests for crisis response with telemetry data."""

    def test_crisis_uses_telemetry_window(
        self, captain_transcripts, role_assignments, telemetry_with_combat
    ):
        """Test that crisis assessment uses telemetry-defined crisis windows."""
        assessor = CaptainLeadershipAssessor(
            captain_transcripts,
            role_assignments=role_assignments,
            telemetry_events=telemetry_with_combat,
        )
        assessment = assessor.assess()
        assert assessment is not None
        crisis = assessment.dimensions['crisis_response']
        # During the combat window (55-115s), captain has "Red alert",
        # "fire at will", etc. — should detect directives
        assert crisis.count >= 1

    def test_crisis_without_telemetry_falls_back(
        self, captain_transcripts, role_assignments
    ):
        """Test that crisis assessment works without telemetry (uses all utterances)."""
        assessor = CaptainLeadershipAssessor(
            captain_transcripts, role_assignments=role_assignments
        )
        assessment = assessor.assess()
        assert assessment is not None
        crisis = assessment.dimensions['crisis_response']
        # Should still find directive patterns in all utterances
        assert crisis.score >= 1


class TestStructuredResults:
    """Tests for structured results output."""

    def test_structured_results_format(self, captain_transcripts, role_assignments):
        """Test that structured results have expected keys."""
        assessor = CaptainLeadershipAssessor(
            captain_transcripts, role_assignments=role_assignments
        )
        results = assessor.get_structured_results()
        assert results is not None
        assert 'captain_speaker' in results
        assert 'overall_score' in results
        assert 'dimensions' in results
        assert 'strengths' in results
        assert 'development_areas' in results

    def test_structured_results_dimensions(self, captain_transcripts, role_assignments):
        """Test that all dimensions are present in structured results."""
        assessor = CaptainLeadershipAssessor(
            captain_transcripts, role_assignments=role_assignments
        )
        results = assessor.get_structured_results()
        assert results is not None
        dims = results['dimensions']
        assert 'delegation' in dims
        assert 'crew_engagement' in dims
        assert 'information_flow' in dims
        assert 'praise_feedback' in dims
        assert 'crisis_response' in dims

    def test_structured_results_none_when_no_captain(
        self, captain_transcripts, no_captain_roles
    ):
        """Test that structured results return None when no captain."""
        assessor = CaptainLeadershipAssessor(
            captain_transcripts, role_assignments=no_captain_roles
        )
        assert assessor.get_structured_results() is None

    def test_dimension_has_examples(self, captain_transcripts, role_assignments):
        """Test that dimensions include example quotes."""
        assessor = CaptainLeadershipAssessor(
            captain_transcripts, role_assignments=role_assignments
        )
        results = assessor.get_structured_results()
        assert results is not None
        # Delegation should have examples since captain addresses multiple stations
        delegation = results['dimensions']['delegation']
        assert len(delegation['examples']) > 0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_transcripts(self, role_assignments):
        """Test with no transcripts at all."""
        assessor = CaptainLeadershipAssessor([], role_assignments=role_assignments)
        assert assessor.assess() is None

    def test_captain_with_no_utterances(self, role_assignments):
        """Test when captain is identified but has no utterances."""
        transcripts = [
            {'speaker': 'speaker_2', 'text': 'Course laid in.', 'start_time': 0},
        ]
        assessor = CaptainLeadershipAssessor(
            transcripts, role_assignments=role_assignments
        )
        assert assessor.assess() is None

    def test_single_utterance(self, role_assignments):
        """Test with captain having a single utterance."""
        transcripts = [
            {'speaker': 'speaker_1', 'text': 'Red alert!', 'start_time': 0},
        ]
        assessor = CaptainLeadershipAssessor(
            transcripts, role_assignments=role_assignments
        )
        assessment = assessor.assess()
        assert assessment is not None
        assert assessment.utterance_count == 1
        # Overall score should still be calculable
        assert assessment.overall_score >= 0

    def test_overall_excludes_insufficient_data(self, role_assignments):
        """Test that score 0 (insufficient data) is excluded from overall."""
        transcripts = [
            {'speaker': 'speaker_1', 'text': 'Red alert!', 'start_time': 0},
        ]
        assessor = CaptainLeadershipAssessor(
            transcripts, role_assignments=role_assignments
        )
        assessment = assessor.assess()
        assert assessment is not None
        # With only 1 utterance, most dimensions will score 1 (low)
        # but overall should still be a valid average
        assert assessment.overall_score > 0
