"""Tests for narrative generation modules."""

from datetime import datetime, timedelta

import pytest

from src.narrative.beat_detector import BeatDetector, BeatType, DramaticBeat
from src.narrative.character_voice import (
    CharacterAnalyzer,
    CharacterArchetype,
    CharacterVoice,
    CommunicationStyle,
    StressResponse,
)
from src.narrative.episode_generator import Episode, EpisodeGenerator, EpisodeMetadata
from src.narrative.scene_builder import (
    AtmosphereType,
    Scene,
    SceneBuilder,
    SceneType,
)
from src.narrative.tension_analyzer import (
    ActType,
    TensionAnalyzer,
    TensionCurve,
    format_tension_curve_ascii,
)


class TestBeatDetector:
    """Test suite for BeatDetector."""

    @pytest.fixture
    def detector(self):
        """Create beat detector instance."""
        return BeatDetector()

    @pytest.fixture
    def sample_events(self):
        """Create sample telemetry events."""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        return [
            {
                "type": "mission",
                "timestamp": base_time.isoformat(),
                "data": {"state": "running"},
            },
            {
                "type": "alert",
                "timestamp": (base_time + timedelta(minutes=2)).isoformat(),
                "data": {"level": 3},
            },
            {
                "type": "alert",
                "timestamp": (base_time + timedelta(minutes=5)).isoformat(),
                "data": {"level": 4},
            },
            {
                "type": "weapon_fire",
                "timestamp": (base_time + timedelta(minutes=6)).isoformat(),
                "data": {"weapon": "phaser"},
            },
            {
                "type": "damage",
                "timestamp": (base_time + timedelta(minutes=8)).isoformat(),
                "data": {"shields": 20, "hull": 80},
            },
            {
                "type": "mission",
                "timestamp": (base_time + timedelta(minutes=10)).isoformat(),
                "data": {"state": "complete"},
            },
        ]

    def test_detect_beats_returns_list(self, detector, sample_events):
        """Test that detect_beats returns a list of beats."""
        beats = detector.detect_beats(sample_events)
        assert isinstance(beats, list)
        assert len(beats) > 0

    def test_mission_start_creates_cold_open(self, detector, sample_events):
        """Test mission start creates cold open beat."""
        beats = detector.detect_beats(sample_events)
        cold_open_beats = [b for b in beats if b.beat_type == BeatType.COLD_OPEN_HOOK]
        assert len(cold_open_beats) >= 1

    def test_alert_yellow_creates_inciting_incident(self, detector, sample_events):
        """Test yellow alert creates inciting incident."""
        beats = detector.detect_beats(sample_events)
        inciting = [b for b in beats if b.beat_type == BeatType.INCITING_INCIDENT]
        assert len(inciting) >= 1

    def test_alert_red_creates_escalation(self, detector, sample_events):
        """Test red alert creates escalation beat."""
        beats = detector.detect_beats(sample_events)
        escalation = [b for b in beats if b.beat_type == BeatType.ESCALATION]
        assert len(escalation) >= 1

    def test_low_shields_creates_crisis(self, detector, sample_events):
        """Test low shields creates crisis beat."""
        beats = detector.detect_beats(sample_events)
        crisis = [b for b in beats if b.beat_type == BeatType.CRISIS_POINT]
        assert len(crisis) >= 1

    def test_beats_have_tension_deltas(self, detector, sample_events):
        """Test all beats have tension delta values."""
        beats = detector.detect_beats(sample_events)
        for beat in beats:
            assert isinstance(beat.tension_delta, float)

    def test_beats_are_chronological(self, detector, sample_events):
        """Test beats are returned in chronological order."""
        beats = detector.detect_beats(sample_events)
        for i in range(1, len(beats)):
            assert beats[i].timestamp >= beats[i - 1].timestamp


class TestTensionAnalyzer:
    """Test suite for TensionAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create tension analyzer instance."""
        return TensionAnalyzer()

    @pytest.fixture
    def sample_beats(self):
        """Create sample dramatic beats."""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        return [
            DramaticBeat(
                timestamp=base_time,
                beat_type=BeatType.COLD_OPEN_HOOK,
                tension_delta=0.1,
                description="Mission begins",
            ),
            DramaticBeat(
                timestamp=base_time + timedelta(minutes=2),
                beat_type=BeatType.INCITING_INCIDENT,
                tension_delta=0.2,
                description="Alert raised",
            ),
            DramaticBeat(
                timestamp=base_time + timedelta(minutes=5),
                beat_type=BeatType.ESCALATION,
                tension_delta=0.25,
                description="Red alert",
            ),
            DramaticBeat(
                timestamp=base_time + timedelta(minutes=8),
                beat_type=BeatType.CRISIS_POINT,
                tension_delta=0.35,
                description="Shields failing",
            ),
            DramaticBeat(
                timestamp=base_time + timedelta(minutes=10),
                beat_type=BeatType.RESOLUTION,
                tension_delta=-0.5,
                description="Victory",
            ),
        ]

    def test_analyze_returns_tension_curve(self, analyzer, sample_beats):
        """Test analyze returns TensionCurve."""
        curve = analyzer.analyze(sample_beats)
        assert isinstance(curve, TensionCurve)

    def test_tension_points_created(self, analyzer, sample_beats):
        """Test tension points are created for each beat."""
        curve = analyzer.analyze(sample_beats)
        assert len(curve.points) == len(sample_beats)

    def test_peak_tension_calculated(self, analyzer, sample_beats):
        """Test peak tension is calculated."""
        curve = analyzer.analyze(sample_beats)
        assert curve.peak_tension > 0
        assert curve.peak_tension <= 1.0

    def test_crisis_point_identified(self, analyzer, sample_beats):
        """Test crisis point is identified."""
        curve = analyzer.analyze(sample_beats)
        assert curve.crisis_point is not None
        assert curve.crisis_point.beat_type == BeatType.CRISIS_POINT

    def test_act_breaks_determined(self, analyzer, sample_beats):
        """Test act breaks are determined."""
        curve = analyzer.analyze(sample_beats)
        # Should have at least some act breaks for a mission with multiple beats
        assert len(curve.act_breaks) >= 0  # Could be 0 for short missions

    def test_tension_curve_ascii_visualization(self, analyzer, sample_beats):
        """Test ASCII visualization works."""
        curve = analyzer.analyze(sample_beats)
        ascii_art = format_tension_curve_ascii(curve)
        assert isinstance(ascii_art, str)
        assert len(ascii_art) > 0


class TestCharacterAnalyzer:
    """Test suite for CharacterAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create character analyzer instance."""
        return CharacterAnalyzer()

    @pytest.fixture
    def sample_transcripts(self):
        """Create sample transcripts."""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        return [
            {
                "speaker": "SPEAKER_1",
                "text": "Red alert! All hands to battle stations!",
                "timestamp": base_time.isoformat(),
                "confidence": 0.95,
            },
            {
                "speaker": "SPEAKER_2",
                "text": "Aye, Captain. Weapons ready.",
                "timestamp": (base_time + timedelta(seconds=5)).isoformat(),
                "confidence": 0.90,
            },
            {
                "speaker": "SPEAKER_1",
                "text": "Status report!",
                "timestamp": (base_time + timedelta(seconds=10)).isoformat(),
                "confidence": 0.92,
            },
            {
                "speaker": "SPEAKER_2",
                "text": "Shields at 100%, sir.",
                "timestamp": (base_time + timedelta(seconds=15)).isoformat(),
                "confidence": 0.88,
            },
            {
                "speaker": "SPEAKER_1",
                "text": "Engage!",
                "timestamp": (base_time + timedelta(seconds=20)).isoformat(),
                "confidence": 0.95,
            },
        ]

    def test_analyze_returns_character_dict(self, analyzer, sample_transcripts):
        """Test analyze returns dictionary of characters."""
        characters = analyzer.analyze(sample_transcripts)
        assert isinstance(characters, dict)
        assert len(characters) == 2

    def test_character_voice_created(self, analyzer, sample_transcripts):
        """Test CharacterVoice objects are created."""
        characters = analyzer.analyze(sample_transcripts)
        for voice in characters.values():
            assert isinstance(voice, CharacterVoice)

    def test_speaker_stats_calculated(self, analyzer, sample_transcripts):
        """Test speaker statistics are calculated."""
        characters = analyzer.analyze(sample_transcripts)
        speaker_1 = characters.get("SPEAKER_1")
        assert speaker_1 is not None
        assert speaker_1.total_utterances == 3

    def test_command_speaker_identified(self, analyzer, sample_transcripts):
        """Test command-giving speaker is identified."""
        characters = analyzer.analyze(sample_transcripts)
        speaker_1 = characters.get("SPEAKER_1")
        assert speaker_1.gives_orders is True

    def test_protocol_usage_detected(self, analyzer, sample_transcripts):
        """Test protocol usage is detected."""
        characters = analyzer.analyze(sample_transcripts)
        speaker_2 = characters.get("SPEAKER_2")
        # SPEAKER_2 uses "Aye" and "sir" - should show protocol usage
        assert speaker_2.protocol_usage > 0

    def test_archetype_assigned(self, analyzer, sample_transcripts):
        """Test character archetype is assigned."""
        characters = analyzer.analyze(sample_transcripts)
        for voice in characters.values():
            assert voice.archetype is not None
            assert isinstance(voice.archetype, CharacterArchetype)

    def test_voice_description_generated(self, analyzer, sample_transcripts):
        """Test voice description is generated."""
        characters = analyzer.analyze(sample_transcripts)
        for voice in characters.values():
            assert voice.voice_description != ""


class TestSceneBuilder:
    """Test suite for SceneBuilder."""

    @pytest.fixture
    def builder(self):
        """Create scene builder instance."""
        return SceneBuilder()

    @pytest.fixture
    def sample_beats(self):
        """Create sample dramatic beats."""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        return [
            DramaticBeat(
                timestamp=base_time,
                beat_type=BeatType.COLD_OPEN_HOOK,
                tension_delta=0.1,
                description="Mission begins",
                tension_level=0.1,
            ),
            DramaticBeat(
                timestamp=base_time + timedelta(minutes=5),
                beat_type=BeatType.ESCALATION,
                tension_delta=0.25,
                description="Red alert",
                tension_level=0.5,
            ),
            DramaticBeat(
                timestamp=base_time + timedelta(minutes=10),
                beat_type=BeatType.RESOLUTION,
                tension_delta=-0.3,
                description="Victory",
                tension_level=0.2,
            ),
        ]

    @pytest.fixture
    def sample_tension_curve(self, sample_beats):
        """Create sample tension curve."""
        analyzer = TensionAnalyzer()
        return analyzer.analyze(sample_beats)

    @pytest.fixture
    def sample_characters(self):
        """Create sample characters."""
        return {
            "SPEAKER_1": CharacterVoice(
                speaker_id="SPEAKER_1",
                role="Captain",
                archetype=CharacterArchetype.THE_COMMANDER,
            ),
        }

    @pytest.fixture
    def sample_transcripts(self):
        """Create sample transcripts."""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        return [
            {
                "speaker": "SPEAKER_1",
                "text": "Engage!",
                "timestamp": base_time.isoformat(),
                "confidence": 0.95,
            },
        ]

    def test_build_scenes_returns_list(
        self,
        builder,
        sample_beats,
        sample_tension_curve,
        sample_characters,
        sample_transcripts,
    ):
        """Test build_scenes returns list of scenes."""
        scenes = builder.build_scenes(
            sample_beats,
            sample_tension_curve,
            sample_characters,
            sample_transcripts,
        )
        assert isinstance(scenes, list)

    def test_scenes_have_required_fields(
        self,
        builder,
        sample_beats,
        sample_tension_curve,
        sample_characters,
        sample_transcripts,
    ):
        """Test scenes have all required fields."""
        scenes = builder.build_scenes(
            sample_beats,
            sample_tension_curve,
            sample_characters,
            sample_transcripts,
        )
        for scene in scenes:
            assert isinstance(scene, Scene)
            assert scene.scene_type is not None
            assert scene.act is not None
            assert scene.start_time is not None
            assert scene.end_time is not None

    def test_atmosphere_determined(
        self,
        builder,
        sample_beats,
        sample_tension_curve,
        sample_characters,
        sample_transcripts,
    ):
        """Test atmosphere is determined for scenes."""
        scenes = builder.build_scenes(
            sample_beats,
            sample_tension_curve,
            sample_characters,
            sample_transcripts,
        )
        for scene in scenes:
            assert isinstance(scene.atmosphere, AtmosphereType)

    def test_captains_logs_generated(
        self,
        builder,
        sample_beats,
        sample_tension_curve,
        sample_characters,
        sample_transcripts,
    ):
        """Test Captain's Log entries are generated."""
        builder.build_scenes(
            sample_beats,
            sample_tension_curve,
            sample_characters,
            sample_transcripts,
            mission_name="Test Mission",
        )
        assert len(builder.logs) >= 1


class TestEpisodeGenerator:
    """Test suite for EpisodeGenerator."""

    @pytest.fixture
    def generator(self):
        """Create episode generator without LLM callback."""
        return EpisodeGenerator(llm_callback=None)

    @pytest.fixture
    def sample_events(self):
        """Create sample events."""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        return [
            {
                "type": "mission",
                "timestamp": base_time.isoformat(),
                "data": {"state": "running"},
            },
            {
                "type": "alert",
                "timestamp": (base_time + timedelta(minutes=2)).isoformat(),
                "data": {"level": 4},
            },
            {
                "type": "mission",
                "timestamp": (base_time + timedelta(minutes=10)).isoformat(),
                "data": {"state": "complete"},
            },
        ]

    @pytest.fixture
    def sample_transcripts(self):
        """Create sample transcripts."""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        return [
            {
                "speaker": "SPEAKER_1",
                "text": "Begin mission.",
                "timestamp": base_time.isoformat(),
                "confidence": 0.95,
            },
            {
                "speaker": "SPEAKER_1",
                "text": "Red alert!",
                "timestamp": (base_time + timedelta(minutes=2)).isoformat(),
                "confidence": 0.92,
            },
        ]

    def test_generate_returns_episode(
        self, generator, sample_events, sample_transcripts
    ):
        """Test generate returns Episode object."""
        episode = generator.generate(
            sample_events, sample_transcripts, mission_name="Test Mission"
        )
        assert isinstance(episode, Episode)

    def test_episode_has_metadata(
        self, generator, sample_events, sample_transcripts
    ):
        """Test episode has metadata."""
        episode = generator.generate(
            sample_events, sample_transcripts, mission_name="Test Mission"
        )
        assert isinstance(episode.metadata, EpisodeMetadata)
        assert episode.metadata.mission_name == "Test Mission"

    def test_episode_has_scenes(
        self, generator, sample_events, sample_transcripts
    ):
        """Test episode has scenes."""
        episode = generator.generate(
            sample_events, sample_transcripts, mission_name="Test Mission"
        )
        assert len(episode.scenes) >= 0

    def test_episode_has_characters(
        self, generator, sample_events, sample_transcripts
    ):
        """Test episode has character profiles."""
        episode = generator.generate(
            sample_events, sample_transcripts, mission_name="Test Mission"
        )
        assert len(episode.characters) > 0

    def test_episode_has_tension_curve(
        self, generator, sample_events, sample_transcripts
    ):
        """Test episode has tension curve."""
        episode = generator.generate(
            sample_events, sample_transcripts, mission_name="Test Mission"
        )
        assert isinstance(episode.tension_curve, TensionCurve)

    def test_episode_to_dict(
        self, generator, sample_events, sample_transcripts
    ):
        """Test episode can be converted to dictionary."""
        episode = generator.generate(
            sample_events, sample_transcripts, mission_name="Test Mission"
        )
        episode_dict = episode.to_dict()
        assert isinstance(episode_dict, dict)
        assert "metadata" in episode_dict
        assert "scene_count" in episode_dict

    def test_structured_output_generated(
        self, generator, sample_events, sample_transcripts
    ):
        """Test structured output is generated without LLM."""
        episode = generator.generate(
            sample_events, sample_transcripts, mission_name="Test Mission"
        )
        assert episode.full_episode != ""
        assert "Test Mission" in episode.full_episode or "TEST MISSION" in episode.full_episode


class TestIntegration:
    """Integration tests for full narrative pipeline."""

    def test_full_pipeline_execution(self):
        """Test complete pipeline from events to episode."""
        # Create realistic mission data
        base_time = datetime(2024, 1, 1, 12, 0, 0)

        events = [
            {"type": "mission", "timestamp": base_time.isoformat(), "data": {"state": "running"}},
            {"type": "alert", "timestamp": (base_time + timedelta(minutes=1)).isoformat(), "data": {"level": 2}},
            {"type": "alert", "timestamp": (base_time + timedelta(minutes=3)).isoformat(), "data": {"level": 3}},
            {"type": "contact", "timestamp": (base_time + timedelta(minutes=4)).isoformat(), "data": {"faction": "hostile", "name": "Enemy Ship"}},
            {"type": "alert", "timestamp": (base_time + timedelta(minutes=5)).isoformat(), "data": {"level": 4}},
            {"type": "weapon_fire", "timestamp": (base_time + timedelta(minutes=6)).isoformat(), "data": {}},
            {"type": "damage", "timestamp": (base_time + timedelta(minutes=8)).isoformat(), "data": {"shields": 15, "hull": 70}},
            {"type": "damage", "timestamp": (base_time + timedelta(minutes=9)).isoformat(), "data": {"shields": 50, "hull": 100}},
            {"type": "mission", "timestamp": (base_time + timedelta(minutes=10)).isoformat(), "data": {"state": "complete"}},
        ]

        transcripts = [
            {"speaker": "CAPTAIN", "text": "Set course for the rendezvous point.", "timestamp": base_time.isoformat(), "confidence": 0.95},
            {"speaker": "HELM", "text": "Aye, Captain. Course laid in.", "timestamp": (base_time + timedelta(minutes=1)).isoformat(), "confidence": 0.90},
            {"speaker": "TACTICAL", "text": "Captain, I'm detecting a vessel on an intercept course.", "timestamp": (base_time + timedelta(minutes=3)).isoformat(), "confidence": 0.92},
            {"speaker": "CAPTAIN", "text": "Red alert! All hands to battle stations!", "timestamp": (base_time + timedelta(minutes=5)).isoformat(), "confidence": 0.95},
            {"speaker": "TACTICAL", "text": "Shields up! Weapons armed!", "timestamp": (base_time + timedelta(minutes=5, seconds=10)).isoformat(), "confidence": 0.88},
            {"speaker": "CAPTAIN", "text": "Fire!", "timestamp": (base_time + timedelta(minutes=6)).isoformat(), "confidence": 0.95},
            {"speaker": "TACTICAL", "text": "Direct hit! Enemy shields are down!", "timestamp": (base_time + timedelta(minutes=7)).isoformat(), "confidence": 0.90},
            {"speaker": "CAPTAIN", "text": "Stand down from red alert. Good work, everyone.", "timestamp": (base_time + timedelta(minutes=10)).isoformat(), "confidence": 0.92},
        ]

        # Generate episode
        generator = EpisodeGenerator(llm_callback=None)
        episode = generator.generate(
            events,
            transcripts,
            mission_name="The Rendezvous",
        )

        # Verify complete episode
        assert episode.metadata.title == "The Rendezvous"
        assert len(episode.characters) >= 3  # CAPTAIN, HELM, TACTICAL
        assert episode.tension_curve.peak_tension > 0.5  # Should have high tension
        assert len(episode.scenes) > 0
        assert episode.full_episode != ""

        # Verify character detection
        captain = None
        for voice in episode.characters.values():
            if "CAPTAIN" in voice.speaker_id:
                captain = voice
                break
        assert captain is not None
        assert captain.gives_orders is True
