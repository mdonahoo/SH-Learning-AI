#!/usr/bin/env python3
"""
Test script for enhanced report generation with LLM.

This script tests the new enhanced analysis components:
- RoleInferenceEngine
- ConfidenceAnalyzer
- MissionPhaseAnalyzer
- QualityVerifier
- SpeakerScorecardGenerator
- CommunicationQualityAnalyzer
- EnhancedReportBuilder

It generates a comprehensive report matching the example format.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_transcripts(session_path: Path) -> list:
    """Load transcripts from session directory."""
    transcript_file = session_path / 'transcripts.json'
    if not transcript_file.exists():
        logger.error(f"Transcripts file not found: {transcript_file}")
        return []

    with open(transcript_file, 'r') as f:
        transcripts = json.load(f)

    # Normalize transcript format
    normalized = []
    for t in transcripts:
        normalized.append({
            'timestamp': t.get('timestamp', 0),
            'speaker': t.get('speaker_id') or t.get('speaker', 'unknown'),
            'text': t.get('text', ''),
            'confidence': t.get('confidence', 0),
        })

    return normalized


def load_events(session_path: Path) -> list:
    """Load game events if available."""
    events_file = session_path / 'game_events.json'
    if events_file.exists():
        with open(events_file, 'r') as f:
            return json.load(f)
    return []


def test_individual_components(transcripts: list, events: list):
    """Test each analysis component individually."""
    from src.metrics.role_inference import RoleInferenceEngine
    from src.metrics.confidence_analyzer import ConfidenceAnalyzer
    from src.metrics.phase_analyzer import MissionPhaseAnalyzer
    from src.metrics.quality_verifier import QualityVerifier
    from src.metrics.speaker_scorecard import SpeakerScorecardGenerator
    from src.metrics.communication_quality import CommunicationQualityAnalyzer

    print("\n" + "="*60)
    print("TESTING INDIVIDUAL COMPONENTS")
    print("="*60)

    # 1. Role Inference
    print("\n--- Role Inference Engine ---")
    role_engine = RoleInferenceEngine(transcripts)
    role_results = role_engine.get_structured_results()
    print(role_results['role_table'])
    print("\nMethodology excerpt:")
    print(role_results['methodology'][:500] + "...")

    # 2. Confidence Analysis
    print("\n--- Confidence Analyzer ---")
    conf_analyzer = ConfidenceAnalyzer(transcripts)
    conf_results = conf_analyzer.get_structured_results()
    print(conf_results['distribution_table'])
    print(f"\nQuality Assessment: {conf_results['quality_assessment']}")

    # 3. Mission Phase Analysis
    print("\n--- Mission Phase Analyzer ---")
    phase_analyzer = MissionPhaseAnalyzer(transcripts, events, min_phase_utterances=5)
    phase_results = phase_analyzer.get_structured_results()
    print(f"Total phases detected: {phase_results['total_phases']}")
    for phase in phase_results['phases'][:3]:
        print(f"  Phase {phase['phase_number']}: {phase['display_name']} ({phase['utterance_count']} utterances)")

    # 4. Quality Verifier
    print("\n--- Quality Verifier ---")
    verifier = QualityVerifier(transcripts, events)
    verification_results = verifier.get_structured_results()
    print(verification_results['verification_table'])

    # 5. Communication Quality
    print("\n--- Communication Quality Analyzer ---")
    comm_analyzer = CommunicationQualityAnalyzer(transcripts)
    comm_results = comm_analyzer.get_structured_results()
    print(f"Effective communications: {comm_results['statistics']['effective_count']}")
    print(f"Needs improvement: {comm_results['statistics']['improvement_count']}")

    # 6. Speaker Scorecards
    print("\n--- Speaker Scorecard Generator ---")
    role_assignments = {
        speaker: data['role']
        for speaker, data in role_results['speaker_roles'].items()
    }
    scorecard_gen = SpeakerScorecardGenerator(transcripts, role_assignments=role_assignments)
    scorecard_results = scorecard_gen.get_structured_results()
    for speaker, data in list(scorecard_results['speaker_scorecards'].items())[:3]:
        print(f"  {speaker} ({data['role']}): Overall {data['overall_score']}/5")

    print("\n" + "="*60)
    print("ALL COMPONENTS TESTED SUCCESSFULLY")
    print("="*60)


def generate_enhanced_report(transcripts: list, events: list, mission_name: str) -> str:
    """Generate enhanced report using EnhancedReportBuilder."""
    from src.metrics.enhanced_report_builder import EnhancedReportBuilder

    print("\n" + "="*60)
    print("GENERATING ENHANCED REPORT")
    print("="*60)

    builder = EnhancedReportBuilder(
        transcripts=transcripts,
        events=events,
        mission_data={'mission_name': mission_name}
    )

    # Generate full report
    report = builder.generate_full_report()

    print(f"Report generated: {len(report)} characters")

    return report


def generate_llm_report(transcripts: list, events: list, mission_name: str, use_llm: bool = True) -> str:
    """Generate report with LLM narrative enhancement."""
    from src.metrics.enhanced_report_builder import EnhancedReportBuilder
    from src.llm.hybrid_prompts import build_enhanced_report_prompt
    from src.llm.ollama_client import OllamaClient

    print("\n" + "="*60)
    print("GENERATING LLM-ENHANCED REPORT")
    print("="*60)

    # Build analysis data
    builder = EnhancedReportBuilder(
        transcripts=transcripts,
        events=events,
        mission_data={'mission_name': mission_name}
    )

    enhanced_data = builder.get_all_structured_data()

    if not use_llm:
        print("LLM disabled - returning pre-computed report")
        return enhanced_data['full_report']

    # Build prompt for LLM
    prompt = build_enhanced_report_prompt(enhanced_data, mission_name, style="professional")

    print(f"Prompt size: {len(prompt)} characters")

    # Initialize Ollama client
    client = OllamaClient(
        model=os.getenv('OLLAMA_MODEL', 'qwen2.5:14b'),
        timeout=600  # 10 minute timeout for large reports
    )

    # Check connection
    if not client.check_connection():
        print("ERROR: Cannot connect to Ollama server")
        print("Returning pre-computed report instead")
        return enhanced_data['full_report']

    print(f"Using model: {client.model}")
    print("Generating report with LLM...")

    # Generate with progress
    system_prompt = """You are a training assessment formatter for Starship Horizons bridge simulator.
You format pre-computed data into readable narratives matching professional military debrief standards.
You NEVER calculate, modify, or invent data. You ONLY format provided facts."""

    result = client.generate_with_progress(
        prompt,
        system=system_prompt,
        temperature=0.3,
        max_tokens=8192,
        transcript_size_kb=len(json.dumps(transcripts)) / 1024,
        show_progress=True
    )

    if result:
        print(f"\nLLM report generated: {len(result)} characters")
        return result
    else:
        print("LLM generation failed - returning pre-computed report")
        return enhanced_data['full_report']


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test enhanced report generation')
    parser.add_argument(
        'session_path',
        nargs='?',
        default=None,
        help='Path to session directory (default: latest in data/recordings)'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output file path for generated report'
    )
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Skip LLM generation and only test components'
    )
    parser.add_argument(
        '--components-only',
        action='store_true',
        help='Only test individual components without full report'
    )
    parser.add_argument(
        '--mission-name', '-n',
        default='Test Mission',
        help='Mission name for the report'
    )

    args = parser.parse_args()

    # Find session path
    if args.session_path:
        session_path = Path(args.session_path)
    else:
        # Find latest session
        recordings_dir = Path('data/recordings')
        if not recordings_dir.exists():
            print("ERROR: No recordings directory found")
            sys.exit(1)

        sessions = sorted(recordings_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        if not sessions:
            print("ERROR: No sessions found in data/recordings")
            sys.exit(1)

        session_path = sessions[0]
        print(f"Using latest session: {session_path.name}")

    if not session_path.exists():
        print(f"ERROR: Session path does not exist: {session_path}")
        sys.exit(1)

    # Load data
    print(f"\nLoading data from: {session_path}")
    transcripts = load_transcripts(session_path)
    events = load_events(session_path)

    print(f"Loaded {len(transcripts)} transcripts and {len(events)} events")

    if not transcripts:
        print("ERROR: No transcripts found")
        sys.exit(1)

    # Test components
    test_individual_components(transcripts, events)

    if args.components_only:
        print("\nComponents-only mode - skipping full report generation")
        return

    # Generate report
    if args.no_llm:
        report = generate_enhanced_report(transcripts, events, args.mission_name)
    else:
        report = generate_llm_report(transcripts, events, args.mission_name, use_llm=True)

    # Output report
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = session_path / f'enhanced_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.md'

    with open(output_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {output_path}")

    # Print preview
    print("\n" + "="*60)
    print("REPORT PREVIEW (first 2000 chars)")
    print("="*60)
    print(report[:2000])
    if len(report) > 2000:
        print(f"\n... ({len(report) - 2000} more characters)")


if __name__ == "__main__":
    main()
