#!/usr/bin/env python3
"""
Learning evaluation framework for Starship Horizons missions.

Implements industry-standard learning and teamwork assessment frameworks:
- Kirkpatrick's Training Evaluation Model (4 levels)
- Bloom's Taxonomy (cognitive learning levels)
- NASA Teamwork Framework (team performance dimensions)
- Starship Horizons mission-specific metrics
"""

import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict, Counter
from datetime import datetime

logger = logging.getLogger(__name__)


class LearningEvaluator:
    """Evaluates mission performance using established learning frameworks."""

    def __init__(
        self,
        events: List[Dict[str, Any]],
        transcripts: List[Dict[str, Any]],
        speech_action_data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize evaluator with mission data.

        Args:
            events: List of mission events
            transcripts: List of crew communications
            speech_action_data: Optional cross-reference data from TelemetryAudioCorrelator
        """
        self.events = events
        self.transcripts = transcripts
        self.speech_action_data = speech_action_data

    def evaluate_all(self) -> Dict[str, Any]:
        """
        Run all evaluation frameworks.

        Returns:
            Complete evaluation results
        """
        results = {
            'kirkpatrick': self.evaluate_kirkpatrick(),
            'blooms_taxonomy': self.evaluate_blooms_taxonomy(),
            'nasa_teamwork': self.evaluate_nasa_teamwork(),
            'mission_specific': self.evaluate_mission_metrics(),
        }

        # Include speech-action alignment if telemetry data is available
        if self.speech_action_data:
            results['speech_action_alignment'] = self._evaluate_speech_action_alignment()

        return results

    def evaluate_kirkpatrick(self) -> Dict[str, Any]:
        """
        Evaluate using Kirkpatrick's 4-Level Training Evaluation Model.

        Level 1 - Reaction: How learners feel about training
        Level 2 - Learning: Knowledge/skills acquired
        Level 3 - Behavior: Application of learning
        Level 4 - Results: Impact on organizational goals

        Returns:
            Kirkpatrick evaluation results
        """
        evaluation = {
            'level_1_reaction': self._evaluate_reaction(),
            'level_2_learning': self._evaluate_learning(),
            'level_3_behavior': self._evaluate_behavior(),
            'level_4_results': self._evaluate_results()
        }
        return evaluation

    def _evaluate_reaction(self) -> Dict[str, Any]:
        """Level 1: Engagement and participation."""
        total_comms = len(self.transcripts)
        speaker_counts = Counter(t['speaker'] for t in self.transcripts)
        unique_speakers = len(speaker_counts)

        # Communication distribution (equity of participation)
        if unique_speakers > 0:
            avg_comms = total_comms / unique_speakers
            std_dev = sum((count - avg_comms) ** 2 for count in speaker_counts.values()) ** 0.5
            participation_equity = max(0, 100 - (std_dev / avg_comms * 100)) if avg_comms > 0 else 0
        else:
            participation_equity = 0

        # Average confidence (proxy for clarity/comfort)
        avg_confidence = sum(t.get('confidence', 0) for t in self.transcripts) / total_comms if total_comms > 0 else 0

        return {
            'total_communications': total_comms,
            'unique_speakers': unique_speakers,
            'participation_equity_score': round(participation_equity, 2),
            'avg_transcription_confidence': round(avg_confidence, 3),
            'engagement_level': 'high' if total_comms > 50 else 'moderate' if total_comms > 20 else 'low',
            'interpretation': f"{'High' if participation_equity > 70 else 'Moderate' if participation_equity > 40 else 'Low'} participation equity across {unique_speakers} speakers"
        }

    def _evaluate_learning(self) -> Dict[str, Any]:
        """Level 2: Knowledge and skills demonstrated."""
        # Extract objective completion
        objectives = self._extract_objectives()
        total_objectives = len(objectives)
        completed_objectives = sum(1 for obj in objectives.values() if obj.get('complete', False))

        # Protocol adherence (based on structured communications)
        protocol_keywords = ['captain', 'aye', 'affirmative', 'negative', 'sir', 'ma\'am', 'reporting']
        protocol_comms = sum(1 for t in self.transcripts
                           if any(kw in t.get('text', '').lower() for kw in protocol_keywords))
        protocol_adherence = (protocol_comms / len(self.transcripts) * 100) if self.transcripts else 0

        return {
            'total_objectives': total_objectives,
            'completed_objectives': completed_objectives,
            'objective_completion_rate': round(completed_objectives / total_objectives * 100, 1) if total_objectives > 0 else 0,
            'protocol_adherence_score': round(protocol_adherence, 1),
            'knowledge_level': 'advanced' if protocol_adherence > 30 else 'intermediate' if protocol_adherence > 10 else 'novice',
            'interpretation': f"Completed {completed_objectives}/{total_objectives} objectives with {protocol_adherence:.0f}% protocol adherence"
        }

    def _evaluate_behavior(self) -> Dict[str, Any]:
        """Level 3: Application of training in real scenarios."""
        # Response times between speakers (coordination)
        response_times = []
        for i in range(1, len(self.transcripts)):
            prev_t = self.transcripts[i-1]
            curr_t = self.transcripts[i]

            # Only count if different speakers (actual response, not monologue)
            if prev_t['speaker'] != curr_t['speaker']:
                try:
                    prev_ts = prev_t.get('timestamp', prev_t.get('start_time'))
                    curr_ts = curr_t.get('timestamp', curr_t.get('start_time'))
                    # Handle float/int timestamps (seconds from session start)
                    if isinstance(prev_ts, (int, float)) and isinstance(curr_ts, (int, float)):
                        gap = curr_ts - prev_ts
                    elif isinstance(prev_ts, str) and isinstance(curr_ts, str):
                        gap = (datetime.fromisoformat(prev_ts) - datetime.fromisoformat(curr_ts)).total_seconds()
                        gap = abs(gap)
                    else:
                        continue
                    if 0 < gap < 30:  # Reasonable response time
                        response_times.append(gap)
                except (ValueError, TypeError):
                    pass

        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        # Decision-making events
        decision_keywords = ['engage', 'proceed', 'execute', 'confirm', 'negative', 'abort']
        decision_comms = sum(1 for t in self.transcripts
                           if any(kw in t.get('text', '').lower() for kw in decision_keywords))

        return {
            'avg_response_time_seconds': round(avg_response_time, 2),
            'decision_communications': decision_comms,
            'coordination_score': round(min(100, (1 / avg_response_time * 10)) if avg_response_time > 0 else 0, 1),
            'behavior_quality': 'excellent' if avg_response_time < 5 else 'good' if avg_response_time < 10 else 'needs_improvement',
            'interpretation': f"Team responded in avg {avg_response_time:.1f}s with {decision_comms} decision-making communications"
        }

    def _evaluate_results(self) -> Dict[str, Any]:
        """Level 4: Mission success and organizational impact."""
        # Mission completion
        objectives = self._extract_objectives()
        completion_rate = sum(1 for obj in objectives.values() if obj.get('complete', False)) / len(objectives) * 100 if objectives else 0

        # Mission grade (if available)
        mission_grade = None
        for event in self.events:
            if event.get('event_type') == 'mission_update':
                grade = event.get('data', {}).get('Grade')
                if grade is not None:
                    mission_grade = grade * 100
                    break

        # Critical failures
        failure_events = [e for e in self.events if 'fail' in str(e.get('event_type', '')).lower() or
                         'destroy' in str(e.get('event_type', '')).lower()]

        return {
            'mission_completion_rate': round(completion_rate, 1),
            'mission_grade': round(mission_grade, 1) if mission_grade is not None else None,
            'critical_failures': len(failure_events),
            'success_level': 'high' if completion_rate > 80 else 'moderate' if completion_rate > 50 else 'low',
            'organizational_impact': 'Training objectives achieved' if completion_rate > 70 else 'Partial training success',
            'interpretation': f"{'Success' if completion_rate > 70 else 'Partial completion'}: {completion_rate:.0f}% objectives completed"
        }

    def _evaluate_speech_action_alignment(self) -> Dict[str, Any]:
        """
        Evaluate alignment between crew speech and game actions.

        Uses cross-reference data from TelemetryAudioCorrelator to assess
        whether crew followed through on stated intentions.

        Returns:
            Speech-action alignment evaluation
        """
        if not self.speech_action_data:
            return {
                'alignment_score': 0.0,
                'assessment': 'No telemetry data available for speech-action analysis',
                'aligned_count': 0,
                'speech_only_count': 0,
                'action_only_count': 0,
                'examples': [],
            }

        alignment_score = self.speech_action_data.get('alignment_score', 0.0)
        aligned = self.speech_action_data.get('aligned', [])
        speech_only = self.speech_action_data.get('speech_only', [])
        action_only = self.speech_action_data.get('action_only', [])

        # Build assessment
        if alignment_score >= 0.7:
            assessment = (
                "Strong speech-action alignment. The crew consistently followed through "
                "on stated intentions, demonstrating effective communication and execution."
            )
            quality = 'excellent'
        elif alignment_score >= 0.4:
            assessment = (
                "Moderate speech-action alignment. The crew followed through on many "
                "stated intentions, but some actions occurred without verbal coordination "
                "and some stated plans were not executed."
            )
            quality = 'good'
        elif alignment_score > 0:
            assessment = (
                "Weak speech-action alignment. Many crew statements were not followed "
                "by corresponding game actions, or significant actions occurred without "
                "verbal coordination. This suggests a gap between communication and execution."
            )
            quality = 'needs_improvement'
        else:
            assessment = (
                "No meaningful speech-action alignment detected. This may indicate "
                "the crew operated silently or the telemetry capture was limited."
            )
            quality = 'insufficient_data'

        # Build examples for narrative
        examples = []
        for item in aligned[:5]:
            examples.append({
                'type': 'aligned',
                'description': (
                    f"{item['speaker']} said \"{item['speech'][:80]}\" and "
                    f"the action '{item['action']}' followed {item['time_delta']}s later"
                ),
            })
        for item in speech_only[:3]:
            examples.append({
                'type': 'speech_only',
                'description': (
                    f"{item['speaker']} said \"{item['text'][:80]}\" "
                    f"but no matching game action was detected"
                ),
            })
        for item in action_only[:3]:
            examples.append({
                'type': 'action_only',
                'description': (
                    f"Game action '{item['description'][:80]}' occurred "
                    f"without verbal coordination"
                ),
            })

        return {
            'alignment_score': alignment_score,
            'alignment_percentage': round(alignment_score * 100, 1),
            'quality': quality,
            'assessment': assessment,
            'aligned_count': len(aligned),
            'speech_only_count': len(speech_only),
            'action_only_count': len(action_only),
            'total_speech_intentions': self.speech_action_data.get('total_speech_intentions', 0),
            'total_game_actions': self.speech_action_data.get('total_game_actions', 0),
            'examples': examples,
        }

    def evaluate_blooms_taxonomy(self) -> Dict[str, Any]:
        """
        Evaluate cognitive learning levels using Bloom's Taxonomy.

        Levels (lowest to highest):
        1. Remember: Recall facts
        2. Understand: Explain concepts
        3. Apply: Use knowledge in new situations
        4. Analyze: Draw connections
        5. Evaluate: Justify decisions
        6. Create: Produce new work

        Returns:
            Bloom's taxonomy assessment
        """
        # Keyword mapping to cognitive levels
        keywords_by_level = {
            'remember': ['what', 'where', 'status', 'report', 'confirm'],
            'understand': ['why', 'how', 'because', 'explain', 'means'],
            'apply': ['execute', 'implement', 'use', 'demonstrate', 'operate'],
            'analyze': ['compare', 'examine', 'investigate', 'scan', 'analyze'],
            'evaluate': ['decide', 'recommend', 'assess', 'priority', 'critical'],
            'create': ['design', 'plan', 'strategy', 'develop', 'construct']
        }

        level_counts = defaultdict(int)

        for transcript in self.transcripts:
            text = transcript.get('text', '').lower()
            for level, keywords in keywords_by_level.items():
                if any(kw in text for kw in keywords):
                    level_counts[level] += 1

        total_categorized = sum(level_counts.values())

        # Determine highest demonstrated level
        highest_level = 'remember'  # default
        if level_counts['create'] > 0:
            highest_level = 'create'
        elif level_counts['evaluate'] > 0:
            highest_level = 'evaluate'
        elif level_counts['analyze'] > 0:
            highest_level = 'analyze'
        elif level_counts['apply'] > 0:
            highest_level = 'apply'
        elif level_counts['understand'] > 0:
            highest_level = 'understand'

        return {
            'cognitive_levels': dict(level_counts),
            'total_cognitive_indicators': total_categorized,
            'highest_level_demonstrated': highest_level,
            'distribution_percentage': {level: round(count / total_categorized * 100, 1)
                                      for level, count in level_counts.items()} if total_categorized > 0 else {},
            'interpretation': f"Crew demonstrated cognitive skills up to '{highest_level}' level with {total_categorized} indicators"
        }

    def evaluate_nasa_teamwork(self) -> Dict[str, Any]:
        """
        Evaluate using NASA's Teamwork Framework dimensions.

        Dimensions:
        1. Communication
        2. Coordination
        3. Leadership
        4. Monitoring & Situational Awareness
        5. Adaptability

        Returns:
            NASA teamwork evaluation
        """
        speaker_counts = Counter(t['speaker'] for t in self.transcripts)
        total_comms = len(self.transcripts)

        # 1. Communication (frequency, clarity, completeness)
        avg_confidence = sum(t.get('confidence', 0) for t in self.transcripts) / total_comms if total_comms > 0 else 0
        communication_score = round(avg_confidence * 100, 1)

        # 2. Coordination (turn-taking, response patterns)
        speaker_switches = sum(1 for i in range(1, len(self.transcripts))
                             if self.transcripts[i]['speaker'] != self.transcripts[i-1]['speaker'])
        coordination_score = round(min(100, speaker_switches / total_comms * 200), 1) if total_comms > 0 else 0

        # 3. Leadership (one speaker significantly more active)
        if speaker_counts:
            max_speaker_comms = max(speaker_counts.values())
            leadership_clarity = (max_speaker_comms / total_comms * 100) if total_comms > 0 else 0
            leadership_score = round(min(100, leadership_clarity * 2), 1)  # 50% = strong leader
        else:
            leadership_score = 0

        # 4. Monitoring (status updates, reporting)
        monitoring_keywords = ['status', 'report', 'scan', 'check', 'monitor', 'systems', 'nominal']
        monitoring_comms = sum(1 for t in self.transcripts
                             if any(kw in t.get('text', '').lower() for kw in monitoring_keywords))
        monitoring_score = round(min(100, monitoring_comms / total_comms * 300), 1) if total_comms > 0 else 0

        # 5. Adaptability (problem-solving, decision changes)
        adapt_keywords = ['problem', 'issue', 'change', 'adjust', 'adapt', 'alternative', 'instead']
        adapt_comms = sum(1 for t in self.transcripts
                        if any(kw in t.get('text', '').lower() for kw in adapt_keywords))
        adaptability_score = round(min(100, adapt_comms / total_comms * 500), 1) if total_comms > 0 else 0

        # Overall teamwork score
        overall_score = (communication_score + coordination_score + leadership_score +
                        monitoring_score + adaptability_score) / 5

        return {
            'communication': {
                'score': communication_score,
                'clarity_avg': round(avg_confidence, 3),
                'assessment': 'excellent' if communication_score > 80 else 'good' if communication_score > 60 else 'needs_improvement'
            },
            'coordination': {
                'score': coordination_score,
                'speaker_switches': speaker_switches,
                'assessment': 'excellent' if coordination_score > 70 else 'good' if coordination_score > 50 else 'needs_improvement'
            },
            'leadership': {
                'score': leadership_score,
                'primary_speaker_percentage': round(max(speaker_counts.values()) / total_comms * 100, 1) if total_comms > 0 else 0,
                'assessment': 'clear' if 30 < leadership_score < 70 else 'dominant' if leadership_score >= 70 else 'distributed'
            },
            'monitoring': {
                'score': monitoring_score,
                'status_communications': monitoring_comms,
                'assessment': 'excellent' if monitoring_score > 60 else 'good' if monitoring_score > 30 else 'needs_improvement'
            },
            'adaptability': {
                'score': adaptability_score,
                'adaptation_communications': adapt_comms,
                'assessment': 'excellent' if adaptability_score > 40 else 'good' if adaptability_score > 20 else 'limited'
            },
            'overall_teamwork_score': round(overall_score, 1),
            'interpretation': f"Overall teamwork rated {overall_score:.0f}/100 across NASA's 5 dimensions"
        }

    def evaluate_mission_metrics(self) -> Dict[str, Any]:
        """
        Starship Horizons mission-specific metrics.

        Returns:
            Mission-specific evaluation
        """
        # Calculate mission duration
        if self.events:
            timestamps = [datetime.fromisoformat(e['timestamp']) for e in self.events
                         if isinstance(e.get('timestamp'), str)]
            if timestamps:
                duration = max(timestamps) - min(timestamps)
                duration_str = str(duration).split('.')[0]  # Remove microseconds
            else:
                duration_str = "Unknown"
        else:
            duration_str = "Unknown"

        # Extract objectives
        objectives = self._extract_objectives()

        # Count event types
        event_types = Counter(e.get('event_type', 'unknown') for e in self.events)

        # Speaker statistics
        speaker_counts = Counter(t['speaker'] for t in self.transcripts)

        return {
            'mission_duration': duration_str,
            'total_events': len(self.events),
            'total_communications': len(self.transcripts),
            'unique_speakers': len(speaker_counts),
            'speaker_distribution': dict(speaker_counts),
            'objectives': {
                'total': len(objectives),
                'completed': sum(1 for obj in objectives.values() if obj.get('complete', False)),
                'details': objectives
            },
            'event_distribution': dict(event_types.most_common(10)),
            'communications_per_minute': round(len(self.transcripts) / (duration.total_seconds() / 60), 2) if 'duration' in locals() and duration.total_seconds() > 0 else 0
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
                            'current_count': obj_details.get('CurrentCount', 0),
                            'total_count': obj_details.get('Count', 0)
                        }

        return objectives

    def generate_structured_report(self) -> Dict[str, Any]:
        """
        Generate complete structured report for LLM formatting.

        Returns:
            Comprehensive structured data ready for narrative formatting
        """
        all_evaluations = self.evaluate_all()

        # Extract top quotes (highest confidence, diverse speakers)
        speaker_quotes = defaultdict(list)
        for t in self.transcripts:
            speaker_quotes[t['speaker']].append(t)

        top_quotes = []
        for speaker, quotes in speaker_quotes.items():
            sorted_quotes = sorted(quotes, key=lambda x: x.get('confidence', 0), reverse=True)
            top_quotes.extend(sorted_quotes[:5])  # Top 5 per speaker

        top_quotes = sorted(top_quotes, key=lambda x: x.get('confidence', 0), reverse=True)[:15]

        # Speaker statistics
        speaker_counts = Counter(t['speaker'] for t in self.transcripts)
        total_comms = len(self.transcripts)
        speaker_stats = [
            {
                'speaker': speaker,
                'utterances': count,
                'percentage': round(count / total_comms * 100, 2)
            }
            for speaker, count in speaker_counts.most_common()
        ]

        return {
            'metadata': {
                'total_events': len(self.events),
                'total_communications': len(self.transcripts),
                'unique_speakers': len(speaker_counts),
                'duration': all_evaluations['mission_specific']['mission_duration'],
                'avg_confidence': round(sum(t.get('confidence', 0) for t in self.transcripts) / total_comms, 3) if total_comms > 0 else 0
            },
            'speaker_statistics': speaker_stats,
            'top_communications': [
                {
                    'speaker': q['speaker'],
                    'text': q['text'],
                    'timestamp': q['timestamp'],
                    'confidence': q['confidence']
                }
                for q in top_quotes
            ],
            'evaluations': all_evaluations,
            # NOTE: raw_transcripts intentionally excluded to prevent LLM from
            # "recounting" and overriding pre-computed statistics. The LLM should
            # only format the facts provided, not verify them against raw data.
            'objectives': all_evaluations['mission_specific']['objectives']
        }
