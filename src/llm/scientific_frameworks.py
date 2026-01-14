"""
Scientific teamwork and learning frameworks for mission analysis.

This module implements research-backed frameworks for evaluating team performance:
- TeamSTEPPS (Agency for Healthcare Research and Quality)
- NASA 4-D System (Dr. Charlie Pellerin)
- Kirkpatrick Model (Training Evaluation)
- Bloom's Taxonomy (Cognitive Learning Levels)

References:
- https://www.ahrq.gov/teamstepps-program/resources/tools/index.html
- https://appel.nasa.gov/2018/05/09/supporting-effective-teamwork-at-nasa/
- https://www.valamis.com/hub/kirkpatrick-model
- https://uwaterloo.ca/centre-for-teaching-excellence/catalogs/tip-sheets/blooms-taxonomy
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime


class TeamSTEPPSDomain(Enum):
    """TeamSTEPPS five core domains for team performance measurement."""
    TEAM_STRUCTURE = "team_structure"
    LEADERSHIP = "leadership"
    SITUATION_MONITORING = "situation_monitoring"
    MUTUAL_SUPPORT = "mutual_support"
    COMMUNICATION = "communication"


class NASA4DDimension(Enum):
    """NASA 4-D System dimensions for team effectiveness."""
    CULTIVATING = "cultivating"  # People-building behaviors
    VISIONING = "visioning"      # Idea-building behaviors
    DIRECTING = "directing"      # System-building behaviors
    INCLUDING = "including"      # Relationship-building behaviors


class BloomLevel(Enum):
    """Bloom's Taxonomy cognitive levels (revised 2001)."""
    REMEMBER = 1     # Recall facts and basic concepts
    UNDERSTAND = 2   # Explain ideas or concepts
    APPLY = 3        # Use information in new situations
    ANALYZE = 4      # Draw connections among ideas
    EVALUATE = 5     # Justify a decision or course of action
    CREATE = 6       # Produce new or original work


class KirkpatrickLevel(Enum):
    """Kirkpatrick's Four Levels of Training Evaluation."""
    REACTION = 1     # How participants feel about training
    LEARNING = 2     # Knowledge/skills acquired
    BEHAVIOR = 3     # Application of learning
    RESULTS = 4      # Impact on outcomes


@dataclass
class TeamSTEPPSIndicators:
    """Observable behavior indicators for TeamSTEPPS domains."""

    # Team Structure indicators
    TEAM_STRUCTURE_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)(captain|helm|tactical|science|engineering|operations|communications)",
        r"(?i)(my station|my console|at my position)",
        r"(?i)(reporting|standing by|ready|online|operational)",
        r"(?i)(i have|i've got|taking|assuming)",
    ])

    # Leadership indicators
    LEADERSHIP_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)(set course|engage|execute|make it so|proceed)",
        r"(?i)(red alert|yellow alert|battle stations|stand down)",
        r"(?i)(all hands|attention|listen up|everyone)",
        r"(?i)(i want|we need|let's|should we)",
        r"(?i)(good work|well done|excellent|nice job)",
    ])

    # Situation Monitoring indicators
    SITUATION_MONITORING_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)(detecting|reading|scanning|picking up|sensors show)",
        r"(?i)(status|report|update|what's|how's|where)",
        r"(?i)(bearing|range|distance|coordinates|heading)",
        r"(?i)(enemy|hostile|threat|contact|target)",
        r"(?i)(shields at|hull at|power at|\d+%)",
    ])

    # Mutual Support indicators
    MUTUAL_SUPPORT_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)(help|assist|support|backup|cover)",
        r"(?i)(rerouting|diverting|transferring) power",
        r"(?i)(i can|let me|i'll handle|got it)",
        r"(?i)(watch out|be careful|heads up|warning)",
    ])

    # Communication indicators
    # Note: Whisper often transcribes "aye" as "eye", "I", or "ay"
    COMMUNICATION_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)\b(aye|ay|eye|acknowledged|understood|copy|roger|affirmative)\b",
        r"(?i)(sir|captain|ma'am)",
        r"(?i)(channel open|hailing|transmitting|receiving)",
        r"(?i)(confirm|verify|repeat|say again)",
        r"(?i)(negative|unable|cannot|problem)",
    ])


@dataclass
class NASA4DBehaviors:
    """Observable behaviors for NASA 4-D System dimensions."""

    # Cultivating dimension (People-building)
    CULTIVATING_PATTERNS: Dict[str, List[str]] = field(default_factory=lambda: {
        "authentic_appreciation": [
            r"(?i)(good work|well done|excellent|nice|great job|thank you|thanks)",
            r"(?i)(appreciate|grateful|impressed)",
        ],
        "shared_interests": [
            r"(?i)(our mission|our objective|we need to|together|team)",
            r"(?i)(let's|we should|shall we)",
        ],
    })

    # Visioning dimension (Idea-building)
    VISIONING_PATTERNS: Dict[str, List[str]] = field(default_factory=lambda: {
        "reality_based_optimism": [
            r"(?i)(we can|possible|solution|option|alternative)",
            r"(?i)(if we|what if|could we|might work)",
        ],
        "including_others": [
            r"(?i)(what do you think|any ideas|suggestions|input)",
            r"(?i)(your assessment|your opinion|thoughts)",
        ],
    })

    # Directing dimension (System-building)
    # Note: Whisper often transcribes "aye" as "eye", "I", or "ay"
    DIRECTING_PATTERNS: Dict[str, List[str]] = field(default_factory=lambda: {
        "keeping_agreements": [
            r"(?i)\b(aye|ay|eye|yes sir|understood|on it|doing it now)\b",
            r"(?i)(as ordered|as requested|will do)",
        ],
        "outcome_committed": [
            r"(?i)(objective|mission|goal|target|complete)",
            r"(?i)(priority|critical|essential|must)",
        ],
    })

    # Including dimension (Relationship-building)
    INCLUDING_PATTERNS: Dict[str, List[str]] = field(default_factory=lambda: {
        "resisting_blaming": [
            r"(?i)(it happens|no worries|we'll fix|don't worry)",
            r"(?i)(learn from|next time|adjust)",
        ],
        "clarifying_roles": [
            r"(?i)(you handle|i'll take|your job|my responsibility)",
            r"(?i)(who has|who is|which station)",
        ],
    })


@dataclass
class BloomVerbIndicators:
    """Observable verbs mapped to Bloom's Taxonomy cognitive levels."""

    LEVEL_PATTERNS: Dict[BloomLevel, List[str]] = field(default_factory=lambda: {
        BloomLevel.REMEMBER: [
            r"(?i)\b(recall|list|name|state|define|identify|repeat|label)\b",
            r"(?i)\b(what is|tell me|report)\b",
        ],
        BloomLevel.UNDERSTAND: [
            r"(?i)\b(explain|describe|interpret|summarize|classify|compare)\b",
            r"(?i)\b(shows|indicates|means|because|therefore)\b",
        ],
        BloomLevel.APPLY: [
            r"(?i)\b(use|execute|implement|operate|perform|apply)\b",
            r"(?i)\b(firing|launching|engaging|activating|deploying)\b",
        ],
        BloomLevel.ANALYZE: [
            r"(?i)\b(analyze|differentiate|organize|compare|contrast)\b",
            r"(?i)\b(pattern|connection|relationship|cause|effect)\b",
            r"(?i)\b(if.*then|because.*therefore|either.*or)\b",
        ],
        BloomLevel.EVALUATE: [
            r"(?i)\b(judge|assess|evaluate|recommend|decide|prioritize)\b",
            r"(?i)\b(should|best|better|worse|critical|important)\b",
            r"(?i)\b(i think|in my opinion|my assessment)\b",
        ],
        BloomLevel.CREATE: [
            r"(?i)\b(create|design|develop|propose|plan|construct)\b",
            r"(?i)\b(new approach|alternative|different way|modify)\b",
            r"(?i)\b(what if we|could we try|how about)\b",
        ],
    })


def analyze_teamstepps(
    transcripts: List[Dict[str, Any]],
    indicators: Optional[TeamSTEPPSIndicators] = None
) -> Dict[TeamSTEPPSDomain, Dict[str, Any]]:
    """
    Analyze transcripts using TeamSTEPPS framework.

    Args:
        transcripts: List of transcript dictionaries with 'text' and 'speaker'
        indicators: Optional custom indicators (uses defaults if None)

    Returns:
        Dictionary mapping domains to analysis results
    """
    if indicators is None:
        indicators = TeamSTEPPSIndicators()

    results = {domain: {"count": 0, "examples": [], "speakers": {}} for domain in TeamSTEPPSDomain}

    domain_patterns = {
        TeamSTEPPSDomain.TEAM_STRUCTURE: indicators.TEAM_STRUCTURE_PATTERNS,
        TeamSTEPPSDomain.LEADERSHIP: indicators.LEADERSHIP_PATTERNS,
        TeamSTEPPSDomain.SITUATION_MONITORING: indicators.SITUATION_MONITORING_PATTERNS,
        TeamSTEPPSDomain.MUTUAL_SUPPORT: indicators.MUTUAL_SUPPORT_PATTERNS,
        TeamSTEPPSDomain.COMMUNICATION: indicators.COMMUNICATION_PATTERNS,
    }

    for transcript in transcripts:
        text = transcript.get('text', '')
        speaker = transcript.get('speaker', 'unknown')
        timestamp = transcript.get('timestamp', '')

        for domain, patterns in domain_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    results[domain]["count"] += 1

                    # Track by speaker
                    if speaker not in results[domain]["speakers"]:
                        results[domain]["speakers"][speaker] = 0
                    results[domain]["speakers"][speaker] += 1

                    # Store example (limit to 5 per domain)
                    if len(results[domain]["examples"]) < 5:
                        results[domain]["examples"].append({
                            "timestamp": timestamp,
                            "speaker": speaker,
                            "text": text,
                            "pattern": pattern,
                        })
                    break  # Count each utterance once per domain

    # Calculate domain scores (1-5 scale based on frequency)
    total_utterances = len(transcripts) if transcripts else 1
    for domain in results:
        frequency = results[domain]["count"] / total_utterances
        # Scale: 0-10% = 1, 10-25% = 2, 25-40% = 3, 40-60% = 4, 60%+ = 5
        if frequency >= 0.60:
            results[domain]["score"] = 5
        elif frequency >= 0.40:
            results[domain]["score"] = 4
        elif frequency >= 0.25:
            results[domain]["score"] = 3
        elif frequency >= 0.10:
            results[domain]["score"] = 2
        else:
            results[domain]["score"] = 1
        results[domain]["frequency"] = round(frequency * 100, 1)

    return results


def analyze_nasa_4d(
    transcripts: List[Dict[str, Any]],
    behaviors: Optional[NASA4DBehaviors] = None
) -> Dict[NASA4DDimension, Dict[str, Any]]:
    """
    Analyze transcripts using NASA 4-D System framework.

    Args:
        transcripts: List of transcript dictionaries
        behaviors: Optional custom behavior patterns

    Returns:
        Dictionary mapping dimensions to analysis results
    """
    if behaviors is None:
        behaviors = NASA4DBehaviors()

    results = {dim: {"behaviors": {}, "total_count": 0, "examples": []} for dim in NASA4DDimension}

    dimension_patterns = {
        NASA4DDimension.CULTIVATING: behaviors.CULTIVATING_PATTERNS,
        NASA4DDimension.VISIONING: behaviors.VISIONING_PATTERNS,
        NASA4DDimension.DIRECTING: behaviors.DIRECTING_PATTERNS,
        NASA4DDimension.INCLUDING: behaviors.INCLUDING_PATTERNS,
    }

    for transcript in transcripts:
        text = transcript.get('text', '')
        speaker = transcript.get('speaker', 'unknown')
        timestamp = transcript.get('timestamp', '')

        for dimension, behavior_dict in dimension_patterns.items():
            for behavior_name, patterns in behavior_dict.items():
                for pattern in patterns:
                    if re.search(pattern, text):
                        # Initialize behavior tracking
                        if behavior_name not in results[dimension]["behaviors"]:
                            results[dimension]["behaviors"][behavior_name] = {
                                "count": 0,
                                "speakers": {}
                            }

                        results[dimension]["behaviors"][behavior_name]["count"] += 1
                        results[dimension]["total_count"] += 1

                        # Track by speaker
                        speakers = results[dimension]["behaviors"][behavior_name]["speakers"]
                        if speaker not in speakers:
                            speakers[speaker] = 0
                        speakers[speaker] += 1

                        # Store example
                        if len(results[dimension]["examples"]) < 3:
                            results[dimension]["examples"].append({
                                "behavior": behavior_name,
                                "timestamp": timestamp,
                                "speaker": speaker,
                                "text": text,
                            })
                        break

    # Calculate dimension scores
    total_utterances = len(transcripts) if transcripts else 1
    for dimension in results:
        frequency = results[dimension]["total_count"] / total_utterances
        if frequency >= 0.30:
            results[dimension]["score"] = 5
        elif frequency >= 0.20:
            results[dimension]["score"] = 4
        elif frequency >= 0.10:
            results[dimension]["score"] = 3
        elif frequency >= 0.05:
            results[dimension]["score"] = 2
        else:
            results[dimension]["score"] = 1

    return results


def analyze_bloom_levels(
    transcripts: List[Dict[str, Any]],
    indicators: Optional[BloomVerbIndicators] = None
) -> Dict[BloomLevel, Dict[str, Any]]:
    """
    Analyze transcripts for cognitive complexity using Bloom's Taxonomy.

    Args:
        transcripts: List of transcript dictionaries
        indicators: Optional custom verb indicators

    Returns:
        Dictionary mapping Bloom levels to analysis results
    """
    if indicators is None:
        indicators = BloomVerbIndicators()

    results = {level: {"count": 0, "examples": [], "speakers": {}} for level in BloomLevel}

    for transcript in transcripts:
        text = transcript.get('text', '')
        speaker = transcript.get('speaker', 'unknown')
        timestamp = transcript.get('timestamp', '')

        # Find highest matching Bloom level for this utterance
        highest_level = None
        matched_pattern = None

        for level in reversed(list(BloomLevel)):  # Start from CREATE down to REMEMBER
            for pattern in indicators.LEVEL_PATTERNS[level]:
                if re.search(pattern, text):
                    highest_level = level
                    matched_pattern = pattern
                    break
            if highest_level:
                break

        if highest_level:
            results[highest_level]["count"] += 1

            # Track by speaker
            if speaker not in results[highest_level]["speakers"]:
                results[highest_level]["speakers"][speaker] = 0
            results[highest_level]["speakers"][speaker] += 1

            # Store example
            if len(results[highest_level]["examples"]) < 3:
                results[highest_level]["examples"].append({
                    "timestamp": timestamp,
                    "speaker": speaker,
                    "text": text,
                })

    # Calculate cognitive complexity distribution
    total_classified = sum(r["count"] for r in results.values())
    for level in results:
        if total_classified > 0:
            results[level]["percentage"] = round(results[level]["count"] / total_classified * 100, 1)
        else:
            results[level]["percentage"] = 0

    # Calculate average cognitive level
    if total_classified > 0:
        weighted_sum = sum(level.value * results[level]["count"] for level in BloomLevel)
        avg_level = weighted_sum / total_classified
    else:
        avg_level = 1.0

    return {
        "levels": results,
        "average_cognitive_level": round(avg_level, 2),
        "total_classified": total_classified,
    }


def calculate_response_times(
    transcripts: List[Dict[str, Any]],
    command_patterns: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate response times between commands and acknowledgments.

    This uses transcript timestamps to measure how quickly crew members
    respond to leadership communications.

    Args:
        transcripts: List of transcript dictionaries with timestamps
        command_patterns: Regex patterns that identify commands

    Returns:
        Dictionary with response time metrics
    """
    if command_patterns is None:
        command_patterns = [
            r"(?i)(set course|engage|execute|fire|launch|raise shields|red alert)",
            r"(?i)(helm|tactical|science|engineering|operations),?\s+",
            r"(?i)(status|report|what's|how's|update)",
        ]

    # Note: Whisper often transcribes "aye" as "eye", "I", or "ay"
    acknowledgment_patterns = [
        r"(?i)\b(aye|ay|eye|acknowledged|yes sir|understood|copy|roger|on it)\b",
        r"(?i)(ready|standing by|online|operational)",
    ]

    response_times = []
    command_response_pairs = []

    # Parse timestamps
    def parse_timestamp(ts: str) -> Optional[datetime]:
        if isinstance(ts, datetime):
            return ts
        try:
            if 'T' in str(ts):
                return datetime.fromisoformat(str(ts).replace('Z', '+00:00'))
        except (ValueError, TypeError):
            pass
        return None

    # Find command-response pairs
    for i, transcript in enumerate(transcripts):
        text = transcript.get('text', '')
        speaker = transcript.get('speaker', '')
        timestamp = parse_timestamp(transcript.get('timestamp', ''))

        # Check if this is a command
        is_command = any(re.search(p, text) for p in command_patterns)

        if is_command and timestamp:
            # Look for acknowledgment in next few utterances
            for j in range(i + 1, min(i + 5, len(transcripts))):
                next_transcript = transcripts[j]
                next_text = next_transcript.get('text', '')
                next_speaker = next_transcript.get('speaker', '')
                next_timestamp = parse_timestamp(next_transcript.get('timestamp', ''))

                # Check if different speaker and is acknowledgment
                if next_speaker != speaker and next_timestamp:
                    is_ack = any(re.search(p, next_text) for p in acknowledgment_patterns)
                    if is_ack:
                        delta = (next_timestamp - timestamp).total_seconds()
                        if 0 < delta < 30:  # Reasonable response window
                            response_times.append(delta)
                            command_response_pairs.append({
                                "command": {
                                    "speaker": speaker,
                                    "text": text,
                                    "timestamp": str(timestamp),
                                },
                                "response": {
                                    "speaker": next_speaker,
                                    "text": next_text,
                                    "timestamp": str(next_timestamp),
                                },
                                "response_time_seconds": round(delta, 2),
                            })
                        break

    # Calculate statistics
    if response_times:
        avg_response = sum(response_times) / len(response_times)
        min_response = min(response_times)
        max_response = max(response_times)
    else:
        avg_response = min_response = max_response = 0

    return {
        "command_response_pairs": len(command_response_pairs),
        "average_response_time_seconds": round(avg_response, 2),
        "min_response_time_seconds": round(min_response, 2),
        "max_response_time_seconds": round(max_response, 2),
        "examples": command_response_pairs[:5],  # Top 5 examples
    }


def generate_kirkpatrick_assessment(
    transcripts: List[Dict[str, Any]],
    mission_objectives: List[Dict[str, Any]],
    bloom_analysis: Dict[str, Any],
    teamstepps_analysis: Dict[TeamSTEPPSDomain, Dict[str, Any]],
) -> Dict[KirkpatrickLevel, Dict[str, Any]]:
    """
    Generate Kirkpatrick model assessment based on available data.

    Level 1 (Reaction) requires survey data - marked as N/A
    Level 2 (Learning) uses Bloom's taxonomy analysis
    Level 3 (Behavior) uses TeamSTEPPS observable behaviors
    Level 4 (Results) uses mission objective completion

    Args:
        transcripts: List of transcript dictionaries
        mission_objectives: List of mission objective dictionaries
        bloom_analysis: Results from analyze_bloom_levels()
        teamstepps_analysis: Results from analyze_teamstepps()

    Returns:
        Dictionary mapping Kirkpatrick levels to assessments
    """
    results = {}

    # Level 1: Reaction (requires survey - not available from transcripts)
    results[KirkpatrickLevel.REACTION] = {
        "available": False,
        "note": "Requires post-mission survey data",
        "recommendation": "Implement crew satisfaction survey after each mission",
    }

    # Level 2: Learning (from Bloom's analysis)
    avg_cognitive = bloom_analysis.get("average_cognitive_level", 1.0)
    learning_score = min(5, max(1, int(avg_cognitive)))
    results[KirkpatrickLevel.LEARNING] = {
        "available": True,
        "score": learning_score,
        "average_cognitive_level": avg_cognitive,
        "interpretation": _interpret_learning_level(avg_cognitive),
        "distribution": {
            level.name: bloom_analysis["levels"][level]["percentage"]
            for level in BloomLevel
        },
    }

    # Level 3: Behavior (from TeamSTEPPS)
    behavior_scores = [analysis["score"] for analysis in teamstepps_analysis.values()]
    avg_behavior = sum(behavior_scores) / len(behavior_scores) if behavior_scores else 1
    results[KirkpatrickLevel.BEHAVIOR] = {
        "available": True,
        "score": round(avg_behavior, 1),
        "domain_scores": {
            domain.value: analysis["score"]
            for domain, analysis in teamstepps_analysis.items()
        },
        "interpretation": _interpret_behavior_level(avg_behavior),
    }

    # Level 4: Results (from mission objectives)
    completed = sum(1 for obj in mission_objectives if obj.get('complete', False))
    total = len(mission_objectives) if mission_objectives else 1
    completion_rate = completed / total

    if completion_rate >= 0.8:
        results_score = 5
    elif completion_rate >= 0.6:
        results_score = 4
    elif completion_rate >= 0.4:
        results_score = 3
    elif completion_rate >= 0.2:
        results_score = 2
    else:
        results_score = 1

    results[KirkpatrickLevel.RESULTS] = {
        "available": True,
        "score": results_score,
        "objectives_completed": completed,
        "objectives_total": total,
        "completion_rate": round(completion_rate * 100, 1),
        "interpretation": _interpret_results_level(completion_rate),
    }

    return results


def _interpret_learning_level(avg_cognitive: float) -> str:
    """Interpret average cognitive level."""
    if avg_cognitive >= 5:
        return "Exceptional - Crew demonstrates high-level evaluation and creative problem-solving"
    elif avg_cognitive >= 4:
        return "Strong - Crew actively analyzes situations and makes informed judgments"
    elif avg_cognitive >= 3:
        return "Competent - Crew applies knowledge effectively to mission tasks"
    elif avg_cognitive >= 2:
        return "Developing - Crew understands procedures but needs more application practice"
    else:
        return "Basic - Crew primarily operates at recall level; needs deeper engagement"


def _interpret_behavior_level(avg_behavior: float) -> str:
    """Interpret average TeamSTEPPS behavior score."""
    if avg_behavior >= 4.5:
        return "Exemplary teamwork - All TeamSTEPPS domains consistently demonstrated"
    elif avg_behavior >= 3.5:
        return "Strong teamwork - Most domains well-practiced with minor gaps"
    elif avg_behavior >= 2.5:
        return "Adequate teamwork - Core behaviors present but inconsistent"
    elif avg_behavior >= 1.5:
        return "Developing teamwork - Several domains need significant improvement"
    else:
        return "Minimal teamwork behaviors observed - Comprehensive training recommended"


def _interpret_results_level(completion_rate: float) -> str:
    """Interpret mission completion rate."""
    if completion_rate >= 0.8:
        return "Mission success - Crew achieved primary and secondary objectives"
    elif completion_rate >= 0.6:
        return "Partial success - Most primary objectives achieved"
    elif completion_rate >= 0.4:
        return "Mixed results - Some objectives achieved but gaps remain"
    elif completion_rate >= 0.2:
        return "Limited success - Significant improvement needed"
    else:
        return "Mission incomplete - Critical training gaps identified"
