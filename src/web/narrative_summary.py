"""
LLM-powered narrative summary generator for audio analysis.

Generates engaging, insightful narrative summaries of bridge crew
communication sessions using Ollama models.

Includes hallucination prevention:
- Constrained context building (only verified data)
- Lower temperature for factual content
- Post-generation validation
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5:14b-instruct')
OLLAMA_TIMEOUT = float(os.getenv('OLLAMA_TIMEOUT', '600'))  # 10 minutes for large models
LLM_REPORT_STYLE = os.getenv('LLM_REPORT_STYLE', 'entertaining')

# Import hallucination prevention if available
try:
    from src.llm.hallucination_prevention import (
        OutputValidator,
        clean_hallucinations,
        ANTI_HALLUCINATION_PARAMS,
        STORY_PARAMS,
    )
    HALLUCINATION_PREVENTION_AVAILABLE = True
except ImportError:
    HALLUCINATION_PREVENTION_AVAILABLE = False
    # Default parameters if module not available
    ANTI_HALLUCINATION_PARAMS = {
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "num_predict": 500,
    }
    STORY_PARAMS = {
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 50,
        "repeat_penalty": 1.1,
        "num_predict": 1200,
    }

# Telemetry timeline builder for story event grouping
try:
    from src.metrics.telemetry_timeline import TelemetryTimelineBuilder
    TELEMETRY_TIMELINE_AVAILABLE = True
except ImportError:
    TELEMETRY_TIMELINE_AVAILABLE = False
    TelemetryTimelineBuilder = None


class NarrativeSummaryGenerator:
    """
    Generates narrative summaries of audio analysis using LLM.

    Takes structured analysis data and generates human-readable
    narrative summaries highlighting key moments, crew dynamics,
    and actionable insights.
    """

    def __init__(
        self,
        ollama_host: Optional[str] = None,
        ollama_model: Optional[str] = None,
        timeout: Optional[float] = None,
        style: Optional[str] = None
    ):
        """
        Initialize narrative summary generator.

        Args:
            ollama_host: Ollama API host URL
            ollama_model: Model to use for generation
            timeout: Request timeout in seconds
            style: Report style (entertaining, professional, technical, casual)
        """
        self.ollama_host = ollama_host or OLLAMA_HOST
        self.ollama_model = ollama_model or OLLAMA_MODEL
        self.timeout = timeout or OLLAMA_TIMEOUT
        self.style = style or LLM_REPORT_STYLE
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def check_ollama_available(self) -> bool:
        """Check if Ollama is available and model is loaded."""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.ollama_host}/api/tags")
            if response.status_code != 200:
                return False

            # Check if our model is available
            data = response.json()
            models = data.get('models', [])
            model_names = [m.get('name', '') for m in models]
            model_base = self.ollama_model.split(':')[0]

            return any(
                self.ollama_model in name or model_base in name
                for name in model_names
            )
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return False

    def _assess_metric_reliability(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """
        Assess reliability of each metric (0.0-1.0).

        Returns reliability scores based on data quality indicators.

        Args:
            analysis: Full analysis results

        Returns:
            Dictionary mapping metric names to reliability scores (0.0-1.0)
        """
        reliability = {}

        # Seven Habits reliability based on match rate
        habits = analysis.get('seven_habits', {})
        if habits:
            habits_list = habits.get('habits', [])
            if habits_list:
                total_obs = sum(
                    h.get('observation_count', h.get('count', 0))
                    for h in habits_list
                )
                total_utterances = len(analysis.get('transcription', []))
                match_rate = total_obs / max(total_utterances, 1)

                # Reliability thresholds
                if match_rate < 0.05:
                    reliability['seven_habits'] = 0.2  # Very low: sparse observations
                elif match_rate < 0.15:
                    reliability['seven_habits'] = 0.6  # Moderate: limited observations
                else:
                    reliability['seven_habits'] = 1.0  # High: good coverage
            else:
                reliability['seven_habits'] = 0.0
        else:
            reliability['seven_habits'] = 0.0

        # Role assignments reliability based on speaker count
        roles = analysis.get('role_assignments', [])
        speaker_count = len(analysis.get('speakers', []))
        if roles and speaker_count > 0:
            role_coverage = len(roles) / max(speaker_count, 6)
            reliability['role_assignments'] = 1.0 if role_coverage >= 0.5 else 0.5
        else:
            reliability['role_assignments'] = 0.0

        # Transcription confidence
        conf_dist = analysis.get('confidence_distribution', {})
        avg_confidence = conf_dist.get('average_confidence', 1.0)
        if avg_confidence < 0.40:
            reliability['transcription'] = 0.3
        elif avg_confidence < 0.60:
            reliability['transcription'] = 0.7
        else:
            reliability['transcription'] = 1.0

        return reliability

    def _build_analysis_context(self, analysis: Dict[str, Any]) -> str:
        """
        Build context string from analysis results, filtered by reliability.

        Unreliable metrics are excluded or marked with warnings to prevent
        the LLM from receiving conflicting data.

        Args:
            analysis: Full analysis results dictionary

        Returns:
            Formatted context string for LLM
        """
        sections = []

        # Assess reliability of all metrics first
        reliability = self._assess_metric_reliability(analysis)

        # Check transcription confidence first
        conf_dist = analysis.get('confidence_distribution', {})
        avg_confidence = conf_dist.get('average_confidence', 1.0)

        if avg_confidence < 0.40:
            sections.append("⚠️ DATA QUALITY WARNING ⚠️")
            sections.append(f"Transcription confidence is only {avg_confidence*100:.0f}%.")
            sections.append("The transcript may contain significant errors.")
            sections.append("Be cautious about specific quotes and use hedging language")
            sections.append("(e.g., 'appeared to', 'seemed to', 'what sounded like').")
            sections.append("")
        elif avg_confidence < 0.60:
            sections.append("ℹ️ Note: Transcription confidence is {:.0f}% - some details may be imprecise.".format(avg_confidence*100))
            sections.append("")

        # Duration and basic stats
        duration = analysis.get('duration_seconds', 0)
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        sections.append(f"SESSION DURATION: {minutes}m {seconds}s")

        # Speaker information with roles
        speakers = analysis.get('speakers', [])
        role_assignments = analysis.get('role_assignments', [])

        # Build role map
        role_map = {}
        for ra in role_assignments:
            if ra.get('speaker_id') and ra.get('role'):
                role_map[ra['speaker_id']] = {
                    'role': ra['role'],
                    'confidence': ra.get('confidence', 0)
                }

        if speakers:
            sections.append("\nCREW MEMBERS DETECTED:")
            for s in speakers:
                speaker_id = s.get('speaker_id', 'Unknown')
                role_info = role_map.get(speaker_id, {})
                role = role_info.get('role', 'Crew Member')
                conf = role_info.get('confidence', 0)
                speaking_time = s.get('total_speaking_time', 0)
                utterances = s.get('utterance_count', 0)
                sections.append(
                    f"  - {role} ({speaker_id}): {utterances} utterances, "
                    f"{speaking_time:.1f}s speaking time, {conf*100:.0f}% role confidence"
                )

        # Seven Habits analysis WITH EVIDENCE + RELIABILITY FILTERING
        habits = analysis.get('seven_habits', {})
        habits_reliability = reliability.get('seven_habits', 0.0)

        if habits:
            # Add reliability warning if low
            if habits_reliability < 0.6:
                sections.append("\n" + "⚠️ LOW DATA QUALITY WARNING ⚠️")
                sections.append("Seven Habits scores have LIMITED RELIABILITY.")
                sections.append("Root cause: Few observable demonstrations of habits in transcript.")
                sections.append("→ USE TRANSCRIPT EVIDENCE AS PRIMARY SOURCE")
                sections.append("→ IGNORE contradictory scores if metrics show <5% frequency")
                sections.append("")

            if habits_reliability >= 0.5:
                # Show full habits section only if moderate+ reliability
                sections.append("\n" + "="*50)
                sections.append("SEVEN HABITS ANALYSIS (with evidence)")
                sections.append("="*50)

                # Get habits list - could be 'habits' or 'habit_scores'
                habits_list = habits.get('habits', habits.get('habit_scores', []))
                for h in habits_list:
                    name = h.get('youth_friendly_name') or h.get('name', 'Unknown')
                    score = h.get('score', 0)
                    interpretation = h.get('interpretation', '')
                    sections.append(f"\n{name}: {score}/5")
                    sections.append(f"  Assessment: {interpretation}")

                    # Include actual examples as evidence
                    examples = h.get('examples', [])
                    if examples:
                        sections.append("  EVIDENCE FROM TRANSCRIPT:")
                        for ex in examples[:3]:
                            if isinstance(ex, dict):
                                speaker = ex.get('speaker', 'Unknown')
                                text = ex.get('text', '')
                                role = role_map.get(speaker, {}).get('role', speaker)
                                sections.append(f"    - [{role}]: \"{text}\"")
                            elif isinstance(ex, str):
                                sections.append(f"    - \"{ex}\"")

                    # Include development tip
                    tip = h.get('development_tip', '')
                    if tip:
                        sections.append(f"  Growth tip: {tip}")

                # Overall score
                overall = habits.get('overall_score', 0)
                sections.append(f"\nOVERALL TEAM SCORE: {overall}/5")

        # Communication quality WITH EVIDENCE
        quality = analysis.get('communication_quality', {})
        if quality:
            sections.append("\n" + "="*50)
            sections.append("COMMUNICATION QUALITY ANALYSIS (with evidence)")
            sections.append("="*50)

            effective_pct = quality.get('effective_percentage', 0)
            effective_count = quality.get('effective_count', 0)
            improvement_count = quality.get('improvement_count', 0)
            sections.append(f"\nOverall: {effective_pct:.0f}% effective ({effective_count} good, {improvement_count} need work)")

            # Key patterns with examples
            patterns = quality.get('patterns', [])
            effective_patterns = [p for p in patterns if p.get('category') == 'effective']
            improvement_patterns = [p for p in patterns if p.get('category') == 'needs_improvement']

            if effective_patterns:
                sections.append("\nEFFECTIVE COMMUNICATION PATTERNS:")
                for p in effective_patterns[:5]:
                    pattern_name = p.get('pattern_name', '')
                    count = p.get('count', 0)
                    sections.append(f"  {pattern_name}: {count} instances")
                    # Include examples if available
                    examples = p.get('examples', [])
                    for ex in examples[:2]:
                        if isinstance(ex, dict):
                            speaker = ex.get('speaker', ex.get('speaker_id', 'Unknown'))
                            text = ex.get('text', '')
                            role = role_map.get(speaker, {}).get('role', speaker)
                            sections.append(f"    Example: [{role}]: \"{text[:150]}\"")

            if improvement_patterns:
                sections.append("\nPATTERNS NEEDING IMPROVEMENT:")
                for p in improvement_patterns[:3]:
                    pattern_name = p.get('pattern_name', '')
                    count = p.get('count', 0)
                    sections.append(f"  {pattern_name}: {count} instances")

        # Speaker Scorecards WITH EVIDENCE
        scorecards = analysis.get('speaker_scorecards', [])
        if scorecards:
            sections.append("\n" + "="*50)
            sections.append("INDIVIDUAL PERFORMANCE (with evidence)")
            sections.append("="*50)

            for sc in scorecards:
                speaker_id = sc.get('speaker_id', 'Unknown')
                role = role_map.get(speaker_id, {}).get('role', speaker_id)
                overall = sc.get('overall_score', 0)
                sections.append(f"\n{role} - Overall: {overall}/5")

                # Include metric scores with evidence
                metrics = sc.get('metrics', [])
                for m in metrics[:4]:  # Top 4 metrics
                    metric_name = m.get('metric_name', '')
                    score = m.get('score', 0)
                    sections.append(f"  {metric_name}: {score}/5")

                    # Include evidence quotes
                    evidence = m.get('evidence', [])
                    for ev in evidence[:1]:  # One example per metric
                        if isinstance(ev, dict):
                            text = ev.get('text', '')
                            if text:
                                sections.append(f"    Evidence: \"{text[:120]}\"")
                        elif isinstance(ev, str):
                            sections.append(f"    Evidence: \"{ev[:120]}\"")

                # Strengths and areas for growth
                strengths = sc.get('strengths', [])
                if strengths:
                    sections.append(f"  Strengths: {', '.join(strengths[:3])}")

                areas = sc.get('areas_for_growth', [])
                if areas:
                    sections.append(f"  Growth areas: {', '.join(areas[:2])}")

        # Learning evaluation
        learning = analysis.get('learning_evaluation', {})
        if learning:
            if learning.get('key_insights'):
                sections.append("\nKEY LEARNING INSIGHTS:")
                for insight in learning['key_insights'][:5]:
                    sections.append(f"  - {insight}")

            if learning.get('recommendations'):
                sections.append("\nTRAINING RECOMMENDATIONS:")
                for rec in learning['recommendations'][:5]:
                    sections.append(f"  - {rec}")

        # Telemetry game events (actual in-game data)
        telemetry_summary = analysis.get('telemetry_summary')
        if telemetry_summary and telemetry_summary.get('total_events', 0) > 0:
            sections.append("\n" + "="*50)
            sections.append("GAME TELEMETRY DATA (actual in-game events)")
            sections.append("="*50)
            sections.append(f"\nTotal game events: {telemetry_summary['total_events']}")

            # Event category breakdown
            event_summ = telemetry_summary.get('event_summary', {})
            cat_dist = event_summ.get('category_distribution', {})
            if cat_dist:
                sections.append("\nActivity by category:")
                for cat, count in cat_dist.items():
                    sections.append(f"  - {cat}: {count} events")

            # Mission phases from telemetry
            phases = telemetry_summary.get('phases', [])
            if phases:
                sections.append("\nMission phases (from game telemetry):")
                for phase in phases:
                    sections.append(
                        f"  - {phase.get('start_formatted', '?')}-{phase.get('end_formatted', '?')}: "
                        f"{phase.get('display_name', 'Unknown')} ({phase.get('event_count', 0)} events)"
                    )

            # Key game events
            key_events = telemetry_summary.get('key_events', [])
            if key_events:
                sections.append("\nKEY GAME EVENTS (use these to ground your analysis):")
                for event in key_events[:15]:
                    sections.append(
                        f"  [{event.get('time_formatted', '?')}] {event.get('description', 'Unknown event')}"
                    )
                if len(key_events) > 15:
                    sections.append(f"  ... and {len(key_events) - 15} more events")

        # Selected transcript highlights (high confidence, interesting content)
        transcripts = analysis.get('transcription', [])
        if transcripts:
            # Sort by confidence and pick notable ones
            sorted_trans = sorted(
                [t for t in transcripts if t.get('confidence', 0) > 0.5],
                key=lambda x: x.get('confidence', 0),
                reverse=True
            )

            # Get a good mix of communications to show team dynamics
            if sorted_trans:
                sections.append("\nKEY COMMUNICATIONS (use these as evidence in your narrative):")
                shown = 0
                for t in sorted_trans:
                    if shown >= 20:
                        break
                    speaker = t.get('speaker_id', 'Unknown')
                    role = role_map.get(speaker, {}).get('role', speaker)
                    text = t.get('text', '').strip()
                    if len(text) > 10:  # Skip very short utterances
                        sections.append(f"  [{role}]: \"{text[:250]}\"")
                        shown += 1

            # Include representative transcript sample for context
            # For long recordings, sample from beginning, middle, and end
            MAX_CONTEXT_SEGMENTS = 100
            total_count = len(transcripts)

            if total_count > MAX_CONTEXT_SEGMENTS:
                # Sample: 30 from start, 40 from middle (high confidence), 30 from end
                beginning = transcripts[:30]
                ending = transcripts[-30:]
                middle_pool = transcripts[30:-30]
                # Take highest confidence from middle
                middle_sorted = sorted(
                    middle_pool,
                    key=lambda x: x.get('confidence', 0.5),
                    reverse=True
                )[:40]
                # Re-sort by time (using start_time)
                middle_sorted.sort(key=lambda x: x.get('start_time', 0) if isinstance(x.get('start_time'), (int, float)) else 0)
                context_sample = beginning + middle_sorted + ending
                sections.append(f"\nTRANSCRIPT SAMPLE ({len(context_sample)} of {total_count} utterances):")
            else:
                context_sample = transcripts
                sections.append(f"\nFULL TRANSCRIPT ({total_count} utterances):")

            for t in context_sample:
                speaker = t.get('speaker_id', 'Unknown')
                role = role_map.get(speaker, {}).get('role', speaker)
                text = t.get('text', '').strip()
                if text:
                    sections.append(f"  [{role}]: \"{text[:200]}\"")

        return '\n'.join(sections)

    def _get_style_instructions(self) -> str:
        """Get style-specific instructions for the LLM."""
        styles = {
            'entertaining': """
## VOICE & TONE
Write like an enthusiastic coach who genuinely loves watching teams grow.
Be warm and personable - you're impressed by good teamwork and excited to share what you observed.
Use vivid, energetic language that makes the crew feel proud of their achievements.
Sprinkle in occasional wit or clever observations, but stay professional.
Your goal: make this crew feel seen, valued, and motivated to keep improving.
""",
            'professional': """
## VOICE & TONE
Write as a seasoned training consultant presenting findings to leadership.
Be objective but encouraging - acknowledge successes before suggesting improvements.
Use precise language and reference the framework data (Seven Habits scores, communication patterns).
Your goal: provide credible, actionable insights that leadership can use for training decisions.
""",
            'technical': """
## VOICE & TONE
Write as a crew resource management (CRM) specialist analyzing communication patterns.
Reference closed-loop communication, situational awareness, and team coordination theory.
Be analytical but appreciative of good technique when you see it.
Your goal: help the crew understand the science behind effective teamwork.
""",
            'casual': """
## VOICE & TONE
Write like a friendly mentor debriefing with the crew over coffee.
Be conversational and relatable - use "you" and "your team" language.
Focus on celebrating wins and framing growth areas as exciting challenges.
Avoid jargon - explain concepts simply.
Your goal: make feedback feel like friendly advice, not evaluation.
"""
        }
        return styles.get(self.style, styles['entertaining'])

    def _compute_mission_grade(self, analysis: Dict[str, Any]) -> str:
        """
        Compute mission grade from data using explicit criteria with reliability weighting.

        Grade criteria:
        - A: ≥80% objectives complete AND avg scorecard ≥4.0 AND avg habits ≥4.0
        - B: ≥60% objectives OR (avg scorecard ≥3.0 AND avg habits ≥3.0)
        - C: ≥40% objectives OR (avg scorecard ≥2.5 AND avg habits ≥2.5)
        - D: ≥20% objectives OR avg scorecard ≥2.0
        - F: <20% objectives AND avg scorecard <2.0

        When objectives data is unavailable, grade is based on scorecard
        and habits scores only. Habits score is weighted by its reliability.

        Args:
            analysis: Full analysis results

        Returns:
            Grade string with justification, e.g. "B (scorecard 3.4/5, habits 3.2/5*)"
        """
        # Assess metric reliability
        reliability = self._assess_metric_reliability(analysis)
        habits_reliability = reliability.get('seven_habits', 0.0)

        # Objective completion rate
        game_context = analysis.get('game_context') or {}
        objectives = game_context.get('objectives', [])
        if objectives:
            completed = sum(1 for o in objectives if o.get('complete', False))
            obj_rate = completed / len(objectives)
        else:
            obj_rate = None  # No objective data

        # Average scorecard score
        scorecards = analysis.get('speaker_scorecards', [])
        if scorecards:
            sc_scores = []
            for sc in scorecards:
                overall = sc.get('overall_score', 0)
                if isinstance(overall, (int, float)) and overall > 0:
                    sc_scores.append(overall)
            avg_scorecard = sum(sc_scores) / len(sc_scores) if sc_scores else 0
        else:
            avg_scorecard = 0

        # Average habits score (with reliability weighting)
        habits = analysis.get('seven_habits') or {}
        avg_habits = habits.get('overall_score', 0)
        if isinstance(avg_habits, (int, float)):
            avg_habits = float(avg_habits)
        else:
            avg_habits = 0

        # Apply reliability weighting to habits score
        # Low reliability habits shouldn't drag down the grade as much
        if habits_reliability < 0.6:
            # De-weight unreliable habits - use scorecard as primary source
            avg_perf_for_grade = avg_scorecard
        else:
            # Normal weighting when habits are reliable
            avg_perf_for_grade = (avg_scorecard + avg_habits) / 2 if (avg_scorecard and avg_habits) else max(avg_scorecard, avg_habits)

        # Determine grade
        if obj_rate is not None:
            if obj_rate >= 0.80 and avg_scorecard >= 4.0 and avg_habits >= 4.0:
                grade = "A"
            elif obj_rate >= 0.60 or (avg_scorecard >= 3.0 and avg_habits >= 3.0):
                grade = "B"
            elif obj_rate >= 0.40 or (avg_scorecard >= 2.5 and avg_habits >= 2.5):
                grade = "C"
            elif obj_rate >= 0.20 or avg_scorecard >= 2.0:
                grade = "D"
            else:
                grade = "F"
        else:
            # No objectives — grade on team performance only
            if avg_perf_for_grade >= 4.0:
                grade = "A"
            elif avg_perf_for_grade >= 3.0:
                grade = "B"
            elif avg_perf_for_grade >= 2.5:
                grade = "C"
            elif avg_perf_for_grade >= 2.0:
                grade = "D"
            else:
                grade = "F"

        # Build justification string
        parts = []
        if obj_rate is not None:
            parts.append(f"objectives {obj_rate*100:.0f}%")
        parts.append(f"scorecard {avg_scorecard:.1f}/5")

        # Add asterisk to habits if low reliability
        habits_str = f"habits {avg_habits:.1f}/5"
        if habits_reliability < 0.6:
            habits_str += "*"
        parts.append(habits_str)

        return f"{grade} ({', '.join(parts)})"

    def _build_prompt(self, analysis: Dict[str, Any]) -> str:
        """
        Build the complete prompt for narrative generation.

        Args:
            analysis: Full analysis results

        Returns:
            Complete prompt string
        """
        context = self._build_analysis_context(analysis)
        style_instructions = self._get_style_instructions()
        mission_grade = self._compute_mission_grade(analysis)

        # Build captain leadership context if available
        captain_section = ""
        captain_leadership = analysis.get('captain_leadership')
        if captain_leadership:
            captain_section = self._build_captain_context(captain_leadership)

        prompt = f"""You are an experienced bridge crew instructor providing a structured mission debrief.

## YOUR TASK
Generate a CONCISE, STRUCTURED debrief using bullet points and tables. NO PROSE PARAGRAPHS.

{style_instructions}

## SESSION DATA
{context}
{captain_section}

---

## COMPUTED MISSION GRADE: {mission_grade}
(This grade was computed from the data above. Use it as-is in your output.)

## OUTPUT FORMAT (use exactly this structure)

### Mission Grade: {mission_grade}
One sentence summary explaining why this grade was earned.

### Top 3 Strengths (with evidence)
1. **[Strength]**: "[Exact quote]" - [Role]
2. **[Strength]**: "[Exact quote]" - [Role]
3. **[Strength]**: "[Exact quote]" - [Role]

### Top 2 Growth Areas
1. **[Issue]**: What happened and ONE specific fix
2. **[Issue]**: What happened and ONE specific fix

### Individual Performance
| Role | Did Well | Work On |
|------|----------|---------|
| Captain | [Specific] | [Specific] |
| Helm | [Specific] | [Specific] |
| Tactical | [Specific] | [Specific] |
| (others) | ... | ... |

### Quick Wins for Next Mission
- [ ] [Action 1 - tied to growth area]
- [ ] [Action 2 - tied to growth area]
- [ ] [Action 3 - communication improvement]

## RULES
1. ONLY use quotes from the transcript - never invent
2. Use role names (Captain, Helm) not speaker IDs
3. Keep total output under 200 words
4. Every claim needs a quote as evidence
5. Be specific and actionable, not vague
6. Use the computed mission grade exactly as provided — do not override it

Generate the structured debrief now:"""

        return prompt

    def _build_captain_context(self, captain_leadership: Dict[str, Any]) -> str:
        """
        Build context string for captain leadership assessment.

        Args:
            captain_leadership: Captain leadership assessment results

        Returns:
            Formatted context string for LLM
        """
        sections = []
        sections.append("\n" + "=" * 50)
        sections.append("CAPTAIN LEADERSHIP ASSESSMENT (with evidence)")
        sections.append("=" * 50)

        captain_id = captain_leadership.get('captain_speaker', 'Unknown')
        overall = captain_leadership.get('overall_score', 0)
        sections.append(f"\nCaptain: {captain_id} — Overall Leadership: {overall}/5")

        dimensions = captain_leadership.get('dimensions', {})
        for dim_name, dim_data in dimensions.items():
            display_name = dim_name.replace('_', ' ').title()
            score = dim_data.get('score', 0)
            evidence = dim_data.get('evidence', '')
            sections.append(f"\n  {display_name}: {score}/5")
            if evidence:
                sections.append(f"    {evidence}")
            examples = dim_data.get('examples', [])
            for ex in examples[:2]:
                sections.append(f"    - \"{ex}\"")

        return '\n'.join(sections)

    async def generate_summary(
        self,
        analysis: Dict[str, Any],
        validate_output: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Generate narrative summary from analysis results.

        Uses anti-hallucination parameters and optional post-generation validation.

        Args:
            analysis: Complete analysis results dictionary
            validate_output: Whether to validate generated content against source

        Returns:
            Dictionary with summary and metadata, or None if generation failed
        """
        # Check if Ollama is available
        if not await self.check_ollama_available():
            logger.warning("Ollama not available for narrative generation")
            return None

        try:
            prompt = self._build_prompt(analysis)

            client = await self._get_client()

            logger.info(f"Generating narrative summary with {self.ollama_model} (anti-hallucination params)...")

            # Use anti-hallucination parameters for factual content
            response = await client.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": ANTI_HALLUCINATION_PARAMS["temperature"],
                        "num_predict": 1024,  # ~600 words
                        "top_p": ANTI_HALLUCINATION_PARAMS["top_p"],
                        "top_k": ANTI_HALLUCINATION_PARAMS["top_k"],
                        "repeat_penalty": ANTI_HALLUCINATION_PARAMS["repeat_penalty"],
                    }
                },
                timeout=self.timeout
            )

            if response.status_code != 200:
                logger.error(f"Ollama returned status {response.status_code}")
                return None

            result = response.json()
            narrative = result.get('response', '').strip()

            if not narrative:
                logger.warning("Empty response from Ollama")
                return None

            logger.info(f"Generated narrative summary: {len(narrative)} characters")

            # Extract LLM token metrics from Ollama response
            prompt_tokens = result.get('prompt_eval_count', 0)
            completion_tokens = result.get('eval_count', 0)
            eval_duration_ns = result.get('eval_duration', 0)
            llm_metrics: Dict[str, Any] = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
                'tokens_per_second': round(
                    completion_tokens / (eval_duration_ns / 1e9), 2
                ) if eval_duration_ns > 0 and completion_tokens > 0 else 0.0,
                'prompt_size_chars': len(prompt),
            }

            # Validate output if requested and module available
            validation_issues = []
            contradiction_count = 0
            if validate_output and HALLUCINATION_PREVENTION_AVAILABLE:
                from src.llm.hallucination_prevention import ContradictionDetector

                transcripts = analysis.get('transcription', [])
                narrative, validation_issues = clean_hallucinations(
                    narrative,
                    transcripts,
                    analysis,
                    add_warning=True
                )
                if validation_issues:
                    logger.warning(f"Validation found {len(validation_issues)} issues")

                # Detect contradictions
                detector = ContradictionDetector(narrative, analysis)
                contradictions = detector.detect_contradictions()
                if contradictions:
                    contradiction_count = len(contradictions)
                    logger.warning(f"Contradiction detector found {contradiction_count} contradictions")
                    for c in contradictions:
                        logger.warning(f"  - {c.description}: {c.original_text}")

            return {
                'narrative': narrative,
                'model': self.ollama_model,
                'style': self.style,
                'generated': True,
                'validation_issues': len(validation_issues) if validation_issues else 0,
                'contradictions_detected': contradiction_count,
                'llm_metrics': llm_metrics,
            }

        except httpx.TimeoutException:
            logger.error(f"Ollama request timed out after {self.timeout}s")
            return None
        except Exception as e:
            logger.error(f"Narrative generation failed: {e}")
            return None

    async def generate_summary_streaming(
        self,
        analysis: Dict[str, Any],
        progress_callback: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate narrative summary with streaming progress updates.

        Args:
            analysis: Complete analysis results dictionary
            progress_callback: Optional callback(chars_generated, is_complete) for progress

        Returns:
            Dictionary with summary and metadata, or None if generation failed
        """
        if not await self.check_ollama_available():
            logger.warning("Ollama not available for narrative generation")
            return None

        try:
            prompt = self._build_prompt(analysis)
            client = await self._get_client()

            logger.info(f"Generating narrative summary with {self.ollama_model} (streaming)...")

            if progress_callback:
                progress_callback(0, False)

            # Use streaming to show progress (with anti-hallucination params)
            async with client.stream(
                'POST',
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": ANTI_HALLUCINATION_PARAMS["temperature"],
                        "num_predict": 1024,
                        "top_p": ANTI_HALLUCINATION_PARAMS["top_p"],
                        "top_k": ANTI_HALLUCINATION_PARAMS["top_k"],
                        "repeat_penalty": ANTI_HALLUCINATION_PARAMS["repeat_penalty"],
                    }
                },
                timeout=self.timeout
            ) as response:
                if response.status_code != 200:
                    logger.error(f"Ollama returned status {response.status_code}")
                    return None

                narrative_parts = []
                chars_generated = 0

                async for line in response.aiter_lines():
                    if line:
                        try:
                            import json
                            chunk = json.loads(line)
                            text = chunk.get('response', '')
                            if text:
                                narrative_parts.append(text)
                                chars_generated += len(text)

                                if progress_callback:
                                    progress_callback(chars_generated, False)

                            # Check if done
                            if chunk.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue

                narrative = ''.join(narrative_parts).strip()

                if progress_callback:
                    progress_callback(len(narrative), True)

                if not narrative:
                    logger.warning("Empty response from Ollama")
                    return None

                logger.info(f"Generated narrative summary: {len(narrative)} characters")

                return {
                    'narrative': narrative,
                    'model': self.ollama_model,
                    'style': self.style,
                    'generated': True
                }

        except httpx.TimeoutException:
            logger.error(f"Ollama request timed out after {self.timeout}s")
            return None
        except Exception as e:
            logger.error(f"Narrative generation failed: {e}")
            return None


    def _build_story_prompt(self, analysis: Dict[str, Any]) -> str:
        """
        Build prompt for narrative nonfiction story generation.

        Produces a prompt that guides the LLM to write a genuine, compelling
        story about what actually happened — grounded in specific game details,
        real quotes, and honest moments.

        Args:
            analysis: Full analysis results

        Returns:
            Complete prompt string for story generation
        """
        context = self._build_story_context(analysis)

        prompt = f"""You are a narrative writer who transforms real game sessions into compelling true stories. Your craft is narrative nonfiction — you find the genuine drama, humor, and meaning in what actually happened.

## YOUR TASK
Write a narrative nonfiction story (800-1200 words) about this bridge crew's session. This is a TRUE STORY — your job is to tell it compellingly, not to invent drama.

## SESSION DATA

{context}

## STRUCTURE
Write 3-6 sections, each with a descriptive markdown header (## format) based on natural story beats. Headers should be specific and evocative — based on what actually happened.

GOOD headers: descriptive, specific to what actually happened in THIS session (e.g., "## The Miscalculated Jump", "## When the Shields Dropped")
BAD headers: "## Opening", "## Rising Action", "## Climax", "## Resolution"

## TONE & STYLE: NARRATIVE NONFICTION

GOOD tone: Grounded in specifics from the transcript, weaving real quotes into prose. Reads like a magazine feature about a real event.
BAD tone: Melodramatic fiction with invented details, characters, or scenarios not in the data.

IMPORTANT: Your ONLY source material is the SESSION DATA above. Do NOT use any content from these instructions as story material. Every detail, quote, and event in your story MUST come from the transcript and game data provided.

## CRITICAL RULES
1. The TRANSCRIPT is your primary source — but respect the confidence markers:
   - Lines in "quotes" = HIGH CONFIDENCE. Safe to quote verbatim in your story.
   - Lines in ~tildes~ = LOW CONFIDENCE. The words may be wrong. Paraphrase the general gist, do NOT quote these verbatim. Use hedging: "someone mentioned...", "the crew discussed...", "what sounded like..."
2. Use EXACT NUMBERS from game data only if game data is provided. If no game context is available, rely solely on transcript content.
3. DO NOT INVENT scenarios, combat, characters, names, numbers, or drama not in the data. If a detail is not in the session data, do not include it.
4. Use GENDER-NEUTRAL language (they/them, "the Captain", "the officer"). Never "he said" or "she ordered".
5. Use role names as characters (the Captain, Flight, Tactical, Operations, Sciences, Engineer).
6. Be HONEST about bugs, mistakes, confusion, miscommunication — these are often the best parts of the story.
7. Include direct quotes ONLY from high-confidence "quoted" lines. Aim for 8-12 quotes woven naturally into prose.
8. Maintain CHRONOLOGICAL order — never jump backward in time.
9. Find the REAL story in the actual transcript — the drama comes from what actually happened, not from invented events.
10. Write in PAST TENSE, third person.

## FORMAT
- 800-1200 words of flowing prose
- 3-6 sections with ## markdown headers
- Quotes woven into narrative (not listed or block-quoted)
- No bullet points or tables — this is a story

Write your narrative nonfiction story now:"""

        return prompt

    def _build_story_context(self, analysis: Dict[str, Any]) -> str:
        """
        Build context for narrative nonfiction story generation.

        Produces three sections: GAME CONTEXT (vessel/mission/variables),
        KEY GAME EVENTS (deduplicated), and CREW COMMUNICATIONS (transcripts).

        Args:
            analysis: Full analysis results

        Returns:
            Formatted context for story generation
        """
        sections = []

        # Build role map
        role_assignments = analysis.get('role_assignments', [])
        role_map: Dict[str, str] = {}
        for ra in role_assignments:
            if ra.get('speaker_id') and ra.get('role'):
                role_map[ra['speaker_id']] = ra['role']

        # === SECTION 1: GAME CONTEXT ===
        game_context = analysis.get('game_context')
        duration = analysis.get('duration_seconds', 0)
        minutes = int(duration // 60)

        sections.append("### GAME CONTEXT")
        sections.append(f"Session length: {minutes} minutes")

        if game_context:
            vessel_name = game_context.get('vessel_name')
            vessel_class = game_context.get('vessel_class')
            faction = game_context.get('faction')
            location = game_context.get('location')
            mission_name = game_context.get('mission_name')

            if vessel_name:
                vessel_str = f"Vessel: {vessel_name}"
                if vessel_class and vessel_class != vessel_name:
                    vessel_str += f" ({vessel_class} class)"
                sections.append(vessel_str)
            if faction:
                sections.append(f"Faction: {faction}")
            if location:
                sections.append(f"Location: {location}")
            if mission_name:
                sections.append(f"Mission: {mission_name}")

            # Game variables (credits, outpost counts, etc.)
            game_vars = game_context.get('game_variables', {})
            if game_vars:
                sections.append("\nGame variables:")
                for key, value in game_vars.items():
                    # Strip var_ prefix for readability
                    display_key = key.replace('var_', '')
                    sections.append(f"  {display_key}: {value}")

            # Game station roles
            game_roles = game_context.get('game_roles', [])
            if game_roles:
                sections.append(f"\nGame stations active: {', '.join(game_roles)}")

            # Objectives
            objectives = game_context.get('objectives', [])
            if objectives:
                sections.append("\nMission objectives:")
                for obj in objectives:
                    if not obj.get('visible', True):
                        continue
                    status = "COMPLETE" if obj.get('complete') else (
                        f"{obj.get('current_count', 0)}/{obj.get('total_count', 0)}"
                    )
                    sections.append(f"  - {obj['name']}: {obj.get('description', '')} [{status}]")

        # Crew roster
        speakers = analysis.get('speakers', [])
        if speakers:
            sections.append("\nCrew detected:")
            for s in speakers:
                speaker_id = s.get('speaker_id', 'Unknown')
                role = role_map.get(speaker_id, 'Crew Member')
                utterances = s.get('utterance_count', 0)
                sections.append(f"  - {role}: {utterances} communications")
        sections.append("")

        # === SECTION 2: KEY GAME EVENTS (deduplicated) ===
        telemetry_summary = analysis.get('telemetry_summary')
        if telemetry_summary and telemetry_summary.get('total_events', 0) > 0:
            sections.append("### KEY GAME EVENTS")

            # Use deduplicated story events if timeline builder is available
            key_events = telemetry_summary.get('key_events', [])
            if TELEMETRY_TIMELINE_AVAILABLE and TelemetryTimelineBuilder:
                try:
                    # Rebuild events from the raw telemetry events stored in the summary
                    # We pass key_events through the grouping logic
                    builder = TelemetryTimelineBuilder([])
                    # Manually set the grouped events using key_events
                    grouped = self._group_story_events(key_events, max_events=25)
                    for event in grouped:
                        sections.append(
                            f"  [{event['time_formatted']}] {event['description']}"
                        )
                except Exception as e:
                    logger.debug(f"Story event grouping failed, using raw events: {e}")
                    for event in key_events[:25]:
                        sections.append(
                            f"  [{event.get('time_formatted', '?')}] "
                            f"{event.get('description', 'Unknown event')}"
                        )
            else:
                for event in key_events[:25]:
                    sections.append(
                        f"  [{event.get('time_formatted', '?')}] "
                        f"{event.get('description', 'Unknown event')}"
                    )

            if len(key_events) > 25:
                sections.append(f"  ... and {len(key_events) - 25} more events")
            sections.append("")

        # === SECTION 3: CREW COMMUNICATIONS (confidence-gated) ===
        # Segments are split into two tiers:
        #   QUOTABLE (confidence >= 0.60): safe to use as direct quotes
        #   CONTEXT  (confidence < 0.60): may contain transcription errors,
        #            should be paraphrased not quoted verbatim
        QUOTE_CONFIDENCE_THRESHOLD = 0.60

        transcripts = analysis.get('transcription', [])
        if transcripts:
            MAX_TRANSCRIPT_SEGMENTS = 200
            total_count = len(transcripts)

            if total_count > MAX_TRANSCRIPT_SEGMENTS:
                # Sample: beginning (scene setting), middle (high confidence), end (resolution)
                beginning_count = 40
                end_count = 40
                middle_count = MAX_TRANSCRIPT_SEGMENTS - beginning_count - end_count

                beginning = transcripts[:beginning_count]
                ending = transcripts[-end_count:]

                middle_transcripts = transcripts[beginning_count:-end_count]
                if middle_transcripts:
                    middle_sorted = sorted(
                        [(i, t) for i, t in enumerate(middle_transcripts)
                         if t.get('text', '').strip()],
                        key=lambda x: x[1].get('confidence', 0.5),
                        reverse=True
                    )[:middle_count]
                    middle_sorted.sort(key=lambda x: x[0])
                    middle = [t for _, t in middle_sorted]
                else:
                    middle = []

                sampled_transcripts = beginning + middle + ending
                sections.append(f"### CREW COMMUNICATIONS (sampled {len(sampled_transcripts)} of {total_count})")
            else:
                sampled_transcripts = transcripts
                sections.append(f"### CREW COMMUNICATIONS ({total_count} total)")

            # Count quotable vs context-only for the LLM's awareness
            quotable_count = sum(
                1 for t in sampled_transcripts
                if t.get('confidence', 0) >= QUOTE_CONFIDENCE_THRESHOLD
                and t.get('text', '').strip()
            )
            context_count = sum(
                1 for t in sampled_transcripts
                if t.get('confidence', 0) < QUOTE_CONFIDENCE_THRESHOLD
                and t.get('text', '').strip()
            )
            sections.append(
                f"({quotable_count} high-confidence QUOTABLE, "
                f"{context_count} low-confidence CONTEXT-ONLY — see legend below)"
            )
            sections.append(
                "QUOTABLE lines (marked with \"quotes\") = safe to quote verbatim."
            )
            sections.append(
                "CONTEXT lines (marked with ~tildes~) = may have transcription errors; "
                "paraphrase the general meaning, do NOT quote verbatim."
            )
            sections.append("")

            for i, t in enumerate(sampled_transcripts):
                speaker = t.get('speaker_id', 'Unknown')
                role = role_map.get(speaker, speaker)
                text = t.get('text', '').strip()
                confidence = t.get('confidence', 0)
                timestamp = t.get('start_time', i)

                if isinstance(timestamp, (int, float)):
                    mins = int(timestamp // 60)
                    secs = int(timestamp % 60)
                    time_str = f"{mins:02d}:{secs:02d}"
                else:
                    time_str = str(timestamp)

                if text:
                    if confidence >= QUOTE_CONFIDENCE_THRESHOLD:
                        # High confidence — quotable
                        sections.append(f"  [{time_str}] {role}: \"{text}\"")
                    else:
                        # Low confidence — context only, marked with tildes
                        sections.append(f"  [{time_str}] {role}: ~{text}~")

        return '\n'.join(sections)

    def _group_story_events(
        self,
        key_events: List[Dict[str, Any]],
        max_events: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Group consecutive similar events for concise story context.

        Args:
            key_events: List of key event dictionaries from telemetry summary
            max_events: Maximum grouped events to return

        Returns:
            List of grouped event dictionaries
        """
        if not key_events:
            return []

        grouped: List[Dict[str, Any]] = []
        current_type: Optional[str] = None
        current_group: List[Dict[str, Any]] = []

        for event in key_events:
            event_type = event.get('event_type', 'unknown')
            if event_type == current_type:
                current_group.append(event)
            else:
                if current_group:
                    grouped.append(self._format_event_group(current_group))
                current_group = [event]
                current_type = event_type

        if current_group:
            grouped.append(self._format_event_group(current_group))

        return grouped[:max_events]

    def _format_event_group(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format a group of consecutive events into a single entry.

        Args:
            events: List of consecutive events of the same type

        Returns:
            Single formatted event dictionary
        """
        if len(events) == 1:
            return events[0]

        first = events[0]
        last = events[-1]
        event_type = first.get('event_type', 'unknown')
        type_label = event_type.replace('_', ' ')

        first_time = first.get('time', 0)
        last_time = last.get('time', 0)
        duration_secs = last_time - first_time

        if duration_secs > 60:
            duration_str = f"{duration_secs / 60:.0f} minutes"
        else:
            duration_str = f"{duration_secs:.0f} seconds"

        return {
            'time_formatted': f"{first.get('time_formatted', '?')}-{last.get('time_formatted', '?')}",
            'event_type': event_type,
            'category': first.get('category', ''),
            'description': (
                f"{len(events)} {type_label} events over {duration_str} "
                f"(e.g., {first.get('description', '')})"
            ),
            'time': first_time,
        }

    async def generate_story(
        self,
        analysis: Dict[str, Any],
        validate_output: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Generate story narrative from analysis results.

        Uses balanced creative parameters with post-generation validation.

        Args:
            analysis: Complete analysis results dictionary
            validate_output: Whether to validate generated content against source

        Returns:
            Dictionary with story and metadata, or None if generation failed
        """
        if not await self.check_ollama_available():
            logger.warning("Ollama not available for story generation")
            return None

        try:
            prompt = self._build_story_prompt(analysis)
            client = await self._get_client()

            logger.info(f"Generating story narrative with {self.ollama_model} (story params)...")

            # Use story-specific parameters (slightly more creative but still constrained)
            response = await client.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": STORY_PARAMS["temperature"],
                        "num_predict": STORY_PARAMS["num_predict"],
                        "top_p": STORY_PARAMS["top_p"],
                        "top_k": STORY_PARAMS["top_k"],
                        "repeat_penalty": STORY_PARAMS["repeat_penalty"],
                    }
                },
                timeout=self.timeout
            )

            if response.status_code != 200:
                logger.error(f"Ollama returned status {response.status_code}")
                return None

            result = response.json()
            story = result.get('response', '').strip()

            if not story:
                logger.warning("Empty response from Ollama")
                return None

            logger.info(f"Generated story narrative: {len(story)} characters")

            # Extract LLM token metrics from Ollama response
            prompt_tokens = result.get('prompt_eval_count', 0)
            completion_tokens = result.get('eval_count', 0)
            eval_duration_ns = result.get('eval_duration', 0)
            llm_metrics: Dict[str, Any] = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
                'tokens_per_second': round(
                    completion_tokens / (eval_duration_ns / 1e9), 2
                ) if eval_duration_ns > 0 and completion_tokens > 0 else 0.0,
                'prompt_size_chars': len(prompt),
            }

            # Validate output if requested and module available
            validation_issues = []
            if validate_output and HALLUCINATION_PREVENTION_AVAILABLE:
                transcripts = analysis.get('transcription', [])
                story, validation_issues = clean_hallucinations(
                    story,
                    transcripts,
                    analysis,
                    add_warning=True
                )
                if validation_issues:
                    logger.warning(f"Story validation found {len(validation_issues)} issues")

            return {
                'story': story,
                'model': self.ollama_model,
                'style': 'narrative',
                'generated': True,
                'validation_issues': len(validation_issues) if validation_issues else 0,
                'llm_metrics': llm_metrics,
            }

        except Exception as e:
            logger.error(f"Story generation failed: {e}")
            return None


def generate_summary_sync(analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Synchronous wrapper for narrative summary generation.

    Args:
        analysis: Complete analysis results dictionary

    Returns:
        Dictionary with summary and metadata, or None if generation failed
    """
    generator = NarrativeSummaryGenerator()

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        return loop.run_until_complete(generator.generate_summary(analysis))
    finally:
        loop.run_until_complete(generator.close())


def generate_story_sync(analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Synchronous wrapper for story narrative generation.

    Args:
        analysis: Complete analysis results dictionary

    Returns:
        Dictionary with story and metadata, or None if generation failed
    """
    generator = NarrativeSummaryGenerator()

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        return loop.run_until_complete(generator.generate_story(analysis))
    finally:
        loop.run_until_complete(generator.close())
