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

    def _build_analysis_context(self, analysis: Dict[str, Any]) -> str:
        """
        Build context string from analysis results.

        Args:
            analysis: Full analysis results dictionary

        Returns:
            Formatted context string for LLM
        """
        sections = []

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

        # Seven Habits analysis WITH EVIDENCE
        habits = analysis.get('seven_habits', {})
        if habits:
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

        prompt = f"""You are an experienced bridge crew instructor providing a structured mission debrief.

## YOUR TASK
Generate a CONCISE, STRUCTURED debrief using bullet points and tables. NO PROSE PARAGRAPHS.

{style_instructions}

## SESSION DATA
{context}

---

## OUTPUT FORMAT (use exactly this structure)

### Mission Grade: [A/B/C/D/F]
One sentence summary based on metrics.

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

Generate the structured debrief now:"""

        return prompt

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

            # Validate output if requested and module available
            validation_issues = []
            if validate_output and HALLUCINATION_PREVENTION_AVAILABLE:
                transcripts = analysis.get('transcription', [])
                narrative, validation_issues = clean_hallucinations(
                    narrative,
                    transcripts,
                    analysis,
                    add_warning=True
                )
                if validation_issues:
                    logger.warning(f"Validation found {len(validation_issues)} issues")

            return {
                'narrative': narrative,
                'model': self.ollama_model,
                'style': self.style,
                'generated': True,
                'validation_issues': len(validation_issues) if validation_issues else 0
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
        Build prompt for story narrative generation.

        Args:
            analysis: Full analysis results

        Returns:
            Complete prompt string for story generation
        """
        context = self._build_story_context(analysis)

        prompt = f"""You are a talented storyteller who transforms bridge crew training sessions into compelling narratives. Your stories capture the drama, tension, and triumph of space operations while staying STRICTLY TRUE to the transcript.

## YOUR TASK
Write a SHORT, ENGAGING story (300-400 words) about this bridge crew's session. Transform the raw communications into a narrative that brings the mission to life.

## THE SESSION DATA

{context}

## STORY GUIDELINES

**STRUCTURE YOUR STORY:**
1. **Opening** (2-3 sentences): Set the scene based on the first communications.
2. **Rising Action** (1 paragraph): What challenges or tasks did they face? Build tension from actual events.
3. **Key Moments** (1-2 paragraphs): The heart of the story - use ACTUAL QUOTES from the transcript.
4. **Resolution** (2-3 sentences): How did the session end based on the final communications?

**CRITICAL RULES - DO NOT VIOLATE:**
- ONLY reference events, names, and details that appear IN THE TRANSCRIPT
- Use GENDER-NEUTRAL language for all crew (they/them, "the Captain", "the officer")
- DO NOT invent ship names, character names, or enemy names not in the transcript
- DO NOT make up events that didn't happen
- DO NOT assume genders - never say "he said" or "she ordered"
- If the transcript is unclear, use vague language rather than inventing specifics

**STORYTELLING RULES:**
- Write in PAST TENSE, third person ("The Captain ordered..." not "Order given")
- Use role names as characters (the Captain, the Helm Officer, the Tactical Officer)
- Weave in ACTUAL QUOTES from the transcript - these bring authenticity
- Create narrative flow - don't just list what happened
- Add atmosphere (tension in voices, urgency) but not invented details
- Find the drama in what actually happened

**TONE:**
- Engaging and cinematic, like a scene from a space opera
- Respectful of the crew's efforts
- Grounded in the actual transcript

**FORMAT:**
- 300-400 words
- Flowing prose paragraphs (not bullet points)
- Include 3-5 direct quotes from the crew woven into the narrative

Write your story now (remember: only use facts from the transcript, gender-neutral language):"""

        return prompt

    def _build_story_context(self, analysis: Dict[str, Any]) -> str:
        """
        Build context for story narrative generation.

        Args:
            analysis: Full analysis results

        Returns:
            Formatted context for story generation
        """
        sections = []

        # Duration
        duration = analysis.get('duration_seconds', 0)
        minutes = int(duration // 60)
        sections.append(f"SESSION LENGTH: {minutes} minutes")

        # Build role map
        role_assignments = analysis.get('role_assignments', [])
        role_map = {}
        for ra in role_assignments:
            if ra.get('speaker_id') and ra.get('role'):
                role_map[ra['speaker_id']] = ra['role']

        # Crew roster
        speakers = analysis.get('speakers', [])
        if speakers:
            sections.append("\nCREW ROSTER:")
            for s in speakers:
                speaker_id = s.get('speaker_id', 'Unknown')
                role = role_map.get(speaker_id, 'Crew Member')
                utterances = s.get('utterance_count', 0)
                sections.append(f"  - {role}: {utterances} communications")

        # Transcript for story context (smart sampling for long recordings)
        transcripts = analysis.get('transcription', [])
        if transcripts:
            # For long transcripts, sample strategically to preserve narrative arc
            # while keeping context size manageable
            MAX_TRANSCRIPT_SEGMENTS = 300
            total_count = len(transcripts)

            if total_count > MAX_TRANSCRIPT_SEGMENTS:
                # Smart sampling: beginning (scene setting), middle (action), end (resolution)
                beginning_count = 50  # First ~5 min for scene setting
                end_count = 50        # Last ~5 min for resolution
                middle_count = MAX_TRANSCRIPT_SEGMENTS - beginning_count - end_count

                # Get beginning and end segments
                beginning = transcripts[:beginning_count]
                ending = transcripts[-end_count:]

                # Sample evenly from middle, preferring high-confidence segments
                middle_transcripts = transcripts[beginning_count:-end_count]
                if middle_transcripts:
                    # Sort by confidence, take top ones, then re-sort by time
                    middle_sorted = sorted(
                        [(i, t) for i, t in enumerate(middle_transcripts)
                         if t.get('text', '').strip()],
                        key=lambda x: x[1].get('confidence', 0.5),
                        reverse=True
                    )[:middle_count]
                    # Re-sort by original index to maintain chronological order
                    middle_sorted.sort(key=lambda x: x[0])
                    middle = [t for _, t in middle_sorted]
                else:
                    middle = []

                sampled_transcripts = beginning + middle + ending
                sections.append(f"\nMISSION LOG (sampled {len(sampled_transcripts)} of {total_count} communications):")
                sections.append("Key communications from throughout the mission:\n")
            else:
                sampled_transcripts = transcripts
                sections.append(f"\nCOMPLETE MISSION LOG ({total_count} communications):")
                sections.append("Use these actual communications to build your story:\n")

            for i, t in enumerate(sampled_transcripts):
                speaker = t.get('speaker_id', 'Unknown')
                role = role_map.get(speaker, speaker)
                text = t.get('text', '').strip()
                timestamp = t.get('start_time', i)

                if isinstance(timestamp, (int, float)):
                    mins = int(timestamp // 60)
                    secs = int(timestamp % 60)
                    time_str = f"{mins:02d}:{secs:02d}"
                else:
                    time_str = str(timestamp)

                if text:
                    sections.append(f"  [{time_str}] {role}: \"{text}\"")

        # Mission outcome if available
        quality = analysis.get('communication_quality', {})
        if quality:
            effective_pct = quality.get('effective_percentage', 0)
            sections.append(f"\nMISSION EFFECTIVENESS: {effective_pct:.0f}%")

        habits = analysis.get('seven_habits', {})
        if habits:
            overall = habits.get('overall_score', 0)
            sections.append(f"TEAM PERFORMANCE SCORE: {overall}/5")

        return '\n'.join(sections)

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
                'validation_issues': len(validation_issues) if validation_issues else 0
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
