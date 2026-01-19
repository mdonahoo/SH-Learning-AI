"""
LLM-powered narrative summary generator for audio analysis.

Generates engaging, insightful narrative summaries of bridge crew
communication sessions using Ollama models.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5:14b-instruct')
OLLAMA_TIMEOUT = float(os.getenv('OLLAMA_TIMEOUT', '600'))  # 10 minutes for large models
LLM_REPORT_STYLE = os.getenv('LLM_REPORT_STYLE', 'entertaining')


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

            # Also include the full transcript for context
            sections.append(f"\nFULL TRANSCRIPT ({len(transcripts)} utterances):")
            for t in transcripts[:50]:  # First 50 utterances for context
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

        prompt = f"""You are Dr. Elena Vasquez, a renowned organizational psychologist and leadership expert who has spent 20 years studying high-performing teams in aerospace, military, and emergency response settings. You've been invited to observe this bridge crew training session and provide your expert assessment.

## YOUR EXPERTISE
- Team dynamics and group psychology
- Leadership communication patterns
- Crew resource management (CRM)
- The 7 Habits of Highly Effective People framework
- Positive psychology and strengths-based feedback

## YOUR TASK
Write a SHORT, FOCUSED narrative (250-350 words) analyzing this crew's teamwork and leadership dynamics. You are an encouraging but honest expert who celebrates what teams do well while noting growth opportunities.

{style_instructions}

## SESSION DATA

{context}

---

## NARRATIVE STRUCTURE

Write your expert observation as a cohesive narrative (NOT bullet points) covering:

**1. Team Dynamics Snapshot** (1 paragraph)
What kind of team are you observing? Describe the overall team energy and interaction style. How do they coordinate? Is there a clear command structure? How do members support each other?

**2. Leadership & Communication Highlights** (1-2 paragraphs)
Celebrate specific examples of GOOD teamwork you observed. Quote actual communications that demonstrate:
- Clear command and acknowledgment
- Proactive information sharing
- Supporting teammates
- Professional bridge protocol
- Any moments of synergy or effective coordination

Be specific! Use actual quotes from the transcript as evidence. Frame these as "I observed..." or "A great example was when..."

**3. Growth Edge** (1 short paragraph)
Based on the Seven Habits scores and communication patterns, identify ONE specific area where this crew could level up. Frame it positively as their "growth edge" - the next skill that would take them from good to great. Be encouraging, not critical.

## CRITICAL RULES - YOU MUST FOLLOW THESE

1. **EVIDENCE REQUIRED**: Every claim must be backed by a direct quote from the transcript.
   - BAD: "The Captain showed strong leadership"
   - GOOD: "The Captain showed strong leadership when they said: 'All stations report status'"

2. **USE THE DATA**: Reference specific scores and metrics provided above.
   - BAD: "Communication was generally good"
   - GOOD: "With 75% effective communications and strong closed-loop patterns, this crew..."

3. **QUOTE VERBATIM**: Use exact quotes from the KEY COMMUNICATIONS and EVIDENCE sections.
   - Put quotes in quotation marks
   - Attribute to the role (Captain, Helm, etc.)

4. **NO INVENTION**: If you can't find evidence for something, don't claim it happened.

5. **ROLE NAMES ONLY**: Use Captain, Helm, Tactical, Science, Engineering, Comms - NOT speaker IDs

6. **KEEP IT SHORT**: 250-350 words maximum. Every sentence must add value.

7. **BALANCE**: ~70% celebrating strengths with evidence, ~30% growth opportunities

## OUTPUT FORMAT

Write a flowing narrative (NOT bullet points) that weaves together:
- What you observed (with quoted evidence)
- What the team did well (with specific examples)
- One growth opportunity (grounded in the data)

Write your evidence-based team dynamics narrative now:"""

        return prompt

    async def generate_summary(
        self,
        analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate narrative summary from analysis results.

        Args:
            analysis: Complete analysis results dictionary

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

            logger.info(f"Generating narrative summary with {self.ollama_model}...")

            response = await client.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 1024,  # ~600 words
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

            # Use streaming to show progress
            async with client.stream(
                'POST',
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 1024,
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

        prompt = f"""You are a talented storyteller who transforms bridge crew training sessions into compelling narratives. Your stories capture the drama, tension, and triumph of space operations while staying true to what actually happened.

## YOUR TASK
Write a SHORT, ENGAGING story (300-400 words) about this bridge crew's session. Transform the raw communications into a narrative that brings the mission to life.

## THE SESSION DATA

{context}

## STORY GUIDELINES

**STRUCTURE YOUR STORY:**
1. **Opening** (2-3 sentences): Set the scene. What was the situation when we join the crew?
2. **Rising Action** (1 paragraph): What challenges or tasks did they face? Build some tension.
3. **Key Moments** (1-2 paragraphs): The heart of the story - what happened? Use actual quotes from the crew to bring authenticity.
4. **Resolution** (2-3 sentences): How did it end? What was accomplished?

**STORYTELLING RULES:**
- Write in PAST TENSE, third person ("The Captain ordered..." not "Order given")
- Use role names as characters (Captain, Helm Officer, Tactical Officer, Science Officer, Engineer)
- Weave in ACTUAL QUOTES from the transcript - these bring authenticity
- Create narrative flow - don't just list what happened
- Add sensory details and atmosphere (the hum of the bridge, tension in voices)
- Find the drama - even routine operations have moments of focus and teamwork
- Keep it grounded in what actually happened - don't invent major events

**TONE:**
- Engaging and cinematic, like a scene from a space opera
- Respectful of the crew's efforts
- Find the heroic in the everyday

**FORMAT:**
- 300-400 words
- Flowing prose paragraphs (not bullet points)
- Include 3-5 direct quotes from the crew woven into the narrative

Write your story now:"""

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

        # Full transcript for story context
        transcripts = analysis.get('transcription', [])
        if transcripts:
            sections.append(f"\nCOMPLETE MISSION LOG ({len(transcripts)} communications):")
            sections.append("Use these actual communications to build your story:\n")

            for i, t in enumerate(transcripts):
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
        analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate story narrative from analysis results.

        Args:
            analysis: Complete analysis results dictionary

        Returns:
            Dictionary with story and metadata, or None if generation failed
        """
        if not await self.check_ollama_available():
            logger.warning("Ollama not available for story generation")
            return None

        try:
            prompt = self._build_story_prompt(analysis)
            client = await self._get_client()

            logger.info(f"Generating story narrative with {self.ollama_model}...")

            response = await client.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,  # Slightly more creative for storytelling
                        "num_predict": 1200,  # ~400 words
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

            return {
                'story': story,
                'model': self.ollama_model,
                'style': 'narrative',
                'generated': True
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
