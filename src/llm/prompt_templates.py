"""
Prompt templates for LLM-powered mission analysis.

This module contains prompt engineering templates for different types of
mission summaries and analyses.
"""

import json
from typing import Dict, Any, List
from datetime import datetime


def build_mission_summary_prompt(mission_data: Dict[str, Any], style: str = "entertaining") -> str:
    """
    Build prompt for mission summary generation.

    Args:
        mission_data: Dictionary containing mission events, transcripts, etc.
        style: Summary style (entertaining, professional, technical, casual)

    Returns:
        Formatted prompt string
    """
    mission_id = mission_data.get('mission_id', 'Unknown')
    mission_name = mission_data.get('mission_name', 'Unknown Mission')
    events = mission_data.get('events', [])
    transcripts = mission_data.get('transcripts', [])

    # Extract key mission details
    mission_duration = mission_data.get('duration', 'Unknown')
    total_events = len(events)
    total_transcripts = len(transcripts)

    # Format transcripts for context
    transcript_text = format_transcripts(transcripts)

    # Extract mission objectives if available
    objectives_text = extract_objectives(events)

    # Build style-specific instructions
    style_instructions = get_style_instructions(style)

    prompt = f"""
Analyze this Starship Horizons bridge simulator mission and create a summary.

**Mission Details:**
- Mission ID: {mission_id}
- Mission Name: {mission_name}
- Duration: {mission_duration}
- Total Events: {total_events}
- Crew Communications: {total_transcripts}

**Crew Transcript:**
{transcript_text}

**Mission Objectives:**
{objectives_text}

**Style Instructions:**
{style_instructions}

**Your Task:**
Create an engaging summary of this mission that captures:
1. What happened during the mission
2. How the crew performed
3. Notable moments or amusing exchanges
4. Mission outcome and key achievements
5. Areas for improvement

Keep the summary concise (2-4 paragraphs) and {style}.
"""

    return prompt.strip()


def build_crew_analysis_prompt(transcripts: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> str:
    """
    Build prompt for crew performance analysis.

    Args:
        transcripts: List of crew communications
        events: List of mission events

    Returns:
        Formatted prompt string
    """
    transcript_text = format_transcripts(transcripts)

    # Count speakers
    speakers = set()
    for t in transcripts:
        speakers.add(t.get('speaker', 'unknown'))

    speaker_count = len(speakers)

    prompt = f"""
Analyze the crew performance from this Starship Horizons mission.

**Crew Communications ({len(transcripts)} utterances, {speaker_count} speakers):**
{transcript_text}

**Your Task:**
Provide a detailed crew performance analysis covering:

1. **Communication Patterns:**
   - Who participated most/least?
   - Quality of communication (clarity, responsiveness)
   - Turn-taking and coordination

2. **Team Dynamics:**
   - Evidence of good teamwork
   - Conflicts or confusion
   - Leadership and decision-making

3. **Professionalism:**
   - Protocol adherence
   - Tone and morale
   - Stress management

4. **Recommendations:**
   - What this crew does well
   - Areas needing improvement
   - Specific training suggestions

Provide your analysis in a structured format with clear sections and specific examples from the transcript.
"""

    return prompt.strip()


def build_full_report_prompt(mission_data: Dict[str, Any], style: str = "entertaining") -> str:
    """
    Build prompt for complete mission report generation.

    Args:
        mission_data: Complete mission data
        style: Report style

    Returns:
        Formatted prompt string
    """
    mission_id = mission_data.get('mission_id', 'Unknown')
    mission_name = mission_data.get('mission_name', 'Unknown Mission')
    events = mission_data.get('events', [])
    transcripts = mission_data.get('transcripts', [])

    transcript_text = format_transcripts(transcripts)
    objectives_text = extract_objectives(events)

    # Extract mission statistics
    stats = extract_mission_stats(mission_data)

    # Count speakers
    speakers = {}
    for t in transcripts:
        speaker = t.get('speaker', 'unknown')
        speakers[speaker] = speakers.get(speaker, 0) + 1

    speaker_stats = '\n'.join([f"- {speaker}: {count} utterances" for speaker, count in speakers.items()])

    prompt = f"""
Create a comprehensive, FACTUAL mission report for this Starship Horizons bridge simulator session.

**CRITICAL INSTRUCTIONS - READ CAREFULLY:**
⚠️ This is a FORENSIC ANALYSIS task. You are analyzing REAL DATA, not writing creative fiction.

**MANDATORY RULES:**
1. Use ONLY the exact data provided below - DO NOT invent, embellish, or assume ANYTHING
2. ALL quotes MUST be verbatim from the transcript below - DO NOT paraphrase or create dialogue
3. ALL statistics MUST be directly calculable from the data provided - DO NOT estimate
4. If you cannot find specific data, write "Data not available" - NEVER guess or infer
5. The crew transcript may NOT match the mission objectives - report what ACTUALLY happened, not what should have happened
6. Speaker names are speaker_1, speaker_2, etc. - DO NOT create character names like "Captain Zara"

**EXAMPLES OF FORBIDDEN BEHAVIOR:**
❌ WRONG: "Enemy detected at grid 09" (if this exact quote is not in the transcript)
❌ WRONG: "Speaker_1 had 45 utterances" (if the actual count is 54)
❌ WRONG: "The crew engaged in tactical combat" (if the transcript shows them talking about containers)
✅ CORRECT: Quote only actual transcript: "the caters, right?" (conf: 0.55)
✅ CORRECT: Count from data: speaker_1: 54 utterances (50.94%)
✅ CORRECT: Describe what's in the transcript, even if mundane: "Crew discussed logistics and containers"

**Mission Information:**
- Mission ID: {mission_id}
- Mission Name: {mission_name}
- Duration: {mission_data.get('duration', 'Not calculated')}
- Events Recorded: {len(events)}
- Crew Communications: {len(transcripts)}

**Speaker Statistics (ACTUAL DATA):**
{speaker_stats}

**Mission Statistics:**
{stats}

**Complete Crew Transcript (USE THESE EXACT QUOTES):**
{transcript_text}

**Mission Objectives (ACTUAL DATA):**
{objectives_text}

**Report Requirements:**

Generate a complete mission report in **Markdown format** with these sections:

# Mission Report: {mission_name}

## Executive Summary
⚠️ CRITICAL: Read the COMPLETE crew transcript below first, then write 2-3 paragraphs describing what ACTUALLY happened.

**Required approach:**
1. Read through the entire transcript to understand what the crew ACTUALLY discussed
2. The crew conversation may NOT relate to mission objectives - that's OK, describe what they actually said
3. Use ONLY information from the actual transcript quotes shown below
4. If the crew talked about mundane topics (containers, supplies, etc.), report that truthfully
5. DO NOT invent dramatic scenarios - even if mission objectives mention combat/enemies, only describe what's in the transcript
6. Use speaker_1, speaker_2, etc. - NEVER invent names like "Captain Zara"

## Mission Overview
Using ONLY the data provided:
- Mission ID: {mission_id}
- Duration: {mission_data.get('duration', 'Unknown')}
- Unique speakers detected: {len(speakers)}
- Total communications: {len(transcripts)}
- Objectives and their ACTUAL completion status from the data

## Crew Engagement Analysis

### Communication Statistics
Create a table showing ACTUAL speaker participation from the data:
| Speaker | Utterances | Percentage |
| --- | --- | --- |
(Calculate from the {len(transcripts)} transcripts above)

### Engagement Patterns
Analyze the ACTUAL transcript:
- Which speakers spoke most/least
- Response patterns between speakers
- Evidence of coordination from actual quotes

### Communication Quality
Based on the transcript confidence scores:
- Average transcription confidence: (calculate from actual data)
- Quality assessment based on actual confidence values
- Professional tone based on actual communications

## Operational Efficiency Analysis

### Mission Execution
Based on ACTUAL objectives data:
- List each objective with its REAL status (complete/incomplete)
- Progress indicators from actual data
- Timeline of key events from actual timestamps

## Team Dynamics Assessment

### Strengths
Identify strengths based on ACTUAL transcript content:
- Quote specific examples from the transcript
- Reference actual coordination events

### Areas for Improvement
Based on what's actually in the data:
- Communication gaps
- Protocol issues mentioned in transcript
- Objective completion status

### Recommended Actions
Provide recommendations based on the ACTUAL data patterns observed

## Key Communications
Extract 3-5 of the most significant or interesting ACTUAL quotes from the transcript below.
⚠️ CRITICAL: Only use VERBATIM quotes from the transcript. DO NOT create, modify, or embellish dialogue.
Format: [HH:MM:SS] speaker_X: "exact quote" (confidence: X.XX)

## Conclusions & Recommendations
Provide a final assessment based on:
- Actual mission completion status
- Real speaker engagement data
- Factual communication quality
- Observed team dynamics from transcript

---

**FINAL REMINDER:**
- DO NOT invent character names like "Captain Zara" - use speaker_1, speaker_2, etc.
- DO NOT make up statistics - calculate from actual data
- DO NOT create fictional events - use only what's in the transcript
- DO NOT guess mission duration - use the provided value: {mission_data.get('duration', 'Unknown')}
- ALL quotes must be from the actual transcript above
- ALL statistics must be calculable from the data provided
"""

    return prompt.strip()


def format_transcripts(transcripts: List[Dict[str, Any]]) -> str:
    """
    Format transcripts for inclusion in prompts.

    Args:
        transcripts: List of transcript dictionaries

    Returns:
        Formatted transcript text
    """
    if not transcripts:
        return "(No crew communications recorded)"

    lines = []
    for t in transcripts:
        timestamp = t.get('timestamp', '')
        if 'T' in timestamp:
            time_only = timestamp.split('T')[1][:8]
        else:
            time_only = timestamp

        speaker = t.get('speaker', 'unknown')
        text = t.get('text', '')
        confidence = t.get('confidence', 0.0)

        lines.append(f"[{time_only}] {speaker}: \"{text}\" (confidence: {confidence:.2f})")

    return '\n'.join(lines)


def extract_objectives(events: List[Dict[str, Any]]) -> str:
    """
    Extract mission objectives from events.

    Args:
        events: List of event dictionaries

    Returns:
        Formatted objectives text
    """
    objectives = []

    for event in events:
        event_type = event.get('event_type') or event.get('type', '')
        data = event.get('data', {})

        if event_type == 'mission_update' and 'Objectives' in data:
            obj_data = data.get('Objectives', {})
            for obj_name, obj_details in obj_data.items():
                if isinstance(obj_details, dict):
                    desc = obj_details.get('Description', obj_name)
                    complete = obj_details.get('Complete', False)
                    rank = obj_details.get('Rank', 'Unknown')
                    current = obj_details.get('CurrentCount', 0)
                    total = obj_details.get('Count', 0)

                    status = "✓ Complete" if complete else f"In Progress ({current}/{total})"
                    objectives.append(f"- [{rank}] {obj_name}: {desc} - {status}")

    if objectives:
        return '\n'.join(objectives)
    else:
        return "(No detailed objectives available)"


def extract_mission_stats(mission_data: Dict[str, Any]) -> str:
    """
    Extract key mission statistics.

    Args:
        mission_data: Mission data dictionary

    Returns:
        Formatted statistics text
    """
    stats = []

    # Duration
    if 'duration' in mission_data:
        stats.append(f"- Duration: {mission_data['duration']}")

    # Events
    events = mission_data.get('events', [])
    if events:
        stats.append(f"- Total Events: {len(events)}")

    # Transcripts
    transcripts = mission_data.get('transcripts', [])
    if transcripts:
        stats.append(f"- Crew Communications: {len(transcripts)}")

        # Speaker count
        speakers = set(t.get('speaker') for t in transcripts)
        stats.append(f"- Unique Speakers: {len(speakers)}")

    # Mission grade
    for event in events:
        if event.get('event_type') == 'mission_update':
            data = event.get('data', {})
            if 'Grade' in data:
                grade = data['Grade']
                stats.append(f"- Mission Grade: {grade:.0%}")
                break

    return '\n'.join(stats) if stats else "(No statistics available)"


def get_style_instructions(style: str) -> str:
    """
    Get style-specific instructions for report generation.

    Args:
        style: Desired style (entertaining, professional, technical, casual)

    Returns:
        Style instruction text
    """
    styles = {
        "entertaining": """
Write in an entertaining, engaging style with:
- Humorous observations about crew interactions
- Pop culture references when appropriate
- Playful commentary on mistakes or funny moments
- Vivid, narrative language
- Keep it fun and enjoyable to read while still being informative
""",
        "professional": """
Write in a professional, analytical style with:
- Formal language and structure
- Objective observations
- Metric-driven analysis
- Clear, concise conclusions
- Focus on facts and performance data
""",
        "technical": """
Write in a technical, detailed style with:
- Precise terminology
- Detailed metrics and statistics
- Technical analysis of procedures
- System-level observations
- Focus on operational details
""",
        "casual": """
Write in a casual, conversational style with:
- Friendly, approachable tone
- Simple language
- Story-like narrative
- Relatable observations
- Like you're explaining to a friend
"""
    }

    return styles.get(style, styles["entertaining"]).strip()
