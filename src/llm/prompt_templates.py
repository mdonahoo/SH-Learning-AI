"""
Prompt templates for LLM-powered mission analysis.

This module contains prompt engineering templates for different types of
mission summaries and analyses, including scientific frameworks:
- TeamSTEPPS (AHRQ team performance)
- NASA 4-D System (8 behaviors across 4 dimensions)
- Kirkpatrick Model (training evaluation)
- Bloom's Taxonomy (cognitive complexity)
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.llm.scientific_frameworks import (
    analyze_teamstepps,
    analyze_nasa_4d,
    analyze_bloom_levels,
    calculate_response_times,
    generate_kirkpatrick_assessment,
    TeamSTEPPSDomain,
    NASA4DDimension,
    BloomLevel,
    KirkpatrickLevel,
)


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
    bridge_crew = mission_data.get('bridge_crew', [])

    # NOTE: Raw transcripts are intentionally NOT included in the prompt.
    # Including them causes the LLM to "recount" speakers and override
    # the pre-computed statistics. Only selected high-confidence quotes are provided.
    objectives_text = extract_objectives(events)

    # Select top quotes by confidence for examples (max 15)
    sorted_transcripts = sorted(transcripts, key=lambda x: x.get('confidence', 0), reverse=True)
    top_quotes = sorted_transcripts[:15]
    quotes_text = format_transcripts(top_quotes) if top_quotes else "(No high-confidence quotes available)"

    # Extract mission statistics
    stats = extract_mission_stats(mission_data)

    # Count speakers with detailed stats
    speakers = {}
    for t in transcripts:
        # Support both 'speaker' (game recordings) and 'speaker_id' (audio-only)
        speaker = t.get('speaker') or t.get('speaker_id') or 'unknown'
        speakers[speaker] = speakers.get(speaker, 0) + 1

    total_utterances = len(transcripts)
    speaker_stats = '\n'.join([
        f"- {speaker}: {count} utterances ({count/total_utterances*100:.1f}%)"
        for speaker, count in sorted(speakers.items(), key=lambda x: -x[1])
    ])

    # Calculate average confidence
    confidences = [t.get('confidence', 0) for t in transcripts if t.get('confidence')]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    # Bridge crew info
    bridge_crew_text = ', '.join(bridge_crew) if bridge_crew else "Not specified"

    prompt = f"""
You are analyzing a Starship Horizons bridge simulator training session. Your goal is to provide
ACTIONABLE TRAINING INSIGHTS that will help this crew improve their performance.

## YOUR ROLE
You are a Starfleet Academy instructor reviewing a training exercise recording. Your feedback
should be constructive, specific, and focused on helping cadets become better bridge officers.

## DATA INTEGRITY RULES
1. Use ONLY verbatim quotes from the transcript - never paraphrase or invent dialogue
2. All statistics must be calculated from the actual data provided
3. Use speaker_1, speaker_2, etc. - do not assign character names
4. If data is missing, state "Data not available" - never guess

---

## MISSION DATA

**Mission:** {mission_name}
**Mission ID:** {mission_id}
**Duration:** {mission_data.get('duration', 'Unknown')}
**Bridge Stations:** {bridge_crew_text}
**Total Communications:** {total_utterances}
**Average Transcription Confidence:** {avg_confidence:.2f}

**Speaker Participation:**
{speaker_stats}

**Mission Objectives:**
{objectives_text}

**Sample Quotes (Top 15 by Confidence - USE ONLY THESE FOR QUOTING):**
{quotes_text}

**NOTE:** The above quotes are the ONLY verbatim text you may use. Do NOT invent dialogue.

---

## REPORT STRUCTURE

Generate your analysis in this exact format:

# Mission Debrief: {mission_name}

## Executive Summary
Write 2-3 paragraphs summarizing:
- What happened during the mission (based on actual transcript)
- Overall crew performance assessment
- Key training takeaways

## Role Analysis

Analyze the transcript to infer probable bridge roles based on what each speaker says:
- Look for command language ("set course", "engage", "red alert") → Captain
- Look for navigation ("course laid in", "ETA", "impulse") → Helm
- Look for combat ("targeting", "weapons", "shields", "firing") → Tactical
- Look for sensor data ("detecting", "scanning", "reading") → Science/Operations
- Look for system status ("reactor", "power levels", "damage") → Engineering
- Look for comms ("hailing", "channel open", "transmitting") → Communications

Create a table:
| Speaker | Likely Role | Key Indicators | Utterances |
| --- | --- | --- | --- |

## Command & Control Assessment

### Command Clarity
- Were orders clear and specific?
- Quote examples of good/poor command communication
- Did crew acknowledge orders?

### Response Patterns
- How quickly did crew respond to captain orders?
- Were there gaps or confusion?
- Quote examples of command-response sequences

### Decision Making
- Were decisions timely and appropriate?
- Any hesitation or missed opportunities?

## Mission Phase Analysis

Break the mission into phases based on the timeline:
1. **Pre-Launch/Status Check** - Initial readiness reports
2. **Navigation Phase** - Travel and waypoint discussions
3. **Tactical Engagement** - Combat or threat response (if any)
4. **Mission Objectives** - Hailing, scanning, docking activities
5. **Post-Action** - Damage control, status updates

For each phase, note:
- What went well
- What could improve
- Key quotes that demonstrate the assessment

## Crew Performance Scorecards

Rate each speaker on a 1-5 scale based on transcript evidence:

### [Speaker_X] - [Inferred Role]
| Metric | Score | Evidence |
| --- | --- | --- |
| Protocol Adherence | X/5 | Quote showing protocol use |
| Communication Clarity | X/5 | Quote showing clear/unclear comms |
| Response Time | X/5 | Based on command-response patterns |
| Technical Accuracy | X/5 | Based on system reports accuracy |
| Team Coordination | X/5 | Evidence of coordination |

**Strengths:** List 2-3 specific strengths with quotes
**Training Needs:** List 2-3 specific improvement areas

## Training Recommendations

### Immediate Actions (This Crew)
Provide 3-5 specific, actionable training recommendations:
1. [Specific skill] - [Why needed] - [How to practice]
2. ...

### Protocol Improvements
Based on observed gaps, recommend specific protocols to practice:
- Standard bridge callouts
- Acknowledgment procedures
- Emergency response sequences

### Team Exercises
Suggest 2-3 bridge drills that would address observed weaknesses:
1. [Drill name] - [What it practices] - [Expected outcome]

## Notable Communications

Select 5-7 quotes that best illustrate crew performance (good and needs improvement):

**Exemplary Communications:**
- [Timestamp] speaker_X: "exact quote" - Why this is good
- ...

**Communications Needing Improvement:**
- [Timestamp] speaker_X: "exact quote" - What could be better
- ...

## Mission Outcome Summary

| Objective | Status | Notes |
| --- | --- | --- |
(List each objective with completion status from data)

**Overall Mission Grade:** Based on objectives and crew performance
**Crew Readiness Assessment:** Ready for next difficulty / Needs additional training in X

---

## QUALITY CHECKLIST
Before finalizing, verify:
- [ ] All quotes are verbatim from transcript
- [ ] All statistics match provided data
- [ ] Recommendations are specific and actionable
- [ ] Each speaker has been analyzed
- [ ] Training suggestions connect to observed behaviors
"""

    return prompt.strip()


def build_scientific_analysis_prompt(
    mission_data: Dict[str, Any],
    include_frameworks: Optional[List[str]] = None
) -> str:
    """
    Build prompt using scientific teamwork and learning frameworks.

    This generates a comprehensive analysis using:
    - TeamSTEPPS (AHRQ) - Team performance observation
    - NASA 4-D System - 8 behaviors across 4 dimensions
    - Kirkpatrick Model - Training evaluation levels
    - Bloom's Taxonomy - Cognitive complexity analysis

    Args:
        mission_data: Complete mission data with events and transcripts
        include_frameworks: List of frameworks to include, or None for all

    Returns:
        Formatted prompt string with pre-computed framework metrics
    """
    if include_frameworks is None:
        include_frameworks = ["teamstepps", "nasa_4d", "kirkpatrick", "bloom"]

    mission_id = mission_data.get('mission_id', 'Unknown')
    mission_name = mission_data.get('mission_name', 'Unknown Mission')
    events = mission_data.get('events', [])
    transcripts = mission_data.get('transcripts', [])
    bridge_crew = mission_data.get('bridge_crew', [])

    # Pre-compute framework analyses
    teamstepps_results = analyze_teamstepps(transcripts) if "teamstepps" in include_frameworks else {}
    nasa_4d_results = analyze_nasa_4d(transcripts) if "nasa_4d" in include_frameworks else {}
    bloom_results = analyze_bloom_levels(transcripts) if "bloom" in include_frameworks else {}
    response_times = calculate_response_times(transcripts)

    # Extract objectives for Kirkpatrick Level 4
    objectives = _extract_objectives_list(events)
    kirkpatrick_results = generate_kirkpatrick_assessment(
        transcripts, objectives, bloom_results, teamstepps_results
    ) if "kirkpatrick" in include_frameworks else {}

    # Format pre-computed metrics
    teamstepps_text = _format_teamstepps_results(teamstepps_results)
    nasa_4d_text = _format_nasa_4d_results(nasa_4d_results)
    bloom_text = _format_bloom_results(bloom_results)
    kirkpatrick_text = _format_kirkpatrick_results(kirkpatrick_results)
    response_time_text = _format_response_times(response_times)

    # NOTE: Raw transcripts are intentionally NOT included in the prompt.
    # Including them causes the LLM to "recount" speakers and override
    # the pre-computed statistics. The LLM should only format facts, not verify them.
    # Only the pre-computed framework results and selected quotes are provided.

    # Objectives text
    objectives_text = extract_objectives(events)

    # Speaker stats
    speakers = {}
    for t in transcripts:
        speaker = t.get('speaker', 'unknown')
        speakers[speaker] = speakers.get(speaker, 0) + 1
    total_utterances = len(transcripts)
    speaker_stats = '\n'.join([
        f"- {speaker}: {count} utterances ({count/total_utterances*100:.1f}%)"
        for speaker, count in sorted(speakers.items(), key=lambda x: -x[1])
    ])

    prompt = f"""
You are a training analyst evaluating a Starship Horizons bridge simulator session using
established scientific frameworks for team performance and learning assessment.

## YOUR ROLE
You are applying research-backed evaluation frameworks used by NASA, healthcare (TeamSTEPPS),
and educational assessment (Bloom's Taxonomy, Kirkpatrick Model) to provide actionable
training insights.

## DATA INTEGRITY RULES - CRITICAL

**FORBIDDEN BEHAVIORS (will invalidate your analysis):**
- ❌ NEVER invent behaviors not shown in transcript (e.g., "crew pointed fingers" without evidence)
- ❌ NEVER claim negative behaviors occurred unless you can quote exact transcript evidence
- ❌ NEVER paraphrase or create dialogue - use ONLY verbatim quotes
- ❌ NEVER assign character names - use speaker_1, speaker_2, etc.
- ❌ NEVER recalculate statistics - use only the pre-computed values provided

**REQUIRED BEHAVIORS:**
- ✅ If a behavior was NOT observed, write: "No evidence of [behavior] found in transcript"
- ✅ Every claim about crew behavior MUST include a verbatim quote as evidence
- ✅ When data is missing or a behavior wasn't detected, state this explicitly
- ✅ Base interpretations ONLY on the pre-computed scores and transcript excerpts provided
- ✅ If transcript quality is low (confidence < 0.6), note this may affect accuracy

**EVIDENCE STANDARD:**
For any negative assessment (e.g., "communication was unclear"), you MUST:
1. Provide the exact quote showing the problem
2. Explain why it demonstrates the issue
3. If no quote exists, write "No specific evidence found" instead of inventing examples

---

## MISSION DATA

**Mission:** {mission_name}
**Mission ID:** {mission_id}
**Duration:** {mission_data.get('duration', 'Unknown')}
**Bridge Stations:** {', '.join(bridge_crew) if bridge_crew else 'Not specified'}
**Total Communications:** {total_utterances}

**Speaker Participation:**
{speaker_stats}

**Mission Objectives:**
{objectives_text}

---

## PRE-COMPUTED SCIENTIFIC ANALYSIS

### TeamSTEPPS Analysis (AHRQ Team Performance Framework)
Reference: Agency for Healthcare Research and Quality
{teamstepps_text}

### NASA 4-D System Analysis
Reference: Dr. Charlie Pellerin's team effectiveness framework
{nasa_4d_text}

### Bloom's Taxonomy Cognitive Analysis
Reference: Revised Bloom's Taxonomy (2001)
{bloom_text}

### Kirkpatrick Training Evaluation
Reference: Kirkpatrick's Four Levels of Training Evaluation
{kirkpatrick_text}

### Command-Response Timing Analysis
{response_time_text}

---

## REPORT STRUCTURE

Generate your analysis in this exact format:

# Scientific Training Analysis: {mission_name}

## Executive Summary
Write 2-3 paragraphs interpreting the pre-computed scientific metrics:
- Overall TeamSTEPPS team performance assessment
- Cognitive engagement level (Bloom's analysis)
- Training effectiveness (Kirkpatrick evaluation)
- Key strengths and development areas

## TeamSTEPPS Domain Analysis

For each of the 5 TeamSTEPPS domains, provide:

### Team Structure (Score: X/5)
- Interpretation of the pre-computed score
- Evidence from transcript examples provided
- Specific improvement recommendations

### Leadership (Score: X/5)
- Command clarity and effectiveness
- Decision-making quality
- Development recommendations

### Situation Monitoring (Score: X/5)
- Environmental awareness
- Information sharing quality
- Improvement areas

### Mutual Support (Score: X/5)
- Cross-functional assistance patterns
- Backup behaviors observed
- Enhancement opportunities

### Communication (Score: X/5)
- SBAR-style structured communication
- Call-outs and check-backs (closed-loop)
- Protocol adherence recommendations

## NASA 4-D Behavioral Assessment

Analyze each dimension based on pre-computed data:

### Cultivating Dimension (People-Building)
- Authentic appreciation behaviors
- Shared interests alignment
- Recommendations

### Visioning Dimension (Idea-Building)
- Reality-based optimism
- Inclusiveness in problem-solving
- Recommendations

### Directing Dimension (System-Building)
- Agreement-keeping (command execution)
- Outcome commitment
- Recommendations

### Including Dimension (Relationship-Building)
- Blame-free culture: If no blaming observed, state "No blaming language detected in transcript"
- Role clarity: Quote evidence of clear/unclear roles
- Recommendations based ONLY on observed behaviors

## Cognitive Engagement Analysis (Bloom's Taxonomy)

Based on the pre-computed cognitive distribution:
- Assess the cognitive complexity of crew communications
- Identify which Bloom levels are over/under-represented
- Recommend ways to elevate cognitive engagement

## Kirkpatrick Training Evaluation

### Level 2: Learning Assessment
- Interpret the cognitive level findings
- Knowledge application evidence

### Level 3: Behavior Change
- Observable skill demonstration
- TeamSTEPPS behavior integration

### Level 4: Results
- Mission objective achievement
- Performance outcome assessment

## Response Time Analysis
- Interpret the command-response timing data
- Identify any concerning delays
- Recommendations for improving crew responsiveness

## Prioritized Training Recommendations

Based on the scientific analysis, provide:

### Immediate Actions (High Priority)
1. [Specific intervention] - [Framework reference] - [Expected outcome]
2. ...

### Short-Term Development (Medium Priority)
1. [Training focus] - [Framework reference] - [Measurement approach]
2. ...

### Long-Term Growth (Lower Priority)
1. [Development area] - [Framework reference] - [Success criteria]
2. ...

## Recommended Drills and Exercises

Based on gaps identified in the frameworks:

| Drill Name | Target Framework | Skill Developed | Duration |
| --- | --- | --- | --- |
| ... | TeamSTEPPS: Communication | Closed-loop communication | 15 min |
| ... | NASA 4-D: Cultivating | Authentic appreciation | 10 min |

## Crew Readiness Assessment

**Overall Readiness Score:** X/5

**Framework Scores Summary:**
| Framework | Score | Status |
| --- | --- | --- |
| TeamSTEPPS Overall | X/5 | On Track / Needs Work |
| NASA 4-D Average | X/5 | On Track / Needs Work |
| Bloom's Cognitive Level | X/6 | Appropriate / Needs Elevation |
| Kirkpatrick Level 4 | X/5 | Achieving Results / Gaps Present |

**Recommendation:** [Ready for advancement / Needs targeted training / Recommend remediation]

---

## QUALITY CHECKLIST
Before finalizing, verify:
- [ ] All framework scores correctly interpreted from pre-computed data
- [ ] Recommendations are specific and tied to framework findings
- [ ] Transcript quotes are verbatim from provided examples
- [ ] Training suggestions are practical and measurable
- [ ] Each framework section provides actionable insights
"""

    return prompt.strip()


def _extract_objectives_list(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract objectives as a list of dictionaries."""
    objectives_map = {}

    for event in events:
        event_type = event.get('event_type') or event.get('type', '')
        data = event.get('data', {})

        if event_type == 'mission_update' and 'Objectives' in data:
            for obj_name, obj_details in data.get('Objectives', {}).items():
                if isinstance(obj_details, dict):
                    objectives_map[obj_name] = {
                        'name': obj_name,
                        'complete': obj_details.get('Complete', False),
                        'rank': obj_details.get('Rank', 'Unknown'),
                    }

    return list(objectives_map.values())


def _format_teamstepps_results(results: Dict) -> str:
    """Format TeamSTEPPS analysis results for prompt."""
    if not results:
        return "(TeamSTEPPS analysis not performed)"

    lines = []
    for domain in TeamSTEPPSDomain:
        if domain in results:
            data = results[domain]
            lines.append(f"**{domain.value.replace('_', ' ').title()}:** "
                        f"Score {data['score']}/5 | "
                        f"Frequency: {data['frequency']}% | "
                        f"Count: {data['count']}")
            if data['examples']:
                ex = data['examples'][0]
                lines.append(f"  Example: [{ex.get('timestamp', 'N/A')}] "
                           f"{ex.get('speaker', 'unknown')}: \"{ex.get('text', '')}\"")

    return '\n'.join(lines)


def _format_nasa_4d_results(results: Dict) -> str:
    """Format NASA 4-D analysis results for prompt."""
    if not results:
        return "(NASA 4-D analysis not performed)"

    lines = []
    for dimension in NASA4DDimension:
        if dimension in results:
            data = results[dimension]
            behaviors = ', '.join(data.get('behaviors', {}).keys())
            lines.append(f"**{dimension.value.title()}:** "
                        f"Score {data.get('score', 'N/A')}/5 | "
                        f"Behaviors: {behaviors or 'None observed'}")
            if data['examples']:
                ex = data['examples'][0]
                lines.append(f"  Example ({ex.get('behavior', 'N/A')}): "
                           f"{ex.get('speaker', 'unknown')}: \"{ex.get('text', '')}\"")

    return '\n'.join(lines)


def _format_bloom_results(results: Dict) -> str:
    """Format Bloom's Taxonomy analysis results for prompt."""
    if not results:
        return "(Bloom's Taxonomy analysis not performed)"

    lines = [
        f"**Average Cognitive Level:** {results.get('average_cognitive_level', 'N/A')}/6",
        f"**Total Classified Utterances:** {results.get('total_classified', 0)}",
        "",
        "**Distribution:**"
    ]

    levels_data = results.get('levels', {})
    for level in BloomLevel:
        if level in levels_data:
            data = levels_data[level]
            lines.append(f"  - {level.name}: {data['count']} ({data['percentage']}%)")

    return '\n'.join(lines)


def _format_kirkpatrick_results(results: Dict) -> str:
    """Format Kirkpatrick analysis results for prompt."""
    if not results:
        return "(Kirkpatrick analysis not performed)"

    lines = []
    for level in KirkpatrickLevel:
        if level in results:
            data = results[level]
            if data.get('available', False):
                lines.append(f"**Level {level.value} ({level.name}):** "
                           f"Score {data.get('score', 'N/A')}/5")
                if 'interpretation' in data:
                    lines.append(f"  {data['interpretation']}")
            else:
                lines.append(f"**Level {level.value} ({level.name}):** "
                           f"{data.get('note', 'Not available')}")

    return '\n'.join(lines)


def _format_response_times(results: Dict) -> str:
    """Format response time analysis for prompt."""
    if not results:
        return "(Response time analysis not performed)"

    lines = [
        f"**Command-Response Pairs Detected:** {results.get('command_response_pairs', 0)}",
        f"**Average Response Time:** {results.get('average_response_time_seconds', 0):.2f} seconds",
        f"**Min/Max Response Time:** {results.get('min_response_time_seconds', 0):.2f}s / "
        f"{results.get('max_response_time_seconds', 0):.2f}s",
    ]

    examples = results.get('examples', [])
    if examples:
        lines.append("")
        lines.append("**Example Command-Response Pairs:**")
        for ex in examples[:3]:
            cmd = ex.get('command', {})
            resp = ex.get('response', {})
            lines.append(f"  - Command: {cmd.get('speaker', 'unknown')}: \"{cmd.get('text', '')}\"")
            lines.append(f"    Response ({ex.get('response_time_seconds', 0)}s): "
                       f"{resp.get('speaker', 'unknown')}: \"{resp.get('text', '')}\"")

    return '\n'.join(lines)


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

        # Handle datetime objects
        if isinstance(timestamp, datetime):
            time_only = timestamp.strftime('%H:%M:%S')
        elif isinstance(timestamp, str) and 'T' in timestamp:
            time_only = timestamp.split('T')[1][:8]
        elif isinstance(timestamp, str):
            time_only = timestamp
        elif isinstance(timestamp, (int, float)):
            # Format seconds as MM:SS or HH:MM:SS
            total_seconds = int(timestamp)
            if total_seconds >= 3600:
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                seconds = total_seconds % 60
                time_only = f"{hours}:{minutes:02d}:{seconds:02d}"
            else:
                minutes = total_seconds // 60
                seconds = total_seconds % 60
                time_only = f"{minutes:02d}:{seconds:02d}"
        else:
            time_only = str(timestamp)

        # Support both 'speaker' (game recordings) and 'speaker_id' (audio-only)
        speaker = t.get('speaker') or t.get('speaker_id') or 'unknown'
        text = t.get('text', '')
        confidence = t.get('confidence', 0.0)

        lines.append(f"[{time_only}] {speaker}: \"{text}\" (confidence: {confidence:.2f})")

    return '\n'.join(lines)


def extract_objectives(events: List[Dict[str, Any]]) -> str:
    """
    Extract mission objectives from events (deduplicated, final state).

    Args:
        events: List of event dictionaries

    Returns:
        Formatted objectives text
    """
    # Use dict to track latest state of each objective
    objectives_map = {}

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
                    # Store latest state (overwrites previous)
                    objectives_map[obj_name] = {
                        'rank': rank,
                        'desc': desc,
                        'status': status,
                        'complete': complete
                    }

    if objectives_map:
        # Format objectives, completed first
        lines = []
        for obj_name, obj_info in sorted(
            objectives_map.items(),
            key=lambda x: (not x[1]['complete'], x[1]['rank'] != 'Primary', x[0])
        ):
            lines.append(
                f"- [{obj_info['rank']}] {obj_name}: {obj_info['desc']} - {obj_info['status']}"
            )
        return '\n'.join(lines)
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
