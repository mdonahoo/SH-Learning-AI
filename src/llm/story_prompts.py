"""
Story prompt templates for mission narrative generation.

Uses the hybrid approach: Python provides facts, LLM creates narrative.
"""

from typing import Dict, Any, List, Optional


def _build_story_telemetry_section(telemetry_summary: Optional[Dict[str, Any]]) -> str:
    """
    Build telemetry section for story prompts.

    Groups consecutive similar events for concise presentation.

    Args:
        telemetry_summary: Telemetry summary data

    Returns:
        Formatted telemetry section for story prompt
    """
    if not telemetry_summary or telemetry_summary.get('total_events', 0) == 0:
        return ""

    sections = [
        "## GAME EVENTS (actual in-game data — use specific numbers and names)",
        f"**Total events recorded:** {telemetry_summary.get('total_events', 0)}",
        "",
    ]

    # Mission phases for story structure
    phases = telemetry_summary.get('phases', [])
    if phases:
        sections.append("**Mission phases:**")
        for phase in phases:
            sections.append(
                f"- {phase.get('start_formatted', '?')}-{phase.get('end_formatted', '?')}: "
                f"{phase.get('display_name', 'Unknown')}"
            )
        sections.append("")

    # Key events — group consecutive same-type events
    key_events = telemetry_summary.get('key_events', [])
    if key_events:
        grouped = _group_consecutive_events(key_events, max_events=25)
        sections.append("**Key game events (weave these into the narrative with specific details):**")
        for event in grouped:
            sections.append(
                f"- [{event['time_formatted']}] {event['description']}"
            )
        sections.append("")

    return '\n'.join(sections)


def _group_consecutive_events(
    events: List[Dict[str, Any]],
    max_events: int = 25
) -> List[Dict[str, Any]]:
    """
    Group consecutive events of the same type into summary entries.

    Args:
        events: List of key event dictionaries
        max_events: Maximum grouped events to return

    Returns:
        List of grouped event dictionaries
    """
    if not events:
        return []

    grouped: List[Dict[str, Any]] = []
    current_type: Optional[str] = None
    current_group: List[Dict[str, Any]] = []

    for event in events:
        event_type = event.get('event_type', 'unknown')
        if event_type == current_type:
            current_group.append(event)
        else:
            if current_group:
                grouped.append(_format_event_group(current_group))
            current_group = [event]
            current_type = event_type

    if current_group:
        grouped.append(_format_event_group(current_group))

    return grouped[:max_events]


def _format_event_group(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Format a group of consecutive same-type events into one entry.

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
        'description': (
            f"{len(events)} {type_label} events over {duration_str} "
            f"(e.g., {first.get('description', '')})"
        ),
    }


def build_mission_story_prompt(structured_data: Dict[str, Any]) -> str:
    """
    Build prompt for creative mission story generation.

    Args:
        structured_data: Pre-computed facts from LearningEvaluator

    Returns:
        Formatted prompt string
    """
    metadata = structured_data['metadata']
    speaker_stats = structured_data['speaker_statistics']
    objectives = structured_data['objectives']
    transcripts = structured_data['raw_transcripts']

    # Extract mission name and ID
    mission_name = structured_data.get('mission_name', 'Unknown Mission')
    mission_id = structured_data.get('mission_id', 'UNKNOWN')

    # Format speaker roles (assign bridge positions based on participation)
    speaker_roles = []
    for i, stat in enumerate(speaker_stats):
        if i == 0:  # Most active speaker = Captain
            role = "Captain"
        elif i == 1:  # Second = First Officer/XO
            role = "First Officer"
        elif i == 2:  # Third = Science/Ops
            role = "Science Officer"
        else:
            role = f"Officer {i+1}"

        speaker_roles.append({
            'speaker_id': stat['speaker'],
            'role': role,
            'utterances': stat['utterances'],
            'percentage': stat['percentage']
        })

    # Format objectives with status
    obj_list = []
    for obj_name, obj_data in objectives['details'].items():
        status = "COMPLETE" if obj_data['complete'] else f"INCOMPLETE ({obj_data['current_count']}/{obj_data['total_count']})"
        obj_list.append(f"- {obj_name}: {obj_data['description']} [{status}]")
    objectives_text = "\n".join(obj_list) if obj_list else "No specific objectives recorded"

    # Group transcripts into dialogue sequences (chronological scenes)
    # CRITICAL: Scenes must remain in chronological order
    scenes = []
    current_scene = []
    last_time = None

    for t in transcripts:
        from datetime import datetime
        try:
            timestamp = datetime.fromisoformat(t['timestamp'])

            # New scene if gap > 30 seconds
            if last_time and (timestamp - last_time).total_seconds() > 30:
                if current_scene:
                    scenes.append(current_scene)
                current_scene = []

            current_scene.append(t)
            last_time = timestamp
        except:
            current_scene.append(t)

    if current_scene:
        scenes.append(current_scene)

    # Format scenes with timestamps
    # IMPORTANT: Scenes are already in chronological order - DO NOT REORDER
    scene_text = "⚠️ **THESE SCENES ARE IN CHRONOLOGICAL ORDER - USE THEM IN THIS EXACT ORDER** ⚠️\n\n"

    # Include first 8 scenes (or fewer if mission is short)
    max_scenes = min(8, len(scenes))

    for i, scene in enumerate(scenes[:max_scenes]):
        scene_text += f"\n**Scene {i+1}/{len(scenes)}** ({len(scene)} exchanges, chronologically sorted):\n"
        # Show key dialogue from this scene
        for t in scene[:12]:  # 12 lines per scene
            timestamp = t['timestamp'].split('T')[1][:8] if 'T' in t['timestamp'] else t['timestamp']
            speaker = t['speaker']
            text = t['text']
            conf = t['confidence']
            scene_text += f"[{timestamp}] {speaker}: \"{text}\" (confidence: {conf:.2f})\n"

        if len(scene) > 12:
            scene_text += f"... ({len(scene) - 12} more exchanges in this scene)\n"

    if len(scenes) > max_scenes:
        scene_text += f"\n... ({len(scenes) - max_scenes} additional scenes follow chronologically)\n"

    # Create character profiles
    character_profiles = "\n".join([
        f"**{role['role']}** ({role['speaker_id']}): {role['utterances']} communications ({role['percentage']}% of total)"
        for role in speaker_roles
    ])

    prompt = f"""You are a narrative writer who transforms real game sessions into compelling true stories. Your craft is narrative nonfiction — you find the genuine drama, humor, and meaning in what actually happened.

## YOUR TASK
Write a narrative nonfiction story (800-1200 words) about this bridge crew's mission. This is a TRUE STORY — your job is to tell it compellingly, not to invent drama.

---

# MISSION FACTS

**Mission Name:** {mission_name}
**Mission ID:** {mission_id}
**Duration:** {metadata['duration']}
**Bridge Crew:** {metadata['unique_speakers']} officers

## Crew
{character_profiles}

## Mission Objectives
{objectives_text}

## Mission Outcome
- Objectives Completed: {objectives['completed']}/{objectives['total']}
- Total Events: {metadata['total_events']}
- Total Communications: {metadata['total_communications']}

## CREW DIALOGUE (chronological — use verbatim)
{scene_text}

{_build_story_telemetry_section(structured_data.get('telemetry_summary'))}

---

## STRUCTURE
Write 3-6 sections, each with a descriptive markdown header (## format) based on natural story beats. Headers should be specific and evocative.

GOOD headers: "## The Turret That Turned", "## Racing to Outpost D-3", "## A Question of Credits"
BAD headers: "## Opening", "## Rising Action", "## Climax"

## TONE & STYLE: NARRATIVE NONFICTION

GOOD example:
"The team started with 620 credits and a clear strategy. 'Taking these first three is gonna be good,' someone noted. What they didn't realize was that every outpost they captured would drain credits from a shared pool."

BAD example:
"Captain Rodriguez surveyed the bridge with steely determination as alarms blared across the command deck."

## CRITICAL RULES
1. The TRANSCRIPT is your primary source. Use exact quotes woven naturally into prose.
2. Use EXACT NUMBERS from game data: credits, outpost names, objective counts.
3. DO NOT INVENT scenarios, combat, characters, or drama not in the data.
4. Use GENDER-NEUTRAL language (they/them). Never "he said" or "she ordered".
5. Maintain CHRONOLOGICAL order — scenes must flow Scene 1 → 2 → 3.
6. Be HONEST about bugs, mistakes, confusion — these are often the best parts.
7. Include 10+ direct quotes woven naturally into prose.
8. Write in PAST TENSE, third person.
9. Find the REAL story in what happened. Don't fabricate danger or drama.

## FORMAT
- 800-1200 words of flowing prose
- 3-6 sections with ## markdown headers
- Quotes woven into narrative, not block-quoted
- No bullet points or tables — this is a story

Write your narrative nonfiction story now:
"""

    return prompt.strip()
