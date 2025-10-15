"""
Story prompt templates for mission narrative generation.

Uses the hybrid approach: Python provides facts, LLM creates narrative.
"""

from typing import Dict, Any, List


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

    prompt = f"""You are a creative writer crafting a short story based on REAL mission data.

🎯 YOUR MISSION: Write an engaging short story (1500-2000 words) about this bridge simulation mission.

📋 CRITICAL RULES:
1. Use ONLY the actual dialogue provided below - these are REAL quotes from the mission
2. You may add narrative, descriptions, internal thoughts, and action between quotes
3. DO NOT modify or paraphrase the quotes - use them VERBATIM
4. DO NOT invent new dialogue - only use provided quotes
5. **FOLLOW THE TIMELINE:** Quotes MUST appear in chronological order (Scene 1 → Scene 2 → Scene 3, etc.)
6. **NEVER jump backward or forward in time** - the story must progress linearly through the mission
7. DO NOT invent dramatic scenarios (enemy attacks, combat, etc.) unless they appear in the dialogue
8. The actual mission may be mundane (logistics, routine patrol) - that's OK! Make it interesting through character dynamics and atmosphere, not fake drama
9. You MAY describe actions, emotions, and scene details
10. Assign the speaker IDs to bridge positions as specified below
11. If the crew discussed containers and supplies, write about that - don't invent space battles
12. Stay true to what ACTUALLY happened, even if it's less exciting than sci-fi combat

---

# MISSION FACTS (USE THESE EXACTLY)

**Mission Name:** {mission_name}
**Mission ID:** {mission_id}
**Duration:** {metadata['duration']}
**Bridge Crew:** {metadata['unique_speakers']} officers

## Character Assignments (USE THESE)
{character_profiles}

## Mission Objectives (ACTUAL)
{objectives_text}

## Mission Outcome
- Objectives Completed: {objectives['completed']}/{objectives['total']}
- Total Events: {metadata['total_events']}
- Total Communications: {metadata['total_communications']}

## ACTUAL DIALOGUE SEQUENCES (USE VERBATIM)
{scene_text}

---

# STORY REQUIREMENTS

**Structure:**
1. **Opening Hook** (200-300 words)
   - Set the scene on the bridge
   - Introduce the characters and their roles
   - Establish the mission stakes

2. **Rising Action** (600-800 words)
   - Use the actual dialogue sequences above
   - Add narrative between quotes (describe actions, thoughts, reactions)
   - Build tension based on mission objectives
   - Show character dynamics and teamwork

3. **Climax** (300-400 words)
   - The critical moment of the mission
   - Use dialogue from the most intense scene
   - Show decision-making under pressure

4. **Resolution** (200-300 words)
   - Reflect the actual mission outcome
   - Character reflections on performance
   - Hint at growth/lessons learned

**Writing Style:**
- Engaging, immersive prose
- Show don't tell (use actions and dialogue)
- Create tension and atmosphere through character interactions, NOT invented combat
- Make it feel like you're on the bridge during a real training mission
- Use naval/space terminology appropriately
- Keep the actual mission duration and stakes realistic ({metadata['duration']} mission)
- Even routine missions can be interesting through character development and team dynamics
- Focus on the human element: learning, teamwork, decision-making under training conditions

**IMPORTANT - TIMELINE ADHERENCE:**
- **USE SCENES IN ORDER:** Start with Scene 1, then Scene 2, then Scene 3, etc.
- **NEVER reorder or shuffle quotes** - they are pre-sorted chronologically
- **You may skip quotes**, but if you use them, use them IN THE ORDER SHOWN
- All dialogue MUST be verbatim from the sequences above
- Add rich descriptions between dialogue
- Create believable internal thoughts for characters
- Make the story compelling while staying 100% true to the facts
- DO NOT ADD: Combat, enemies, battles, hostile forces, or danger UNLESS clearly present in dialogue
- DO ADD: Character development, team dynamics, training atmosphere, realistic bridge operations

**TIMELINE EXAMPLE:**
✅ CORRECT: Use Scene 1 quote → narrative → Scene 1 quote → narrative → Scene 2 quote → etc.
❌ WRONG: Use Scene 3 quote → then Scene 1 quote (backward jump)
❌ WRONG: Use Scene 5 quote → then Scene 2 quote (forward then back)

**Tone:** Professional but engaging - like a realistic bridge simulation training story (NOT action-packed space combat unless the dialogue supports it)

**REALITY CHECK:**
- This is a TRAINING SIMULATION, not a combat mission
- The dialogue may reference routine tasks - that's the actual story
- Make it engaging through characters and atmosphere, NOT fabricated danger

**Format:** Plain text prose, no markdown headers within the story body

---

BEGIN YOUR STORY NOW:
"""

    return prompt.strip()
