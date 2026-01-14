"""
Star Trek style narrative prompts for episode generation.

Specialized prompts for each scene type to generate authentic
Trek-style prose from structured scene data.
"""

from typing import Any, Dict, List, Optional

from src.narrative.scene_builder import (
    AtmosphereType,
    CaptainsLog,
    DialogueLine,
    Scene,
    SceneType,
)
from src.narrative.character_voice import CharacterArchetype, CharacterVoice


def build_captains_log_prompt(
    log: CaptainsLog,
    captain: Optional[CharacterVoice],
    mission_context: Dict[str, Any],
) -> str:
    """
    Build prompt for Captain's Log entry generation.

    Args:
        log: Captain's Log data
        captain: Captain's character voice
        mission_context: Mission details

    Returns:
        Formatted prompt string
    """
    # Determine captain style
    if captain:
        archetype = captain.archetype
        style_guide = _get_captain_style_guide(archetype)
    else:
        style_guide = _get_captain_style_guide(CharacterArchetype.THE_COMMANDER)

    log_type_context = {
        "opening": "This opens the episode. Set up the mission and hint at what's to come.",
        "mid": "This comes at a turning point. Reflect on what's happened and the challenge ahead.",
        "closing": "This concludes the episode. Reflect on lessons learned and the journey.",
    }

    prompt = f"""You are writing a Captain's Log entry for a Star Trek-style episode.

STARDATE: {log.stardate}
LOG TYPE: {log.log_type}
{log_type_context.get(log.log_type, "")}

MISSION CONTEXT:
- Mission Name: {mission_context.get('mission_name', 'Unknown')}
- Situation: {log.situation}
- Key Facts: {', '.join(log.key_facts)}

CAPTAIN'S VOICE STYLE:
{style_guide}

REQUIREMENTS:
1. Start with "Captain's Log, Stardate {log.stardate}."
2. Write 2-4 sentences maximum
3. Balance facts with the captain's personal perspective
4. {"Set up anticipation and stakes" if log.log_type == "opening" else ""}
5. {"Reflect on growth and lessons" if log.log_type == "closing" else ""}
6. Match the captain's voice style exactly
7. Do NOT use emojis

EXAMPLE FORMATS:

PICARD STYLE:
"Captain's Log, Stardate 47634.4. We have arrived at the Devolin system
on a mission of diplomacy. Yet I find myself contemplating the nature
of the task ahead—for true peace requires more than words."

KIRK STYLE:
"Captain's Log, Stardate 3192.1. The Enterprise is en route to Starbase 12,
but something feels wrong. In all my years in space, I've learned to trust
that feeling. It's rarely failed me."

PIKE STYLE:
"Captain's Log, Stardate 1739.2. I've ordered the ship to maintain position
while we assess the situation. My crew is ready. They always are. It's my
job to make sure I don't ask more of them than necessary."

Write the Captain's Log entry now:
"""
    return prompt.strip()


def build_scene_prompt(
    scene: Scene,
    characters: Dict[str, CharacterVoice],
    previous_scene_summary: str = "",
    next_scene_hint: str = "",
) -> str:
    """
    Build prompt for scene prose generation.

    Args:
        scene: Scene data
        characters: Character voice profiles
        previous_scene_summary: Brief summary of previous scene
        next_scene_hint: Hint about what follows

    Returns:
        Formatted prompt string
    """
    # Build dialogue section
    dialogue_text = _format_dialogue_for_prompt(scene.dialogue, characters)

    # Build atmosphere description
    atmosphere_desc = _get_atmosphere_description(scene.atmosphere)

    # Build character profiles for scene
    scene_characters = _get_scene_character_profiles(scene.dialogue, characters)

    # Get scene type specific instructions
    scene_instructions = _get_scene_type_instructions(scene.scene_type)

    prompt = f"""You are writing a scene for a Star Trek-style episode.

SCENE {scene.scene_number}: {scene.scene_type.value.upper()}
LOCATION: {scene.location}
LIGHTING: {scene.lighting}
ATMOSPHERE: {atmosphere_desc}
TENSION LEVEL: {scene.tension_level:.0%}
PURPOSE: {scene.purpose}

{f"PREVIOUS SCENE: {previous_scene_summary}" if previous_scene_summary else ""}
{f"LEADS TO: {next_scene_hint}" if next_scene_hint else ""}

SOUND DESIGN:
{chr(10).join(f"- {s}" for s in scene.sound_effects) if scene.sound_effects else "- Bridge ambient"}

CHARACTERS IN SCENE:
{scene_characters}

POINT OF VIEW: {scene.pov_character or "Omniscient"}

---

ACTUAL DIALOGUE (Use VERBATIM - do not modify):
{dialogue_text}

---

{scene_instructions}

WRITING RULES:
1. Use ALL provided dialogue VERBATIM in the order shown
2. Add narrative prose BETWEEN dialogue lines
3. Include physical actions, expressions, and movements
4. Add internal thoughts for the POV character
5. Describe the environment and atmosphere
6. Show character reactions through body language
7. Maintain Trek terminology (viewscreen, turbolift, etc.)
8. Keep prose tight - this should read like a TV episode novelization
9. Do NOT add new dialogue beyond what's provided
10. Do NOT use emojis

PROSE STYLE:
- Present tense for immediacy
- Mix short punchy sentences with flowing descriptions
- Show don't tell emotions
- Use sensory details (sounds, lights, vibrations)
- Include Trek-appropriate technobabble where natural

Write the scene now (400-600 words):
"""
    return prompt.strip()


def build_cold_open_prompt(
    scene: Scene,
    characters: Dict[str, CharacterVoice],
    episode_title: str,
) -> str:
    """
    Build prompt specifically for Cold Open/Teaser.

    Args:
        scene: Cold open scene data
        characters: Character voice profiles
        episode_title: Title of the episode

    Returns:
        Formatted prompt string
    """
    dialogue_text = _format_dialogue_for_prompt(scene.dialogue, characters)
    scene_characters = _get_scene_character_profiles(scene.dialogue, characters)

    prompt = f"""You are writing the COLD OPEN (Teaser) for a Star Trek episode.

EPISODE: "{episode_title}"
LOCATION: {scene.location}
LIGHTING: {scene.lighting}

The Cold Open must:
1. Hook the audience immediately
2. Establish the setting and tone
3. Introduce a hint of the conflict to come
4. End on a moment that makes viewers want to see more
5. Be followed by the title card

CHARACTERS:
{scene_characters}

AVAILABLE DIALOGUE (Use verbatim):
{dialogue_text if dialogue_text else "[No dialogue - this is a visual/atmospheric opening]"}

---

FORMAT:
Start with "FADE IN:" and establish the scene visually.
Build atmosphere before any dialogue.
End with a beat that leads into the title card.
After your prose, write: [TITLE CARD: "{episode_title}"]

STYLE NOTES:
- Cinematic opening - think establishing shots
- Introduce the ship/bridge with wonder
- Create intrigue without giving everything away
- 200-300 words before title card

Write the Cold Open now:
"""
    return prompt.strip()


def build_crisis_scene_prompt(
    scene: Scene,
    characters: Dict[str, CharacterVoice],
    crisis_details: Dict[str, Any],
) -> str:
    """
    Build prompt for high-tension crisis scenes.

    Args:
        scene: Crisis scene data
        characters: Character voice profiles
        crisis_details: Specific crisis information

    Returns:
        Formatted prompt string
    """
    dialogue_text = _format_dialogue_for_prompt(scene.dialogue, characters)
    scene_characters = _get_scene_character_profiles(scene.dialogue, characters)

    prompt = f"""You are writing the CRISIS scene - the darkest moment of the episode.

LOCATION: {scene.location}
LIGHTING: {scene.lighting} (Red alert strobing)
TENSION: MAXIMUM

CRISIS SITUATION:
- Threat: {crisis_details.get('threat', 'Ship in danger')}
- Stakes: {crisis_details.get('stakes', 'Crew survival')}
- Ship Status: {crisis_details.get('ship_status', 'Shields failing, hull damage')}

CHARACTERS:
{scene_characters}

DIALOGUE (Use EXACTLY as written):
{dialogue_text}

---

CRISIS SCENE REQUIREMENTS:
1. Open with SENSORY OVERLOAD (alarms, sparks, shaking)
2. Show crew struggling to maintain composure
3. Quick cuts between action and dialogue
4. Internal panic vs. external professionalism
5. The moment where all seems lost
6. Do NOT resolve the crisis - leave it on the edge

ATMOSPHERE PALETTE:
- Red alert lights painting faces
- Consoles exploding in showers of sparks
- The deck lurching underfoot
- Smoke curling from damaged systems
- The scream of stressed hull plating

PACING:
- Short, staccato sentences during action
- Brief moments of stillness for impact
- Build to a peak, then cut

Write the crisis scene now (400-500 words):
"""
    return prompt.strip()


def build_resolution_scene_prompt(
    scene: Scene,
    characters: Dict[str, CharacterVoice],
    outcome: str,
) -> str:
    """
    Build prompt for resolution/denouement scenes.

    Args:
        scene: Resolution scene data
        characters: Character voice profiles
        outcome: "victory", "defeat", "pyrrhic", "bittersweet"

    Returns:
        Formatted prompt string
    """
    dialogue_text = _format_dialogue_for_prompt(scene.dialogue, characters)
    scene_characters = _get_scene_character_profiles(scene.dialogue, characters)

    tone_guides = {
        "victory": "Relief and quiet triumph. The crew has earned this moment.",
        "defeat": "Somber acceptance. They did their best. It wasn't enough.",
        "pyrrhic": "Victory at great cost. Success tastes bitter.",
        "bittersweet": "They survived. They learned. The stars continue.",
    }

    prompt = f"""You are writing the RESOLUTION scene - the aftermath.

LOCATION: {scene.location}
LIGHTING: {scene.lighting} (returning to normal)
OUTCOME: {outcome.upper()}

EMOTIONAL TONE:
{tone_guides.get(outcome, tone_guides['bittersweet'])}

CHARACTERS:
{scene_characters}

DIALOGUE (Use EXACTLY as written):
{dialogue_text if dialogue_text else "[Minimal dialogue - this is a reflective moment]"}

---

RESOLUTION REQUIREMENTS:
1. Show the transition from crisis to calm
2. Allow characters a moment to breathe
3. Brief exchanges that show growth/change
4. Environmental details returning to normal
5. A sense of forward motion - the mission continues
6. Set up the closing Captain's Log

ATMOSPHERE:
- Alarms silencing
- Emergency lighting fading
- Crew exchanging looks of relief/exhaustion
- The quiet hum of systems returning to normal
- Stars visible on the viewscreen once more

Write the resolution scene now (200-300 words):
"""
    return prompt.strip()


def build_episode_title_prompt(
    mission_name: str,
    key_themes: List[str],
    outcome: str,
    beats_summary: str,
) -> str:
    """
    Build prompt for generating episode title.

    Args:
        mission_name: Original mission name
        key_themes: Identified themes from the mission
        outcome: Mission outcome
        beats_summary: Brief summary of major beats

    Returns:
        Formatted prompt string
    """
    prompt = f"""Generate a Star Trek episode title.

MISSION: {mission_name}
THEMES: {', '.join(key_themes)}
OUTCOME: {outcome}
STORY SUMMARY: {beats_summary}

GOOD TREK TITLES are:
- Evocative and slightly mysterious
- Often use metaphor or double meaning
- Sometimes reference literature, history, or philosophy
- Short (1-5 words typically)

EXAMPLES:
- "The Long Patrol" (journey/endurance)
- "Balance of Terror" (tension/opposition)
- "The Measure of a Man" (philosophical)
- "Yesterday's Enterprise" (temporal)
- "In the Pale Moonlight" (moral ambiguity)

BAD TITLES:
- Too literal: "The Mission to Mars"
- Too generic: "Space Battle"
- Too long: "When The Enterprise Fought The Romulans"

Generate 3 title options, each with a brief explanation:
"""
    return prompt.strip()


# --- Helper Functions ---


def _get_captain_style_guide(archetype: CharacterArchetype) -> str:
    """Get writing style guide for captain archetype."""
    guides = {
        CharacterArchetype.THE_COMMANDER: """
KIRK-LIKE STYLE:
- Personal and emotional
- References the human element
- Mentions crew by implication ("my ship", "we")
- Slightly dramatic, emphasizes stakes
- Trusts instinct, mentions feelings
""",
        CharacterArchetype.THE_DIPLOMAT: """
PICARD-LIKE STYLE:
- Measured and philosophical
- References history, literature, or principles
- Formal but warm
- Considers multiple perspectives
- Values diplomacy and understanding
""",
        CharacterArchetype.THE_MAVERICK: """
PIKE-LIKE STYLE:
- Protective and forward-looking
- Mentions crew members specifically
- Balances hope with realism
- Personal responsibility emphasized
- Quiet determination
""",
        CharacterArchetype.THE_ANALYST: """
JANEWAY-LIKE STYLE:
- Scientific and curious
- Problem-solving orientation
- Determined and adaptable
- Mentions exploration and discovery
- Balances logic with heart
""",
        CharacterArchetype.THE_VETERAN: """
SISKO-LIKE STYLE:
- Strategic and pragmatic
- Acknowledges difficult choices
- Mission-focused but human
- Mentions duty and sacrifice
- Quiet strength
""",
    }
    return guides.get(archetype, guides[CharacterArchetype.THE_COMMANDER])


def _format_dialogue_for_prompt(
    dialogue: List[DialogueLine],
    characters: Dict[str, CharacterVoice],
) -> str:
    """Format dialogue lines for prompt inclusion."""
    if not dialogue:
        return "[No dialogue in this scene]"

    lines = []
    for dl in dialogue:
        voice = characters.get(dl.speaker_id)
        role = voice.role if voice else dl.role
        delivery = f" {dl.delivery}" if dl.delivery else ""

        lines.append(f"{role.upper()}{delivery}: \"{dl.text}\"")

    return "\n".join(lines)


def _get_atmosphere_description(atmosphere: AtmosphereType) -> str:
    """Get prose description of atmosphere."""
    descriptions = {
        AtmosphereType.CALM: "Routine operations. The bridge hums with quiet efficiency.",
        AtmosphereType.TENSE: "Underlying tension. Crew on edge, watching displays closely.",
        AtmosphereType.URGENT: "Controlled urgency. Quick movements, sharp communications.",
        AtmosphereType.CHAOTIC: "Crisis mode. Alarms, damage, crew under extreme pressure.",
        AtmosphereType.TRIUMPHANT: "Victory achieved. Relief and quiet celebration.",
        AtmosphereType.SOMBER: "Loss weighs heavy. Quiet reflection, subdued movements.",
    }
    return descriptions.get(atmosphere, descriptions[AtmosphereType.TENSE])


def _get_scene_character_profiles(
    dialogue: List[DialogueLine],
    characters: Dict[str, CharacterVoice],
) -> str:
    """Build character profile summary for scene."""
    speakers = set(dl.speaker_id for dl in dialogue)
    profiles = []

    for speaker_id in speakers:
        voice = characters.get(speaker_id)
        if voice:
            profiles.append(
                f"- {voice.role} ({speaker_id}): {voice.get_narrative_intro()}. "
                f"{voice.voice_description}"
            )
        else:
            profiles.append(f"- {speaker_id}: Bridge officer")

    return "\n".join(profiles) if profiles else "- Bridge crew"


def _get_scene_type_instructions(scene_type: SceneType) -> str:
    """Get specific instructions for scene type."""
    instructions = {
        SceneType.COLD_OPEN: """
COLD OPEN FOCUS:
- Cinematic establishment of setting
- Mystery or intrigue introduction
- End on a hook before title card
""",
        SceneType.ESTABLISHING: """
ESTABLISHING SCENE FOCUS:
- Introduce character dynamics
- Set up mission parameters
- Show normal operations before disruption
""",
        SceneType.TACTICAL: """
TACTICAL SCENE FOCUS:
- Technical problem-solving
- Crew coordination shown
- Building tension through complications
""",
        SceneType.COMBAT: """
COMBAT SCENE FOCUS:
- Visceral action description
- Ship movements and weapons
- Crew reactions under fire
- Tactical decisions in real-time
""",
        SceneType.CRISIS: """
CRISIS SCENE FOCUS:
- Maximum tension and stakes
- Character under extreme pressure
- Difficult decisions
- Emotional peaks
""",
        SceneType.CHARACTER_MOMENT: """
CHARACTER MOMENT FOCUS:
- Quiet beat amidst action
- Reveal of character depth
- Connection between crew members
- Emotional truth
""",
        SceneType.CLIMAX: """
CLIMAX SCENE FOCUS:
- Decisive action
- Resolution of central conflict
- Peak dramatic moment
- Consequences of choices
""",
        SceneType.DENOUEMENT: """
DENOUEMENT FOCUS:
- Aftermath and reflection
- Return to equilibrium
- Character growth acknowledged
- Setup for closing log
""",
    }
    return instructions.get(scene_type, "")


def build_full_episode_assembly_prompt(
    scenes_prose: List[str],
    logs: List[str],
    episode_title: str,
    mission_summary: Dict[str, Any],
) -> str:
    """
    Build prompt for assembling full episode from parts.

    Args:
        scenes_prose: Generated prose for each scene
        logs: Generated Captain's Log entries
        episode_title: Episode title
        mission_summary: Mission statistics and outcome

    Returns:
        Formatted prompt string
    """
    prompt = f"""Assemble these scene elements into a cohesive Star Trek episode.

EPISODE TITLE: "{episode_title}"

OPENING LOG:
{logs[0] if logs else "[Generate opening log]"}

SCENES:
{chr(10).join(f"--- SCENE {i+1} ---{chr(10)}{s}" for i, s in enumerate(scenes_prose))}

CLOSING LOG:
{logs[-1] if len(logs) > 1 else "[Generate closing log]"}

MISSION STATISTICS:
- Duration: {mission_summary.get('duration', 'Unknown')}
- Outcome: {mission_summary.get('outcome', 'Unknown')}
- Objectives: {mission_summary.get('objectives_completed', '?')}/{mission_summary.get('objectives_total', '?')}

ASSEMBLY REQUIREMENTS:
1. Add smooth transitions between scenes
2. Ensure consistent character voices throughout
3. Add "[COMMERCIAL BREAK]" markers at act breaks
4. Include act headers (ACT ONE, ACT TWO, etc.)
5. Maintain pacing and tension flow
6. Polish any awkward transitions
7. Ensure opening and closing logs frame the story

Format as a complete episode script/novelization.
"""
    return prompt.strip()
