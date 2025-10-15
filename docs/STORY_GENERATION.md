# Mission Story Generation

## Overview

Generate engaging short stories from actual mission recordings using a hybrid approach that combines real dialogue with creative narrative.

## How It Works

### Hybrid Story Architecture

```
Mission Recording
  ↓
[Python: Extract Facts]
  - Actual verbatim dialogue (106 lines)
  - Mission timeline and events
  - Speaker participation stats
  - Mission objectives and outcomes
  ↓
[LLM: Creative Narrative]
  - Weave dialogue into story
  - Add descriptions and atmosphere
  - Develop characters
  - Create engaging scenes
  - Stay true to actual events
  ↓
Short Story (1500-2000 words)
  - 100% real dialogue
  - Creative narrative between quotes
  - Realistic mission scenario
  - Character development
```

### Key Principle

**The LLM creates the narrative around actual events, not invented drama.**

## Story Guidelines

### What the LLM Can Do

✅ **Add narrative descriptions:**
- Bridge atmosphere and environment
- Character thoughts and emotions
- Physical actions and movements
- Scene transitions
- Background details

✅ **Develop characters:**
- Internal monologue
- Emotional reactions
- Professional relationships
- Team dynamics

✅ **Create atmosphere:**
- Tension through pacing
- Training simulation realism
- Professional bridge operations
- Teamwork and coordination

### What the LLM Cannot Do

❌ **Invent dialogue:**
- All quotes must be verbatim from transcripts
- Cannot paraphrase or modify actual speech
- Cannot create new conversations

❌ **Fabricate scenarios:**
- No invented combat or enemies
- No dramatic events not in the data
- No fake crisis or danger
- Must match actual mission type

❌ **Change facts:**
- Mission duration must be accurate
- Objectives must match reality
- Outcomes must be truthful

## Usage

### Generate Mission Story

```python
from src.metrics.mission_summarizer import MissionSummarizer

# Load mission data
summarizer = MissionSummarizer(mission_id, mission_name)
summarizer.load_events(events)
summarizer.load_transcripts(transcripts)

# Generate story
story = summarizer.generate_mission_story(
    output_file=Path('mission_story.md')
)
```

### Command Line

```bash
# Generate story for a recorded mission
python -c "
from src.metrics.mission_summarizer import MissionSummarizer
import json
from pathlib import Path

# Load mission
mission_dir = Path('game_recordings/GAME_20251007_214703')
with open(mission_dir / 'game_events.json') as f:
    data = json.load(f)

summarizer = MissionSummarizer(data['mission_id'], data['mission_name'])
summarizer.load_events(data['events'])

with open(mission_dir / 'transcripts.json') as f:
    summarizer.load_transcripts(json.load(f)['transcripts'])

# Generate
story = summarizer.generate_mission_story(
    output_file=mission_dir / 'mission_story.md'
)
"
```

## Story Structure

Generated stories follow this structure:

### 1. Opening Hook (200-300 words)
- Sets the scene on the bridge
- Introduces characters and their roles
- Establishes mission context
- Creates initial atmosphere

**Example:**
> The bridge of the starship _Caterpillar_ hummed with the low thrum of its systems. Captain Aria Valerian sat in her command chair, scanning tactical displays. The three seasoned veterans faced a routine patrol—checking communication stations scattered throughout their sector.

### 2. Rising Action (600-800 words)
- Weaves actual dialogue sequences
- Adds narrative between quotes
- Shows character dynamics
- Builds mission progression
- Develops team coordination

**Example:**
> "The caterers, right?" Captain Valerian asked, her tone pragmatic. The mission required meticulous attention.
>
> Dr. Petrova's voice came through crisp and clear. "We have a container incoming. What's inside?"
>
> "Clothing and supplies," Valerian replied, her confidence wavering slightly...

### 3. Climax (300-400 words)
- The critical mission moment
- Uses dialogue from intense scenes
- Shows decision-making
- Highlights teamwork under pressure

### 4. Resolution (200-300 words)
- Reflects actual mission outcome
- Character reflections
- Lessons learned
- Closure and growth

## Example Output

**Mission:** The Long Patrol (10 minutes, 106 communications)

**Story Stats:**
- Length: ~7,800 characters (~1,300 words)
- Actual quotes used: 15+ verbatim lines
- Characters: Captain, First Officer, Science Officer
- Scenario: Routine patrol with logistics (containers, shuttles, comm checks)
- Tone: Professional training simulation

**Key Features:**
- ✅ All dialogue verbatim from transcripts
- ✅ Realistic scenario (logistics, not combat)
- ✅ Character development through interactions
- ✅ Training simulation atmosphere
- ✅ Accurate mission duration and outcomes

## Quality Control

### Verification Checklist

After generating a story, verify:

1. **Dialogue Accuracy**
   ```python
   # Check all quoted dialogue exists in transcripts
   for quote in extracted_quotes:
       assert quote in [t['text'] for t in transcripts]
   ```

2. **Scenario Realism**
   - Does the story match mission objectives?
   - Are events consistent with actual timeline?
   - Is the mission type accurate (training vs combat)?

3. **No Fabrication**
   - No invented enemies or combat (unless in dialogue)
   - No dramatic scenarios not in data
   - No fake crisis or danger

### Common Issues and Fixes

**Problem:** LLM invents combat scenarios

**Solution:** Strengthen prompt with explicit rules:
- "DO NOT invent dramatic scenarios unless in dialogue"
- "This is a TRAINING SIMULATION"
- "Make it engaging through characters, not fake drama"

**Problem:** Dialogue is paraphrased

**Solution:** Emphasize verbatim requirement:
- "Use quotes EXACTLY as provided"
- "DO NOT modify or paraphrase"
- Include actual quotes in prompt with [VERBATIM] markers

## Configuration

### Environment Variables

```bash
# Use same LLM config as reports
OLLAMA_MODEL=qwen2.5:14b  # 14B model for creative writing
OLLAMA_TIMEOUT=300        # 5 minutes for story generation
```

### Temperature Settings

- **Story generation:** 0.7 (creative but controlled)
- **Report generation:** 0.3 (factual)

Higher temperature allows creative narrative while the structured prompt keeps facts locked.

## Implementation Files

### Core Components

1. **`src/llm/story_prompts.py`**
   - Story generation prompts
   - Scene structuring
   - Character assignment
   - Dialogue formatting

2. **`src/llm/ollama_client.py`**
   - `generate_mission_story()` method
   - LLM interface for creative writing

3. **`src/metrics/mission_summarizer.py`**
   - `generate_mission_story()` orchestration
   - Combines evaluator + LLM

4. **`src/metrics/learning_evaluator.py`**
   - Extracts structured mission data
   - Provides facts for story context

## Use Cases

### Training Debriefs

Generate engaging stories for post-mission debriefs:
- Makes dry mission data come alive
- Helps crew remember key moments
- Illustrates teamwork and communication
- Creates shareable training materials

### Educational Content

Create learning materials:
- Case studies for training courses
- Examples of good/bad team dynamics
- Decision-making scenarios
- Communication pattern demonstrations

### Documentation

Archive missions creatively:
- More engaging than raw logs
- Preserves human element
- Shows actual crew interactions
- Easier to review and learn from

### Entertainment

Share mission experiences:
- Blog posts or social media
- Community engagement
- Recruiting materials
- Marketing and promotion

## Best Practices

### 1. Choose Interesting Missions

Best stories come from missions with:
- ✅ Good crew banter and interaction
- ✅ Clear objectives and progression
- ✅ Some challenge or decision points
- ✅ Variety in dialogue and activities

Avoid missions with:
- ❌ Minimal communication
- ❌ Technical issues or confusion
- ❌ Incomplete or corrupt recordings

### 2. Review Before Sharing

Always review generated stories for:
- Accuracy of dialogue
- Appropriateness of content
- Realism of scenario
- Quality of narrative

### 3. Edit if Needed

Stories can be manually edited:
- Fix minor issues
- Remove repetition
- Enhance pacing
- Add section breaks

But maintain the hybrid principle: don't change facts or dialogue.

### 4. Attribute Properly

Include attribution:
```markdown
# Mission Story: [Name]
*Based on actual mission [ID] on [Date]*
*Generated using hybrid AI storytelling*
```

## Future Enhancements

### Planned Features

- [ ] Multiple story styles (noir, comedy, technical thriller)
- [ ] Character personality profiles from communication patterns
- [ ] Multi-mission story arcs (series/seasons)
- [ ] Interactive choose-your-own-adventure format
- [ ] Audio narration generation
- [ ] Visual scene generation (images/video)

### Advanced Options

- [ ] Focus on specific characters/roles
- [ ] Emphasize particular themes (leadership, crisis, teamwork)
- [ ] Adjust length (flash fiction to novella)
- [ ] Multiple perspectives (different POVs)
- [ ] Genre adaptation (mil-sci-fi, procedural, character study)

## Examples

### Routine Patrol Story

**Input:**
- Duration: 10 minutes
- Comms: 106 lines
- Type: Logistics and station checks

**Output:**
- ~1,300 word story
- Focus: Team coordination during routine tasks
- Tone: Professional training atmosphere
- Drama: Character interactions, not combat

### Combat Mission Story

**Input:**
- Duration: 25 minutes
- Comms: 250 lines
- Type: Tactical engagement

**Output:**
- ~2,000 word story
- Focus: Decision-making under pressure
- Tone: Intense but realistic
- Drama: Actual combat from dialogue

## License

Part of the Starship Horizons Learning AI project.
