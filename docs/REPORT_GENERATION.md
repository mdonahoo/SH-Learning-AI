# Mission Report Generation Guide

## Overview

After recording a mission, you can automatically generate comprehensive reports and stories using the `generate_all_reports.py` script.

## Quick Start

### Generate All Reports for One Mission

```bash
python scripts/generate_all_reports.py GAME_20251007_214703
```

This generates **three different reports:**

1. **`mission_report_HYBRID.md`** - Training assessment with learning frameworks
2. **`mission_story_TIMELINE.md`** - Engaging narrative using real dialogue
3. **`mission_summary_FACTUAL.md`** - Pure data with zero interpretation

### Generate Specific Report Types

```bash
# Just the factual summary (fastest, no LLM needed)
python scripts/generate_all_reports.py GAME_20251007_214703 --type factual

# Just the training report
python scripts/generate_all_reports.py GAME_20251007_214703 --type hybrid

# Just the story
python scripts/generate_all_reports.py GAME_20251007_214703 --type story

# Multiple types
python scripts/generate_all_reports.py GAME_20251007_214703 --type hybrid --type story
```

### Batch Process All Missions

```bash
# Generate reports for all recorded missions
python scripts/generate_all_reports.py --batch

# Process only the 5 most recent missions
python scripts/generate_all_reports.py --batch --limit 5

# Overwrite existing reports
python scripts/generate_all_reports.py --batch --force
```

## Report Types Explained

### 1. Hybrid Training Report (`mission_report_HYBRID.md`)

**Purpose:** Comprehensive training assessment using established learning frameworks

**Contains:**
- **Kirkpatrick's 4-Level Training Model**
  - Level 1: Reaction & Engagement
  - Level 2: Learning & Knowledge Acquisition
  - Level 3: Behavior & Application
  - Level 4: Results & Mission Success

- **Bloom's Taxonomy**
  - Cognitive levels demonstrated (Remember → Create)
  - Distribution analysis

- **NASA Teamwork Framework**
  - Communication (clarity, frequency)
  - Coordination (turn-taking, collaboration)
  - Leadership (direction, authority)
  - Monitoring (situational awareness)
  - Adaptability (problem-solving)

- **Mission-Specific Metrics**
  - Duration, events, communications
  - Objective completion rates
  - Speaker participation patterns

**Best For:**
- Post-mission debriefs
- Training evaluation
- Performance assessment
- Educational analysis

**Example Output:**
```
## Learning Assessment: Kirkpatrick's Model

### Level 1: Reaction & Engagement
The engagement metrics indicate a low participation equity across the
three speakers, with an average confidence level of 0.694...

### Level 2: Learning & Knowledge Acquisition
Learning outcomes are minimal, with only one out of seven objectives
completed (14.3%)...
```

**Requirements:**
- Ollama running locally (`ollama serve`)
- Model downloaded (qwen2.5:14b recommended)
- Takes 2-4 minutes to generate

### 2. Mission Story (`mission_story_TIMELINE.md`)

**Purpose:** Engaging narrative based on actual mission events

**Contains:**
- Real dialogue (verbatim from transcripts)
- Chronological timeline (Scene 1 → 2 → 3...)
- Character development
- Narrative atmosphere
- Mission outcome

**Best For:**
- Sharing mission experiences
- Blog posts / social media
- Community engagement
- Making dry data come alive

**Example Output:**
```
The bridge of the starship _Aurora_ was a hive of activity under
the dim glow of neon lights. Captain Elara Voss sat at her console...

"The caters, right?" speaker_1 asked, immediately diving into
the logistics...

This wasn't a tale of heroic space combat. It was something more
real: a snapshot of people learning.
```

**Key Features:**
- ✅ All dialogue verbatim from actual transcripts
- ✅ Follows chronological timeline (no time jumps)
- ✅ Based on what REALLY happened (training, learning, struggles)
- ✅ No invented combat or drama
- ❌ Includes meta-moments (crew discussing simulation itself)

**Requirements:**
- Ollama running locally
- Model downloaded
- Takes 2-4 minutes to generate

### 3. Factual Summary (`mission_summary_FACTUAL.md`)

**Purpose:** Pure data with zero narrative or interpretation

**Contains:**
- Mission metadata (ID, name, date, times)
- Crew participation statistics
- Objective completion status
- Event distribution
- Communication timeline (first 10 & last 10)
- Keyword frequency analysis
- Mission outcome metrics

**Best For:**
- Quick reference
- Data analysis
- Metrics tracking
- Objective records

**Example Output:**
```
# Mission Factual Summary

**Mission ID:** GAME_20251007_214703
**Mission Name:** The Long Patrol
**Duration:** 0:10:10

## Crew Participation

**Total Participants:** 3
**Total Communications:** 106
**Average Confidence:** 69.4%

### Speaker Breakdown
- speaker_1: 54 utterances (50.9%), avg confidence 61.7%
- speaker_2: 30 utterances (28.3%), avg confidence 75.2%
- speaker_3: 22 utterances (20.8%), avg confidence 80.5%
```

**Key Features:**
- ✅ 100% factual data only
- ✅ No interpretation or narrative
- ✅ Directly calculable from raw data
- ✅ Includes actual quotes with timestamps
- ✅ No LLM needed (pure Python calculation)

**Requirements:**
- None (generated from Python, no LLM needed)
- Takes < 1 second to generate

## Usage Examples

### After Recording a Mission

```bash
# Record a mission
python scripts/record_mission_with_audio.py --duration 600

# Generate all reports for the new recording
python scripts/generate_all_reports.py --batch --limit 1
```

### Regular Workflow

```bash
# 1. Record mission
python scripts/record_mission_with_audio.py --duration 600

# 2. Wait for recording to finish
# The script will show: "Recording stopped at: game_recordings/GAME_YYYYMMDD_HHMMSS"

# 3. Generate reports
python scripts/generate_all_reports.py GAME_YYYYMMDD_HHMMSS

# 4. View reports
ls game_recordings/GAME_YYYYMMDD_HHMMSS/*.md
```

### Automated Post-Processing

Add to your recording script:

```python
# At the end of record_mission_with_audio.py
import subprocess

logger.info("Generating reports...")
subprocess.run([
    "python", "scripts/generate_all_reports.py",
    mission_id,
    "--type", "factual"  # Fast, no LLM needed
])
```

## Command Reference

### Basic Usage

```bash
python scripts/generate_all_reports.py <mission_id> [options]
```

### Options

| Option | Description |
|--------|-------------|
| `--type hybrid` | Generate training assessment report |
| `--type story` | Generate mission story narrative |
| `--type factual` | Generate factual data summary |
| `--type all` | Generate all report types (default) |
| `--force` | Overwrite existing reports |
| `--batch` | Process all missions in directory |
| `--limit N` | Process only N most recent missions |
| `--recordings-dir PATH` | Use different recordings directory |

### Examples

```bash
# Quick factual summary (no LLM, instant)
python scripts/generate_all_reports.py GAME_20251007_214703 --type factual

# Full training assessment
python scripts/generate_all_reports.py GAME_20251007_214703 --type hybrid

# Regenerate story (overwrite existing)
python scripts/generate_all_reports.py GAME_20251007_214703 --type story --force

# Batch process last 10 missions
python scripts/generate_all_reports.py --batch --limit 10

# Process all, overwrite all
python scripts/generate_all_reports.py --batch --force
```

## Configuration

### Environment Variables

```bash
# .env file
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:14b  # Recommended for quality
OLLAMA_TIMEOUT=300  # 5 minutes for story generation
ENABLE_LLM_REPORTS=true
```

### Model Requirements

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| llama3.2 | 3B | Fast | Low | ❌ Not recommended (hallucinates) |
| qwen2.5:7b | 7B | Medium | Good | Hybrid reports |
| qwen2.5:14b | 14B | Slow | Excellent | Stories & hybrid reports |

**Recommendation:** Use `qwen2.5:14b` for best results

```bash
# Download the recommended model
ollama pull qwen2.5:14b
```

## Output Files

After running the script, you'll find these files in your mission directory:

```
game_recordings/GAME_20251007_214703/
├── mission_report_HYBRID.md      # Training assessment (~5KB)
├── mission_story_TIMELINE.md     # Mission narrative (~8KB)
├── mission_summary_FACTUAL.md    # Pure data (~4KB)
├── game_events.json              # Raw events
├── transcripts.json              # Raw transcripts
├── combined_timeline.json        # Merged timeline
└── audio.wav                     # Audio recording
```

## Troubleshooting

### "Ollama server not available"

```bash
# Start Ollama
ollama serve

# In another terminal, test connection
curl http://localhost:11434/api/tags
```

### "Report generation returned empty"

This usually means timeout or prompt too long.

**Solutions:**
1. Increase timeout in `.env`: `OLLAMA_TIMEOUT=600`
2. Use smaller model: `OLLAMA_MODEL=qwen2.5:7b`
3. Check Ollama logs: `ollama ps`

### "Story has invented dialogue"

This shouldn't happen with the new prompts, but if it does:

1. Regenerate with `--force`
2. Check model: `ollama list` (should be qwen2.5:14b)
3. Verify prompt_templates.py has anti-hallucination rules

### Reports are slow to generate

**Normal timing:**
- Factual summary: < 1 second (no LLM)
- Hybrid report: 2-3 minutes (LLM)
- Mission story: 2-4 minutes (LLM)

**If slower:**
- Use smaller model: qwen2.5:7b instead of :14b
- Reduce mission length (shorter recordings = faster generation)
- Close other Ollama processes

## Best Practices

### 1. Generate Factual Summary First

Always generate the factual summary immediately after recording:

```bash
python scripts/generate_all_reports.py GAME_20251007_214703 --type factual
```

This gives you instant access to mission data without waiting for LLM.

### 2. Batch Generate Overnight

For multiple missions, run batch processing overnight:

```bash
python scripts/generate_all_reports.py --batch --force
```

### 3. Review Before Sharing

Always review LLM-generated content before sharing:
- Check quotes are accurate
- Verify statistics match factual summary
- Ensure timeline is chronological

### 4. Keep Raw Data

Never delete the original files:
- `game_events.json`
- `transcripts.json`
- `audio.wav`

Reports can always be regenerated from raw data.

## Integration with Recording

To auto-generate reports after each recording, add to `.env`:

```bash
AUTO_GENERATE_REPORTS=true
AUTO_REPORT_TYPES=factual,hybrid
```

Then update `record_mission_with_audio.py` to call the script automatically.

## See Also

- [Hybrid Reports Documentation](HYBRID_REPORTS.md) - Deep dive into training frameworks
- [Story Generation Documentation](STORY_GENERATION.md) - How story generation works
- [Mission Recording Guide](../README.md) - How to record missions

## Support

For issues or questions:
1. Check this documentation
2. Review example output in `game_recordings/`
3. Open an issue on GitHub
