# Learning Evaluation Frameworks

This document explains the scientific frameworks used to analyze crew performance in Starship Horizons bridge simulator training sessions. These frameworks are implemented in two key modules:

- `src/metrics/learning_evaluator.py` - Main evaluation orchestrator
- `src/llm/scientific_frameworks.py` - Pattern-based behavior detection

---

## Table of Contents

1. [Overview](#overview)
2. [Kirkpatrick's Training Evaluation Model](#kirkpatricks-training-evaluation-model)
3. [Bloom's Taxonomy](#blooms-taxonomy)
4. [TeamSTEPPS Framework](#teamstepps-framework)
5. [NASA 4-D System](#nasa-4-d-system)
6. [How Scoring Works](#how-scoring-works)
7. [Data Flow](#data-flow)
8. [Implementation Details](#implementation-details)

---

## Overview

The evaluation system uses **regex pattern matching** against crew transcripts to detect observable behaviors. Each framework measures different aspects of team performance:

| Framework | What It Measures | Origin |
|-----------|------------------|--------|
| **Kirkpatrick** | Training effectiveness (4 levels) | Corporate training industry |
| **Bloom's Taxonomy** | Cognitive complexity of utterances | Educational psychology |
| **TeamSTEPPS** | Healthcare team performance (5 domains) | AHRQ (Agency for Healthcare Research and Quality) |
| **NASA 4-D** | Team effectiveness (4 dimensions) | NASA (Dr. Charlie Pellerin) |

All statistics are **pre-computed by Python** before being sent to the LLM. This prevents hallucination and ensures data integrity.

---

## Kirkpatrick's Training Evaluation Model

**Reference:** [Valamis - Kirkpatrick Model](https://www.valamis.com/hub/kirkpatrick-model)

Kirkpatrick's model evaluates training effectiveness at four progressive levels:

### Level 1: Reaction

**Question:** How did participants feel about the training?

**What We Measure:**
- Total communications (engagement proxy)
- Unique speakers (participation breadth)
- Participation equity score (how evenly distributed communication is)
- Average transcription confidence (proxy for clarity/comfort)

**Implementation:**
```python
# Participation equity calculation
avg_comms = total_comms / unique_speakers
std_dev = sqrt(sum((count - avg_comms)^2 for each speaker))
participation_equity = max(0, 100 - (std_dev / avg_comms * 100))
```

**Interpretation Scale:**
- `high`: > 50 total communications
- `moderate`: 20-50 communications
- `low`: < 20 communications

**Note:** True Level 1 assessment requires post-mission surveys (not available from transcripts alone).

---

### Level 2: Learning

**Question:** What knowledge/skills were acquired?

**What We Measure:**
- Objective completion rate
- Protocol adherence (use of formal language)
- Bloom's Taxonomy cognitive level distribution

**Protocol Keywords Detected:**
```python
['captain', 'aye', 'affirmative', 'negative', 'sir', 'ma\'am', 'reporting']
```

**Knowledge Level Classification:**
- `advanced`: > 30% protocol adherence
- `intermediate`: 10-30% protocol adherence
- `novice`: < 10% protocol adherence

---

### Level 3: Behavior

**Question:** Are participants applying what they learned?

**What We Measure:**
- Average response time between speakers (coordination)
- Decision-making communications
- TeamSTEPPS domain scores

**Decision Keywords Detected:**
```python
['engage', 'proceed', 'execute', 'confirm', 'negative', 'abort']
```

**Behavior Quality Classification:**
- `excellent`: < 5 second average response time
- `good`: 5-10 second average response time
- `needs_improvement`: > 10 second average response time

**Coordination Score Formula:**
```python
coordination_score = min(100, (1 / avg_response_time * 10))
```

---

### Level 4: Results

**Question:** What was the organizational impact?

**What We Measure:**
- Mission completion rate (% of objectives achieved)
- Mission grade (if available from game data)
- Critical failures (events containing 'fail' or 'destroy')

**Success Level Classification:**
- `high`: > 80% completion rate
- `moderate`: 50-80% completion rate
- `low`: < 50% completion rate

---

## Bloom's Taxonomy

**Reference:** [University of Waterloo - Bloom's Taxonomy](https://uwaterloo.ca/centre-for-teaching-excellence/catalogs/tip-sheets/blooms-taxonomy)

Bloom's Taxonomy (revised 2001) classifies cognitive complexity into six levels. We use this to measure the sophistication of crew thinking.

### The Six Levels (Lowest to Highest)

| Level | Description | Example Keywords |
|-------|-------------|------------------|
| **1. Remember** | Recall facts and basic concepts | what, where, status, report, confirm |
| **2. Understand** | Explain ideas or concepts | why, how, because, explain, means |
| **3. Apply** | Use information in new situations | execute, implement, use, operate |
| **4. Analyze** | Draw connections among ideas | compare, examine, investigate, scan |
| **5. Evaluate** | Justify a decision or course of action | decide, recommend, assess, priority |
| **6. Create** | Produce new or original work | design, plan, strategy, develop |

### Pattern Matching Implementation

Each utterance is classified by the **highest matching level**:

```python
# Scanning from CREATE down to REMEMBER
for level in reversed(list(BloomLevel)):
    for pattern in indicators.LEVEL_PATTERNS[level]:
        if re.search(pattern, text):
            highest_level = level
            break
```

### Detailed Regex Patterns

**Level 1 - Remember:**
```python
r"(?i)\b(recall|list|name|state|define|identify|repeat|label)\b"
r"(?i)\b(what is|tell me|report)\b"
```

**Level 2 - Understand:**
```python
r"(?i)\b(explain|describe|interpret|summarize|classify|compare)\b"
r"(?i)\b(shows|indicates|means|because|therefore)\b"
```

**Level 3 - Apply:**
```python
r"(?i)\b(use|execute|implement|operate|perform|apply)\b"
r"(?i)\b(firing|launching|engaging|activating|deploying)\b"
```

**Level 4 - Analyze:**
```python
r"(?i)\b(analyze|differentiate|organize|compare|contrast)\b"
r"(?i)\b(pattern|connection|relationship|cause|effect)\b"
r"(?i)\b(if.*then|because.*therefore|either.*or)\b"
```

**Level 5 - Evaluate:**
```python
r"(?i)\b(judge|assess|evaluate|recommend|decide|prioritize)\b"
r"(?i)\b(should|best|better|worse|critical|important)\b"
r"(?i)\b(i think|in my opinion|my assessment)\b"
```

**Level 6 - Create:**
```python
r"(?i)\b(create|design|develop|propose|plan|construct)\b"
r"(?i)\b(new approach|alternative|different way|modify)\b"
r"(?i)\b(what if we|could we try|how about)\b"
```

### Average Cognitive Level

The system calculates a weighted average:

```python
weighted_sum = sum(level.value * count for each level)
average_cognitive_level = weighted_sum / total_classified
```

**Interpretation:**
- >= 5.0: Exceptional - high-level evaluation and creative problem-solving
- >= 4.0: Strong - actively analyzes situations and makes informed judgments
- >= 3.0: Competent - applies knowledge effectively to mission tasks
- >= 2.0: Developing - understands procedures but needs more application practice
- < 2.0: Basic - operates at recall level; needs deeper engagement

---

## TeamSTEPPS Framework

**Reference:** [AHRQ TeamSTEPPS](https://www.ahrq.gov/teamstepps-program/resources/tools/index.html)

TeamSTEPPS (Team Strategies and Tools to Enhance Performance and Patient Safety) was developed for healthcare but applies well to any high-stakes team environment.

### The Five Core Domains

#### 1. Team Structure

**What It Measures:** Role clarity and position awareness

**Patterns Detected:**
```python
r"(?i)(captain|helm|tactical|science|engineering|operations|communications)"
r"(?i)(my station|my console|at my position)"
r"(?i)(reporting|standing by|ready|online|operational)"
r"(?i)(i have|i've got|taking|assuming)"
```

**Example Utterances:**
- "Tactical standing by"
- "Engineering reporting, all systems operational"
- "I have the helm"

---

#### 2. Leadership

**What It Measures:** Command effectiveness and crew direction

**Patterns Detected:**
```python
r"(?i)(set course|engage|execute|make it so|proceed)"
r"(?i)(red alert|yellow alert|battle stations|stand down)"
r"(?i)(all hands|attention|listen up|everyone)"
r"(?i)(i want|we need|let's|should we)"
r"(?i)(good work|well done|excellent|nice job)"
```

**Example Utterances:**
- "Set course for Starbase 12, engage"
- "Red alert, all hands to battle stations"
- "Well done, everyone"

---

#### 3. Situation Monitoring

**What It Measures:** Awareness, scanning, and status reporting

**Patterns Detected:**
```python
r"(?i)(detecting|reading|scanning|picking up|sensors show)"
r"(?i)(status|report|update|what's|how's|where)"
r"(?i)(bearing|range|distance|coordinates|heading)"
r"(?i)(enemy|hostile|threat|contact|target)"
r"(?i)(shields at|hull at|power at|\d+%)"
```

**Example Utterances:**
- "Sensors detecting three contacts at bearing 270"
- "Shields at 85%, hull integrity nominal"
- "Status report, all stations"

---

#### 4. Mutual Support

**What It Measures:** Team coordination and backup behaviors

**Patterns Detected:**
```python
r"(?i)(help|assist|support|backup|cover)"
r"(?i)(rerouting|diverting|transferring) power"
r"(?i)(i can|let me|i'll handle|got it)"
r"(?i)(watch out|be careful|heads up|warning)"
```

**Example Utterances:**
- "Rerouting power to shields"
- "I'll handle the navigation, you focus on tactical"
- "Heads up, incoming fire!"

---

#### 5. Communication

**What It Measures:** Acknowledgments, closed-loop communication, and formal protocols

**Patterns Detected:**
```python
r"(?i)\b(aye|ay|eye|acknowledged|understood|copy|roger|affirmative)\b"
r"(?i)(sir|captain|ma'am)"
r"(?i)(channel open|hailing|transmitting|receiving)"
r"(?i)(confirm|verify|repeat|say again)"
r"(?i)(negative|unable|cannot|problem)"
```

**Note:** Whisper often transcribes "aye" as "eye", "I", or "ay" - the patterns account for this.

**Example Utterances:**
- "Aye aye, Captain"
- "Confirm target coordinates"
- "Negative, shields are not responding"

---

### TeamSTEPPS Scoring

Scores are calculated on a 1-5 scale based on frequency:

```python
frequency = domain_count / total_utterances

if frequency >= 0.60: score = 5
elif frequency >= 0.40: score = 4
elif frequency >= 0.25: score = 3
elif frequency >= 0.10: score = 2
else: score = 1
```

| Frequency | Score | Interpretation |
|-----------|-------|----------------|
| >= 60% | 5 | Excellent - consistently demonstrated |
| 40-60% | 4 | Strong - frequently demonstrated |
| 25-40% | 3 | Adequate - regularly demonstrated |
| 10-25% | 2 | Developing - occasionally demonstrated |
| < 10% | 1 | Minimal - rarely demonstrated |

---

## NASA 4-D System

**Reference:** [NASA APPEL - Supporting Effective Teamwork](https://appel.nasa.gov/2018/05/09/supporting-effective-teamwork-at-nasa/)

The 4-D System was developed by Dr. Charlie Pellerin after investigating NASA mission failures. It measures four dimensions of team effectiveness.

### The Four Dimensions

#### 1. Cultivating (People-Building)

**What It Measures:** Appreciation and shared purpose

**Sub-behaviors:**

**Authentic Appreciation:**
```python
r"(?i)(good work|well done|excellent|nice|great job|thank you|thanks)"
r"(?i)(appreciate|grateful|impressed)"
```

**Shared Interests:**
```python
r"(?i)(our mission|our objective|we need to|together|team)"
r"(?i)(let's|we should|shall we)"
```

---

#### 2. Visioning (Idea-Building)

**What It Measures:** Optimism and including others' ideas

**Sub-behaviors:**

**Reality-Based Optimism:**
```python
r"(?i)(we can|possible|solution|option|alternative)"
r"(?i)(if we|what if|could we|might work)"
```

**Including Others:**
```python
r"(?i)(what do you think|any ideas|suggestions|input)"
r"(?i)(your assessment|your opinion|thoughts)"
```

---

#### 3. Directing (System-Building)

**What It Measures:** Agreement-keeping and outcome commitment

**Sub-behaviors:**

**Keeping Agreements:**
```python
r"(?i)\b(aye|ay|eye|yes sir|understood|on it|doing it now)\b"
r"(?i)(as ordered|as requested|will do)"
```

**Outcome Committed:**
```python
r"(?i)(objective|mission|goal|target|complete)"
r"(?i)(priority|critical|essential|must)"
```

---

#### 4. Including (Relationship-Building)

**What It Measures:** Blame-resistance and role clarity

**Sub-behaviors:**

**Resisting Blaming:**
```python
r"(?i)(it happens|no worries|we'll fix|don't worry)"
r"(?i)(learn from|next time|adjust)"
```

**Clarifying Roles:**
```python
r"(?i)(you handle|i'll take|your job|my responsibility)"
r"(?i)(who has|who is|which station)"
```

---

### NASA 4-D Scoring

Similar to TeamSTEPPS but with different thresholds:

```python
frequency = dimension_count / total_utterances

if frequency >= 0.30: score = 5
elif frequency >= 0.20: score = 4
elif frequency >= 0.10: score = 3
elif frequency >= 0.05: score = 2
else: score = 1
```

---

## How Scoring Works

### The General Algorithm

1. **Pattern Matching:** For each transcript utterance, regex patterns are tested
2. **Counting:** Matches are counted per domain/level/dimension
3. **Frequency Calculation:** `frequency = matches / total_utterances`
4. **Score Assignment:** Frequency is mapped to a 1-5 scale
5. **Aggregation:** Individual scores are combined into overall assessments

### Example Walkthrough

Given this transcript:
```
Speaker 1: "Helm, set course for sector 7"
Speaker 2: "Aye, Captain. Course laid in"
Speaker 1: "Tactical, what's our shield status?"
Speaker 3: "Shields at 100%, all systems nominal"
Speaker 1: "Excellent work. Engage"
```

**TeamSTEPPS Analysis:**

| Domain | Matches | Patterns Matched |
|--------|---------|------------------|
| Team Structure | 2 | "helm", "tactical" |
| Leadership | 3 | "set course", "excellent work", "engage" |
| Situation Monitoring | 2 | "status", "shields at 100%" |
| Mutual Support | 0 | (none) |
| Communication | 2 | "aye", "captain" |

**Bloom's Taxonomy:**
- "set course for sector 7" → **Apply** (execute)
- "Course laid in" → **Remember** (report)
- "what's our shield status?" → **Remember** (status query)
- "Shields at 100%, all systems nominal" → **Remember** (report)
- "Excellent work. Engage" → **Apply** (execute)

**Average Cognitive Level:** (3 + 1 + 1 + 1 + 3) / 5 = **1.8** (Developing)

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Audio Recording                              │
│                  (record_audio_only.py)                          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Transcription                                  │
│              (WhisperTranscriber)                                │
│                                                                  │
│   Output: transcripts.json                                       │
│   {speaker, text, timestamp, confidence}                         │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                 LearningEvaluator                                │
│           (src/metrics/learning_evaluator.py)                    │
│                                                                  │
│   ┌─────────────────┐  ┌─────────────────┐                      │
│   │  Kirkpatrick    │  │  Bloom's        │                      │
│   │  (4 levels)     │  │  (6 levels)     │                      │
│   └────────┬────────┘  └────────┬────────┘                      │
│            │                    │                                │
│   ┌────────┴────────┐  ┌────────┴────────┐                      │
│   │  NASA Teamwork  │  │  Mission        │                      │
│   │  (5 dimensions) │  │  Metrics        │                      │
│   └────────┬────────┘  └────────┬────────┘                      │
│            │                    │                                │
│            └────────┬───────────┘                                │
│                     │                                            │
│                     ▼                                            │
│            evaluate_all() → Structured Data                      │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              Scientific Frameworks                               │
│         (src/llm/scientific_frameworks.py)                       │
│                                                                  │
│   ┌─────────────────┐  ┌─────────────────┐                      │
│   │  TeamSTEPPS     │  │  NASA 4-D       │                      │
│   │  (5 domains)    │  │  (4 dimensions) │                      │
│   └─────────────────┘  └─────────────────┘                      │
│                                                                  │
│   Additional pattern-based analysis with                         │
│   examples stored per domain                                     │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LLM Formatting                                 │
│              (src/llm/ollama_client.py)                          │
│                                                                  │
│   LLM receives:                                                  │
│   - Pre-computed scores (NOT raw transcripts)                    │
│   - Top 10-15 highest-confidence quotes                          │
│   - Framework results with examples                              │
│                                                                  │
│   LLM role: FORMAT only, not calculate                           │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Report Output                                   │
│                                                                  │
│   - mission_summary_FACTUAL.md  (pure data)                      │
│   - mission_report_HYBRID.md    (formatted facts)                │
│   - mission_story_TIMELINE.md   (narrative)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Details

### LearningEvaluator Class

Location: `src/metrics/learning_evaluator.py`

```python
class LearningEvaluator:
    def __init__(self, events: List[Dict], transcripts: List[Dict]):
        self.events = events          # Mission events from game
        self.transcripts = transcripts # Crew communications

    def evaluate_all(self) -> Dict[str, Any]:
        return {
            'kirkpatrick': self.evaluate_kirkpatrick(),
            'blooms_taxonomy': self.evaluate_blooms_taxonomy(),
            'nasa_teamwork': self.evaluate_nasa_teamwork(),
            'mission_specific': self.evaluate_mission_metrics()
        }

    def generate_structured_report(self) -> Dict[str, Any]:
        # Combines all evaluations with top quotes
        # Ready for LLM formatting
```

### Scientific Framework Functions

Location: `src/llm/scientific_frameworks.py`

```python
def analyze_teamstepps(transcripts, indicators=None) -> Dict
def analyze_nasa_4d(transcripts, behaviors=None) -> Dict
def analyze_bloom_levels(transcripts, indicators=None) -> Dict
def calculate_response_times(transcripts, command_patterns=None) -> Dict
def generate_kirkpatrick_assessment(transcripts, objectives, bloom, teamstepps) -> Dict
```

### Key Design Decisions

1. **Pre-computation:** All statistics are calculated by Python, not the LLM
2. **Pattern-based:** Uses regex for consistent, reproducible behavior detection
3. **Quote curation:** Only top 10-15 highest-confidence quotes are sent to LLM
4. **Framework integration:** Multiple academic frameworks provide different lenses
5. **Configurable patterns:** Default patterns can be overridden via dataclass parameters

### Extending the System

To add new patterns to a framework:

```python
# Create custom indicators
custom_teamstepps = TeamSTEPPSIndicators()
custom_teamstepps.LEADERSHIP_PATTERNS.append(r"(?i)(new pattern here)")

# Use in analysis
results = analyze_teamstepps(transcripts, indicators=custom_teamstepps)
```

---

## Summary

The learning evaluation system provides **evidence-based assessment** of crew performance using established academic frameworks:

- **Kirkpatrick** tells us if training was effective
- **Bloom's** tells us how sophisticated the thinking was
- **TeamSTEPPS** tells us how well the team performed as a unit
- **NASA 4-D** tells us about team culture and relationships

All analysis is performed computationally before any LLM involvement, ensuring accuracy and preventing hallucination.
