"""
Hybrid prompt templates for LLM narrative formatting.

This module contains prompts that take PRE-COMPUTED facts and data,
asking the LLM ONLY to format them into readable narratives.
The LLM is NOT allowed to calculate, invent, or modify any facts.
"""

import json
from typing import Dict, Any


def build_hybrid_narrative_prompt(structured_data: Dict[str, Any], style: str = "professional") -> str:
    """
    Build prompt for narrative formatting of pre-computed facts.

    Args:
        structured_data: Complete pre-computed analysis from LearningEvaluator
        style: Narrative style (professional, technical, educational)

    Returns:
        Formatted prompt string
    """
    metadata = structured_data['metadata']
    speaker_stats = structured_data['speaker_statistics']
    evaluations = structured_data['evaluations']
    objectives = structured_data['objectives']
    top_comms = structured_data['top_communications']

    # Format speaker statistics table
    speaker_table = "| Speaker | Utterances | Percentage |\n| --- | --- | --- |\n"
    for stat in speaker_stats:
        speaker_table += f"| {stat['speaker']} | {stat['utterances']} | {stat['percentage']}% |\n"

    # Format objectives
    obj_list = []
    for obj_name, obj_data in objectives['details'].items():
        status = "✓ Complete" if obj_data['complete'] else f"⧗ In Progress ({obj_data['current_count']}/{obj_data['total_count']})"
        obj_list.append(f"- [{obj_data['rank']}] {obj_name}: {obj_data['description']} - {status}")
    objectives_text = "\n".join(obj_list) if obj_list else "No objectives data available"

    # Format top communications (for potential quotes)
    comms_text = "\n".join([
        f"[{c['timestamp'].split('T')[1][:8] if 'T' in c['timestamp'] else c['timestamp']}] "
        f"{c['speaker']}: \"{c['text']}\" (confidence: {c['confidence']:.2f})"
        for c in top_comms[:10]
    ])

    # Kirk patrick summary
    kirk = evaluations['kirkpatrick']
    kirk_summary = f"""
**Level 1 - Reaction (Engagement):**
- Total Communications: {kirk['level_1_reaction']['total_communications']}
- Unique Speakers: {kirk['level_1_reaction']['unique_speakers']}
- Participation Equity: {kirk['level_1_reaction']['participation_equity_score']}/100
- Avg Confidence: {kirk['level_1_reaction']['avg_transcription_confidence']}
- Assessment: {kirk['level_1_reaction']['interpretation']}

**Level 2 - Learning (Knowledge Acquisition):**
- Objectives Completed: {kirk['level_2_learning']['completed_objectives']}/{kirk['level_2_learning']['total_objectives']} ({kirk['level_2_learning']['objective_completion_rate']}%)
- Protocol Adherence: {kirk['level_2_learning']['protocol_adherence_score']}%
- Knowledge Level: {kirk['level_2_learning']['knowledge_level']}
- Assessment: {kirk['level_2_learning']['interpretation']}

**Level 3 - Behavior (Application):**
- Avg Response Time: {kirk['level_3_behavior']['avg_response_time_seconds']}s
- Decision Communications: {kirk['level_3_behavior']['decision_communications']}
- Coordination Score: {kirk['level_3_behavior']['coordination_score']}/100
- Assessment: {kirk['level_3_behavior']['interpretation']}

**Level 4 - Results (Mission Success):**
- Mission Completion: {kirk['level_4_results']['mission_completion_rate']}%
- Mission Grade: {kirk['level_4_results']['mission_grade']}% (if available)
- Critical Failures: {kirk['level_4_results']['critical_failures']}
- Assessment: {kirk['level_4_results']['interpretation']}
"""

    # Bloom's Taxonomy
    blooms = evaluations['blooms_taxonomy']
    blooms_summary = f"""
**Highest Cognitive Level:** {blooms['highest_level_demonstrated']}
**Cognitive Indicators by Level:**
{json.dumps(blooms['cognitive_levels'], indent=2)}
**Distribution:** {json.dumps(blooms['distribution_percentage'], indent=2)}
**Assessment:** {blooms['interpretation']}
"""

    # NASA Teamwork
    nasa = evaluations['nasa_teamwork']
    nasa_summary = f"""
**Overall Teamwork Score:** {nasa['overall_teamwork_score']}/100

**1. Communication:** {nasa['communication']['score']}/100 - {nasa['communication']['assessment']}
   - Clarity Average: {nasa['communication']['clarity_avg']}

**2. Coordination:** {nasa['coordination']['score']}/100 - {nasa['coordination']['assessment']}
   - Speaker Switches: {nasa['coordination']['speaker_switches']}

**3. Leadership:** {nasa['leadership']['score']}/100 - {nasa['leadership']['assessment']}
   - Primary Speaker: {nasa['leadership']['primary_speaker_percentage']}%

**4. Monitoring:** {nasa['monitoring']['score']}/100 - {nasa['monitoring']['assessment']}
   - Status Communications: {nasa['monitoring']['status_communications']}

**5. Adaptability:** {nasa['adaptability']['score']}/100 - {nasa['adaptability']['assessment']}
   - Adaptation Communications: {nasa['adaptability']['adaptation_communications']}

**Assessment:** {nasa['interpretation']}
"""

    # Mission-specific
    mission = evaluations['mission_specific']
    mission_summary = f"""
**Duration:** {mission['mission_duration']}
**Total Events:** {mission['total_events']}
**Communications per Minute:** {mission['communications_per_minute']}
**Event Distribution (Top 10):**
{json.dumps(mission['event_distribution'], indent=2)}
"""

    style_instructions = {
        "professional": "Write in a professional, analytical tone suitable for training assessment reports.",
        "technical": "Write in a technical, detailed style with emphasis on metrics and data.",
        "educational": "Write in an educational, instructive tone that explains concepts and provides learning insights."
    }

    prompt = f"""You are formatting a mission training report. ALL DATA has been pre-calculated.

🚨 CRITICAL RULES:
1. DO NOT calculate, modify, or invent ANY statistics - use ONLY the exact numbers provided
2. DO NOT create or modify quotes - use ONLY the verbatim quotes from "Available Quotes" section
3. Your ONLY job is to format the provided facts into a readable narrative
4. If you quote someone, use EXACT text from "Available Quotes" below
5. All numbers MUST match the data provided exactly

STYLE: {style_instructions.get(style, style_instructions['professional'])}

---

# PRE-COMPUTED MISSION DATA

## Mission Metadata
- Duration: {metadata['duration']}
- Total Events: {metadata['total_events']}
- Total Communications: {metadata['total_communications']}
- Unique Speakers: {metadata['unique_speakers']}
- Average Confidence: {metadata['avg_confidence']}

## Speaker Statistics (EXACT DATA)
{speaker_table}

## Mission Objectives (EXACT DATA)
{objectives_text}

## Kirkpatrick's Training Evaluation Model Results
{kirk_summary}

## Bloom's Taxonomy Cognitive Assessment
{blooms_summary}

## NASA Teamwork Framework Evaluation
{nasa_summary}

## Mission-Specific Metrics
{mission_summary}

## Available Quotes (USE THESE VERBATIM IF QUOTING)
{comms_text}

---

# YOUR TASK

Create a comprehensive mission training report in Markdown format with these sections:

# Mission Training Assessment Report

## Executive Summary
Write 2-3 paragraphs providing a high-level overview of the mission and training outcomes.
Use the statistics above. DO NOT create new statistics.

## Mission Overview
- Duration: [use exact value above]
- Participants: [use exact speaker count above]
- Communication Volume: [use exact numbers above]
- Mission Completion: [use exact objective completion rate above]

## Learning Assessment: Kirkpatrick's Model

### Level 1: Reaction & Engagement
Describe the engagement metrics using the EXACT numbers from the Kirkpatrick summary above.

### Level 2: Learning & Knowledge Acquisition
Describe the learning outcomes using the EXACT numbers from the Kirkpatrick summary above.

### Level 3: Behavior & Application
Describe the behavioral application using the EXACT numbers from the Kirkpatrick summary above.

### Level 4: Results & Mission Success
Describe the mission results using the EXACT numbers from the Kirkpatrick summary above.

## Cognitive Development: Bloom's Taxonomy
Describe the cognitive levels demonstrated using the EXACT data from Bloom's summary above.

## Team Performance: NASA Teamwork Framework
Describe each of the 5 NASA dimensions using the EXACT scores from the NASA summary above:
1. Communication
2. Coordination
3. Leadership
4. Monitoring & Situational Awareness
5. Adaptability

## Speaker Analysis
Create a section analyzing speaker participation using the EXACT statistics from the speaker table above.

## Notable Communications (OPTIONAL)
If relevant, include 2-3 VERBATIM quotes from "Available Quotes" section.
Format: [timestamp] speaker: "exact quote" (confidence: X.XX)

## Strengths & Recommendations
Based on the ACTUAL data above, identify:
- Key strengths (supported by the metrics above)
- Areas for improvement (supported by the metrics above)
- Specific training recommendations

## Conclusions
Summarize the training effectiveness using the ACTUAL metrics provided above.

---

REMEMBER:
- Every number must match the data provided
- Every quote must be verbatim from "Available Quotes"
- Your job is FORMATTING, not CALCULATION
- Do NOT invent, estimate, or modify ANY data
"""

    return prompt.strip()
