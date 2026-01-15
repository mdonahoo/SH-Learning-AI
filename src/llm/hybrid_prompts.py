"""
Hybrid prompt templates for LLM narrative formatting.

This module contains prompts that take PRE-COMPUTED facts and data,
asking the LLM ONLY to format them into readable narratives.
The LLM is NOT allowed to calculate, invent, or modify any facts.

Enhanced version includes integration with:
- Role inference engine
- Confidence distribution analysis
- Mission phase analysis
- Quality verification
- Speaker scorecards
- Communication quality analysis
"""

import json
from typing import Dict, Any, List, Optional


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


def build_enhanced_report_prompt(
    enhanced_data: Dict[str, Any],
    mission_name: str = "Mission Debrief",
    style: str = "professional"
) -> str:
    """
    Build prompt for enhanced mission report using all analysis components.

    This function takes the output from EnhancedReportBuilder and creates
    a prompt that matches the quality of professional example reports.

    Args:
        enhanced_data: Complete analysis data from EnhancedReportBuilder
        mission_name: Name of the mission
        style: Report style (professional, technical, educational)

    Returns:
        Formatted prompt string
    """
    # Extract all analysis sections
    statistics = enhanced_data.get('mission_statistics', {})
    role_analysis = enhanced_data.get('role_analysis', {})
    confidence_analysis = enhanced_data.get('confidence_analysis', {})
    phase_analysis = enhanced_data.get('phase_analysis', {})
    quality_verification = enhanced_data.get('quality_verification', {})
    communication_quality = enhanced_data.get('communication_quality', {})
    speaker_scorecards = enhanced_data.get('speaker_scorecards', {})
    learning_evaluation = enhanced_data.get('learning_evaluation', {})

    # Format mission statistics
    stats_text = f"""
**Mission Duration:** {statistics.get('mission_duration', 'Unknown')}
**Total Voice Communications:** {statistics.get('total_voice_communications', 0)} utterances
**Unique Speakers Detected:** {statistics.get('unique_speakers', 0)}
**Total Game Events:** {statistics.get('total_game_events', 0)}
**Mission Objectives:** {statistics.get('objectives_total', 0)} total, {statistics.get('objectives_completed', 0)} completed ({statistics.get('completion_rate', 0)}%)
**Mission Grade:** {statistics.get('mission_grade', 'N/A')}
"""

    # Format role analysis
    role_table = role_analysis.get('role_table', 'No role analysis available')
    role_methodology = role_analysis.get('methodology', '')

    # Format confidence distribution
    confidence_table = confidence_analysis.get('distribution_table', 'No confidence analysis available')
    confidence_stats = confidence_analysis.get('statistics', {})
    training_implications = confidence_analysis.get('training_implications', [])

    # Format phase analysis
    phase_section = phase_analysis.get('phase_analysis_section', 'No phase analysis available')

    # Format communication quality
    command_control = communication_quality.get('command_control_section', '')
    effective_examples = communication_quality.get('effective_examples', [])
    improvement_examples = communication_quality.get('improvement_examples', [])

    # Format speaker scorecards
    scorecards_section = speaker_scorecards.get('scorecards_section', '')

    # Format quality verification
    verification_section = quality_verification.get('verification_section', '')

    # Build effective communications table
    effective_table = "| Timestamp | Speaker | Communication | Assessment |\n| --- | --- | --- | --- |\n"
    for ex in effective_examples[:5]:
        text = ex.get('text', '')[:50] + "..." if len(ex.get('text', '')) > 50 else ex.get('text', '')
        effective_table += f"| {ex.get('timestamp', '')} | {ex.get('speaker', '')} | \"{text}\" | {ex.get('assessment', '')} |\n"

    # Build improvement communications table
    improvement_table = "| Timestamp | Speaker | Communication | Issue |\n| --- | --- | --- | --- |\n"
    for ex in improvement_examples[:5]:
        text = ex.get('text', '')[:50] + "..." if len(ex.get('text', '')) > 50 else ex.get('text', '')
        improvement_table += f"| {ex.get('timestamp', '')} | {ex.get('speaker', '')} | \"{text}\" | {ex.get('issue', '')} |\n"

    # Format training implications
    implications_text = "\n".join([f"- {impl}" for impl in training_implications])

    style_instructions = {
        "professional": "Write in a professional, analytical tone matching Starfleet Academy standards.",
        "technical": "Write in a technical, metrics-focused style with detailed statistical analysis.",
        "educational": "Write in an educational tone that explains concepts and provides learning insights."
    }

    prompt = f"""You are formatting a comprehensive mission debrief report. ALL DATA has been pre-calculated.

🚨 CRITICAL RULES - VIOLATION WILL INVALIDATE THE REPORT:
1. DO NOT calculate, modify, or invent ANY statistics - use ONLY exact numbers provided
2. DO NOT create or modify quotes - use ONLY verbatim text from provided examples
3. DO NOT assign character names to speakers - use speaker_1, speaker_2, etc.
4. DO NOT claim behaviors occurred unless evidence is provided
5. If data is missing, state "Data not available" - NEVER guess
6. Every claim MUST have supporting evidence from the pre-computed data

STYLE: {style_instructions.get(style, style_instructions['professional'])}

---

# PRE-COMPUTED MISSION DATA

## Mission Statistics (USE THESE EXACT VALUES)
{stats_text}

## Role Analysis (PRE-COMPUTED)
{role_table}

### Role Assignment Methodology (USE THIS EXPLANATION)
{role_methodology}

## Transcription Confidence Distribution (PRE-COMPUTED)
{confidence_table}

**Average Confidence:** {confidence_stats.get('average_confidence', 0):.1%}
**Quality Assessment:** {confidence_analysis.get('quality_assessment', 'Unknown')}

**Training Implications:**
{implications_text}

## Mission Phase Analysis (PRE-COMPUTED)
{phase_section}

## Command and Control Assessment (PRE-COMPUTED)

### Effective Command Examples (USE THESE EXACT QUOTES)
{effective_table}

### Communications Requiring Improvement (USE THESE EXACT QUOTES)
{improvement_table}

## Crew Performance Scorecards (PRE-COMPUTED)
{scorecards_section}

## Quality Verification (PRE-COMPUTED)
{verification_section}

---

# YOUR TASK

Format the above PRE-COMPUTED data into a professional mission debrief report.
Use this EXACT structure:

# Mission Debrief: {mission_name}

## Executive Summary
Write 2-3 paragraphs using the EXACT statistics above. Include:
- Mission duration and communication volume (use exact numbers)
- Objectives completed and mission grade (use exact numbers)
- Key accomplishments and challenges
- Training priorities based on the implications above

## Mission Statistics
Create a table with the EXACT values from "Mission Statistics" above.

## Role Analysis
Include the role table EXACTLY as provided above.
Include the methodology explanation EXACTLY as provided.

## Command and Control Assessment

### Command Clarity Analysis
Include the effective examples table EXACTLY as provided.
Include the improvement examples table EXACTLY as provided.

### Transcription Confidence Distribution
Include the confidence table EXACTLY as provided.
Include the training implications EXACTLY as provided.

## Mission Phase Analysis
Include the phase analysis EXACTLY as provided above.

## Crew Performance Scorecards
Include the scorecards EXACTLY as provided above.

## Mission Objectives Status
List completed and incomplete objectives based on the statistics provided.

## Training Recommendations

### Immediate Actions for This Crew
Based on the improvement examples and training implications, provide 3-5 specific recommendations.

### Protocol Improvements
Suggest specific protocols to address the issues identified in improvement examples.

### Team Exercises
Suggest 2-3 drills based on the gaps identified.

## Quality Verification
Include the verification section EXACTLY as provided above.

---

**Report Generated:** Based on mission data
**Data Sources:** transcripts.json, game_events.json
**Analysis Method:** Complete dataset analysis with keyword frequency role inference

---

REMEMBER:
- ALL numbers must match the pre-computed data EXACTLY
- ALL quotes must be VERBATIM from the provided examples
- Your job is FORMATTING and NARRATIVE FLOW only
- The analysis has already been done - do NOT recalculate anything
"""

    return prompt.strip()


def build_factual_summary_prompt(
    enhanced_data: Dict[str, Any],
    mission_name: str = "Mission Summary"
) -> str:
    """
    Build prompt for a purely factual summary with no narrative interpretation.

    Args:
        enhanced_data: Complete analysis data from EnhancedReportBuilder
        mission_name: Name of the mission

    Returns:
        Formatted prompt for factual-only summary
    """
    statistics = enhanced_data.get('mission_statistics', {})
    role_analysis = enhanced_data.get('role_analysis', {})
    confidence_analysis = enhanced_data.get('confidence_analysis', {})
    quality_verification = enhanced_data.get('quality_verification', {})

    prompt = f"""Generate a purely FACTUAL mission summary. NO interpretation or narrative.

# {mission_name} - Factual Summary

## Mission Data
- Duration: {statistics.get('mission_duration', 'Unknown')}
- Communications: {statistics.get('total_voice_communications', 0)}
- Speakers: {statistics.get('unique_speakers', 0)}
- Events: {statistics.get('total_game_events', 0)}
- Objectives: {statistics.get('objectives_completed', 0)}/{statistics.get('objectives_total', 0)} ({statistics.get('completion_rate', 0)}%)
- Grade: {statistics.get('mission_grade', 'N/A')}

## Speaker Distribution
{role_analysis.get('role_table', 'No data')}

## Confidence Distribution
{confidence_analysis.get('distribution_table', 'No data')}
Average: {confidence_analysis.get('statistics', {}).get('average_confidence', 0):.1%}

## Data Verification
{quality_verification.get('verification_table', 'No verification data')}

Output this data in clean markdown format with NO additional interpretation.
Just present the facts as given.
"""

    return prompt.strip()
