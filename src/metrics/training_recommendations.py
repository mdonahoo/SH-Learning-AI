"""
Comprehensive training recommendation engine for educational contexts.

Generates detailed, actionable training recommendations aligned with multiple
frameworks suitable for scout troops, school districts, and youth programs.

Frameworks integrated:
- 7 Habits of Highly Effective People (Stephen Covey)
- Kirkpatrick's Training Evaluation Model
- Bloom's Taxonomy
- TeamSTEPPS (AHRQ)
- NASA Teamwork Framework
- Scout Law and EDGE Method
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class RecommendationPriority(Enum):
    """Priority levels for recommendations."""
    CRITICAL = 1  # Immediate attention needed
    HIGH = 2      # Address in next session
    MEDIUM = 3    # Develop over time
    LOW = 4       # Nice to have


class SkillCategory(Enum):
    """Categories of skills for development."""
    COMMUNICATION = "Communication"
    LEADERSHIP = "Leadership"
    TEAMWORK = "Teamwork"
    TECHNICAL = "Technical Proficiency"
    DECISION_MAKING = "Decision Making"
    SITUATIONAL_AWARENESS = "Situational Awareness"
    PERSONAL_DEVELOPMENT = "Personal Development"


@dataclass
class TrainingRecommendation:
    """A single training recommendation."""
    title: str
    description: str
    priority: RecommendationPriority
    category: SkillCategory
    frameworks: List[str]  # Which frameworks this addresses
    target_audience: str   # "individual", "team", "leadership"
    time_estimate: str     # e.g., "5-10 minutes", "ongoing"
    activity_type: str     # "drill", "discussion", "exercise", "reflection"
    success_criteria: str  # How to know it's working
    scout_connection: Optional[str] = None  # Connection to Scout Law/Oath
    habit_connection: Optional[str] = None  # Connection to 7 Habits
    bloom_level: Optional[str] = None  # Target Bloom's level


@dataclass
class DrillActivity:
    """A specific drill or training activity."""
    name: str
    purpose: str
    duration: str
    participants: str  # "individual", "pairs", "full team"
    materials_needed: List[str]
    setup_instructions: str
    activity_steps: List[str]
    debrief_questions: List[str]
    variations: List[str]
    frameworks_addressed: List[str]


class TrainingRecommendationEngine:
    """
    Generates comprehensive training recommendations from analysis data.

    Designed for educational contexts including scout troops and school programs.
    Integrates multiple frameworks and provides age-appropriate activities.
    """

    # Scout Law values for connection
    SCOUT_LAW = [
        "Trustworthy", "Loyal", "Helpful", "Friendly",
        "Courteous", "Kind", "Obedient", "Cheerful",
        "Thrifty", "Brave", "Clean", "Reverent"
    ]

    # EDGE Method (Scouting teaching method)
    EDGE_METHOD = {
        "Explain": "Tell them what and why",
        "Demonstrate": "Show them how",
        "Guide": "Let them try with help",
        "Enable": "Let them do it independently"
    }

    def __init__(
        self,
        transcripts: List[Dict[str, Any]],
        analysis_results: Dict[str, Any] = None
    ):
        """
        Initialize the recommendation engine.

        Args:
            transcripts: List of transcript dictionaries
            analysis_results: Pre-computed analysis from other modules
        """
        self.transcripts = transcripts
        self.analysis = analysis_results or {}

    def generate_all_recommendations(self) -> Dict[str, Any]:
        """
        Generate comprehensive recommendations from all frameworks.

        Returns:
            Dictionary with categorized recommendations
        """
        recommendations = {
            'immediate_actions': self._generate_immediate_actions(),
            'communication_improvements': self._generate_communication_recommendations(),
            'leadership_development': self._generate_leadership_recommendations(),
            'teamwork_enhancements': self._generate_teamwork_recommendations(),
            'personal_growth': self._generate_personal_growth_recommendations(),
            'drills_and_exercises': self._generate_drills(),
            'discussion_topics': self._generate_discussion_topics(),
            'reflection_activities': self._generate_reflection_activities(),
            'framework_alignment': self._generate_framework_alignment(),
        }

        return recommendations

    def _generate_immediate_actions(self) -> List[TrainingRecommendation]:
        """Generate immediate action recommendations."""
        actions = []

        # Check communication quality
        comm_stats = self.analysis.get('communication_quality', {}).get('statistics', {})
        improvement_count = comm_stats.get('improvement_count', 0)
        total = comm_stats.get('total_utterances')

        # Only generate clarity recommendation if we have valid data
        if total is None or total == 0:
            logger.debug("No utterance data available for communication quality check")
        elif improvement_count / total > 0.15:
            actions.append(TrainingRecommendation(
                title="Communication Clarity Drill",
                description=(
                    "Practice the 'Complete Thought' technique: Before speaking, "
                    "form the complete sentence in your mind. No trailing off or "
                    "filler words (um, uh, like). Each communication should have "
                    "a clear subject, action, and target."
                ),
                priority=RecommendationPriority.CRITICAL,
                category=SkillCategory.COMMUNICATION,
                frameworks=["TeamSTEPPS: Communication", "7 Habits: Seek First to Understand"],
                target_audience="team",
                time_estimate="10 minutes before each mission",
                activity_type="drill",
                success_criteria="Reduce incomplete communications by 50%",
                scout_connection="A Scout is Trustworthy - clear communication builds trust",
                habit_connection="Habit 5: Seek First to Understand - clarity enables understanding",
                bloom_level="Apply"
            ))

        # Check confidence levels
        conf_stats = self.analysis.get('confidence_analysis', {}).get('statistics', {})
        avg_confidence = conf_stats.get('average_confidence')

        # Only generate voice projection recommendation if we have valid confidence data
        if avg_confidence is None:
            logger.debug("No confidence data available for voice projection check")
        elif avg_confidence < 0.7:
            actions.append(TrainingRecommendation(
                title="Voice Projection Training",
                description=(
                    "Practice speaking clearly and confidently. Use the 'Radio Voice' "
                    "technique: imagine speaking into a radio where clarity is critical. "
                    "Enunciate each word, speak at moderate pace, and project your voice."
                ),
                priority=RecommendationPriority.HIGH,
                category=SkillCategory.COMMUNICATION,
                frameworks=["Kirkpatrick Level 3: Behavior"],
                target_audience="individual",
                time_estimate="5 minutes warm-up",
                activity_type="exercise",
                success_criteria="Increase average transcription confidence above 75%",
                scout_connection="A Scout is Brave - confidence in communication",
                habit_connection="Habit 1: Be Proactive - take ownership of being understood",
                bloom_level="Apply"
            ))

        # Check acknowledgment patterns - only recommend if we have evidence of low acknowledgment rates
        pattern_counts = self.analysis.get('communication_quality', {}).get('pattern_counts', {})
        effective_patterns = pattern_counts.get('effective', {})

        # Get counts for acknowledgments and commands
        acknowledgment_count = effective_patterns.get('proper_acknowledgment', 0)
        command_count = effective_patterns.get('clear_command', 0)

        # Calculate acknowledgment ratio if we have command data
        # Only trigger recommendation if acknowledgment rate is below 50%
        if command_count > 0:
            ack_ratio = acknowledgment_count / command_count if command_count > 0 else 1.0

            if ack_ratio < 0.50:
                # Determine priority based on severity
                priority = (
                    RecommendationPriority.CRITICAL if ack_ratio < 0.25
                    else RecommendationPriority.HIGH
                )

                actions.append(TrainingRecommendation(
                    title="Closed-Loop Communication Protocol",
                    description=(
                        f"Only {acknowledgment_count} acknowledgments detected for {command_count} commands "
                        f"({ack_ratio:.0%} rate). Implement three-part acknowledgments: 1) Repeat the order, "
                        "2) Confirm understanding, 3) Report when complete. "
                        "Example: 'Set course for sector 7' → 'Course for sector 7, aye. "
                        "Course laid in. Engaging now.'"
                    ),
                    priority=priority,
                    category=SkillCategory.COMMUNICATION,
                    frameworks=["TeamSTEPPS: Communication", "NASA: Communication"],
                    target_audience="team",
                    time_estimate="Practice each mission",
                    activity_type="drill",
                    success_criteria="100% of commands receive verbal acknowledgment",
                    scout_connection="A Scout is Obedient - following and confirming orders",
                    habit_connection="Habit 5: Seek First to Understand, Then to Be Understood",
                    bloom_level="Apply"
                ))
        else:
            logger.debug("No clear commands detected - skipping closed-loop communication check")

        # Dynamic priority: Check confidence variance for stress indicator
        std_deviation = conf_stats.get('std_deviation')
        if std_deviation is not None and std_deviation > 0.2:
            actions.append(TrainingRecommendation(
                title="Stress Management Training",
                description=(
                    f"High confidence variance detected (std dev: {std_deviation:.2f}) indicating "
                    "inconsistent voice clarity. This often correlates with stress or pressure. "
                    "Practice steady breathing and deliberate pacing during high-intensity situations."
                ),
                priority=RecommendationPriority.HIGH,
                category=SkillCategory.PERSONAL_DEVELOPMENT,
                frameworks=["TeamSTEPPS: Situation Monitoring", "NASA: Stress Management"],
                target_audience="team",
                time_estimate="5 minutes before high-stress scenarios",
                activity_type="exercise",
                success_criteria="Reduce confidence variance below 0.15",
                scout_connection="A Scout is Brave - maintaining composure under pressure",
                habit_connection="Habit 1: Be Proactive - control responses to stress",
                bloom_level="Apply"
            ))

        # Dynamic priority: Check for speaker dominance (one speaker > 50% of talk time)
        speaker_stats = conf_stats.get('speaker_stats') or {}
        if speaker_stats:
            total_speaker_utterances = sum(
                s.get('count', 0) for s in speaker_stats.values()
            ) if isinstance(speaker_stats, dict) else 0

            if total_speaker_utterances > 0:
                for speaker, stats in speaker_stats.items():
                    if isinstance(stats, dict):
                        count = stats.get('count', 0)
                        ratio = count / total_speaker_utterances
                        if ratio > 0.5 and total_speaker_utterances >= 10:
                            actions.append(TrainingRecommendation(
                                title="Team Participation Balance",
                                description=(
                                    f"One speaker ({speaker}) dominated communication with {ratio:.0%} "
                                    f"of all utterances. Encourage balanced participation to leverage "
                                    "diverse perspectives and develop all team members."
                                ),
                                priority=RecommendationPriority.HIGH,
                                category=SkillCategory.TEAMWORK,
                                frameworks=["TeamSTEPPS: Team Structure", "7 Habits: Synergize"],
                                target_audience="team",
                                time_estimate="Ongoing awareness",
                                activity_type="discussion",
                                success_criteria="No single speaker exceeds 40% of communications",
                                scout_connection="A Scout is Friendly - creating space for all voices",
                                habit_connection="Habit 6: Synergize - value diverse contributions",
                                bloom_level="Analyze"
                            ))
                            break  # Only add once for the dominant speaker

        # Per-speaker recommendations for individuals with combined issues
        speaker_improvement = comm_stats.get('speaker_improvement', {})
        speaker_effective = comm_stats.get('speaker_effective', {})

        # Only generate per-speaker recommendations if we have data
        if speaker_improvement and speaker_stats:
            for speaker in speaker_improvement.keys():
                improvement_for_speaker = speaker_improvement.get(speaker, 0)
                effective_for_speaker = speaker_effective.get(speaker, 0)
                total_for_speaker = improvement_for_speaker + effective_for_speaker

                # Skip if insufficient data
                if total_for_speaker < 3:
                    continue

                # Calculate improvement ratio for this speaker
                improvement_ratio = improvement_for_speaker / total_for_speaker if total_for_speaker > 0 else 0

                # Get confidence for this speaker
                speaker_conf = speaker_stats.get(speaker, {})
                speaker_avg_conf = speaker_conf.get('avg', 1.0) if isinstance(speaker_conf, dict) else 1.0

                # Flag speakers with both high improvement needs (>30%) AND low confidence (<0.75)
                if improvement_ratio > 0.30 and speaker_avg_conf < 0.75:
                    actions.append(TrainingRecommendation(
                        title=f"Individual Coaching: {speaker}",
                        description=(
                            f"{speaker} shows combined challenges: {improvement_ratio:.0%} of communications "
                            f"need improvement and average transcription confidence is {speaker_avg_conf:.0%}. "
                            "Recommend one-on-one practice focusing on clarity, pacing, and projection."
                        ),
                        priority=RecommendationPriority.HIGH,
                        category=SkillCategory.COMMUNICATION,
                        frameworks=["Kirkpatrick Level 3: Behavior", "EDGE Method: Guide"],
                        target_audience="individual",
                        time_estimate="10 minutes individual practice",
                        activity_type="exercise",
                        success_criteria=f"{speaker} reduces improvement needs to <15% and confidence >80%",
                        scout_connection="A Scout is Helpful - supporting individual growth",
                        habit_connection="Habit 7: Sharpen the Saw - personal development",
                        bloom_level="Apply"
                    ))

        # Performance degradation detection: Compare early vs late confidence
        if self.transcripts and len(self.transcripts) >= 10:
            # Split transcripts into halves and compare confidence
            mid_point = len(self.transcripts) // 2
            early_transcripts = self.transcripts[:mid_point]
            late_transcripts = self.transcripts[mid_point:]

            early_confidence = [
                t.get('confidence', 0) for t in early_transcripts
                if isinstance(t.get('confidence'), (int, float))
            ]
            late_confidence = [
                t.get('confidence', 0) for t in late_transcripts
                if isinstance(t.get('confidence'), (int, float))
            ]

            if early_confidence and late_confidence:
                avg_early = sum(early_confidence) / len(early_confidence)
                avg_late = sum(late_confidence) / len(late_confidence)

                # Check for significant degradation (>15% drop)
                if avg_early > 0 and (avg_early - avg_late) / avg_early > 0.15:
                    degradation_pct = ((avg_early - avg_late) / avg_early) * 100
                    actions.append(TrainingRecommendation(
                        title="Stress Inoculation Training",
                        description=(
                            f"Performance degradation detected: confidence dropped {degradation_pct:.0f}% "
                            f"from early mission ({avg_early:.0%}) to late mission ({avg_late:.0%}). "
                            "This suggests fatigue or stress accumulation. Practice graduated stress "
                            "exposure with debriefs to build resilience."
                        ),
                        priority=RecommendationPriority.HIGH,
                        category=SkillCategory.PERSONAL_DEVELOPMENT,
                        frameworks=["NASA: Stress Management", "Kirkpatrick Level 3: Behavior"],
                        target_audience="team",
                        time_estimate="Progressive training across sessions",
                        activity_type="exercise",
                        success_criteria="Maintain consistent confidence throughout missions",
                        scout_connection="A Scout is Brave - building mental resilience",
                        habit_connection="Habit 7: Sharpen the Saw - balanced renewal",
                        bloom_level="Apply"
                    ))

        # 7 Habits integration: Check for low-scoring habits
        seven_habits = self.analysis.get('seven_habits', {})
        habits_raw = seven_habits.get('habits', {}) if isinstance(seven_habits, dict) else {}
        # Handle both dict format (from SevenHabitsAnalyzer) and list format (from audio_processor)
        if isinstance(habits_raw, list):
            habits_data = {
                h.get('habit_name', '').upper().replace(' ', '_'): h
                for h in habits_raw if isinstance(h, dict)
            }
        elif isinstance(habits_raw, dict):
            habits_data = habits_raw
        else:
            habits_data = {}

        # Mapping of habit names to user-friendly descriptions and focus areas
        habit_focus_areas = {
            'BE_PROACTIVE': {
                'name': 'Be Proactive',
                'focus': 'initiative and responsibility',
                'scout': 'A Scout is Trustworthy - taking ownership',
            },
            'BEGIN_WITH_END_IN_MIND': {
                'name': 'Begin with End in Mind',
                'focus': 'goal-setting and planning',
                'scout': 'A Scout is Obedient - understanding purpose',
            },
            'PUT_FIRST_THINGS_FIRST': {
                'name': 'Put First Things First',
                'focus': 'prioritization and time management',
                'scout': 'A Scout is Thrifty - wise use of resources',
            },
            'THINK_WIN_WIN': {
                'name': 'Think Win-Win',
                'focus': 'mutual benefit and cooperation',
                'scout': 'A Scout is Friendly - working for mutual success',
            },
            'SEEK_FIRST_TO_UNDERSTAND': {
                'name': 'Seek First to Understand',
                'focus': 'active listening and clarification',
                'scout': 'A Scout is Courteous - listening with respect',
            },
            'SYNERGIZE': {
                'name': 'Synergize',
                'focus': 'creative cooperation and building on ideas',
                'scout': 'A Scout is Loyal - commitment to team success',
            },
            'SHARPEN_THE_SAW': {
                'name': 'Sharpen the Saw',
                'focus': 'continuous improvement and learning',
                'scout': 'A Scout is Brave - pursuing growth',
            },
        }

        for habit_name, habit_info in habits_data.items():
            if not isinstance(habit_info, dict):
                continue

            score = habit_info.get('score', 5)
            youth_name = habit_info.get('youth_name', habit_name)
            development_tip = habit_info.get('development_tip', '')
            gap = habit_info.get('gap_to_next_score', '')

            # Only trigger for low-scoring habits (score <= 2)
            if score <= 2:
                focus_info = habit_focus_areas.get(habit_name, {})
                priority = RecommendationPriority.CRITICAL if score == 1 else RecommendationPriority.HIGH

                actions.append(TrainingRecommendation(
                    title=f"7 Habits Focus: {focus_info.get('name', youth_name)}",
                    description=(
                        f"Habit score of {score}/5 indicates development opportunity in "
                        f"{focus_info.get('focus', 'this area')}. {development_tip} "
                        f"{gap if gap else ''}"
                    ),
                    priority=priority,
                    category=SkillCategory.PERSONAL_DEVELOPMENT,
                    frameworks=[f"7 Habits: {focus_info.get('name', youth_name)}"],
                    target_audience="team",
                    time_estimate="Ongoing practice",
                    activity_type="exercise",
                    success_criteria=f"Increase {youth_name} score to 4+",
                    scout_connection=focus_info.get('scout', 'Scout Law alignment'),
                    habit_connection=f"Habit: {focus_info.get('name', youth_name)}",
                    bloom_level="Apply"
                ))

        return actions

    def _generate_communication_recommendations(self) -> List[TrainingRecommendation]:
        """Generate communication-focused recommendations."""
        return [
            TrainingRecommendation(
                title="SBAR Communication Framework",
                description=(
                    "Use SBAR for status reports: Situation (what's happening), "
                    "Background (context), Assessment (your analysis), "
                    "Recommendation (what you suggest). This ensures complete, "
                    "actionable communication."
                ),
                priority=RecommendationPriority.MEDIUM,
                category=SkillCategory.COMMUNICATION,
                frameworks=["TeamSTEPPS: Communication", "Bloom's: Analyze"],
                target_audience="team",
                time_estimate="Ongoing practice",
                activity_type="drill",
                success_criteria="Status reports include all four SBAR elements",
                scout_connection="A Scout is Helpful - complete information helps the team",
                habit_connection="Habit 2: Begin with the End in Mind - structured communication",
                bloom_level="Analyze"
            ),
            TrainingRecommendation(
                title="Challenge and Response Protocol",
                description=(
                    "Before executing critical actions, use challenge-response: "
                    "Challenger: 'Fire weapons, confirm?' Responder: 'Weapons fire confirmed.' "
                    "This prevents errors in high-stakes situations."
                ),
                priority=RecommendationPriority.MEDIUM,
                category=SkillCategory.COMMUNICATION,
                frameworks=["TeamSTEPPS: Mutual Support", "NASA: Communication"],
                target_audience="team",
                time_estimate="5 minutes training",
                activity_type="drill",
                success_criteria="All critical actions use challenge-response",
                scout_connection="A Scout is Trustworthy - verification builds trust",
                habit_connection="Habit 5: Seek First to Understand",
                bloom_level="Apply"
            ),
            TrainingRecommendation(
                title="Active Listening Practice",
                description=(
                    "Practice the 'Reflect and Clarify' technique: When receiving "
                    "information, reflect it back ('So you're saying...') and ask "
                    "clarifying questions before acting. This ensures understanding."
                ),
                priority=RecommendationPriority.MEDIUM,
                category=SkillCategory.COMMUNICATION,
                frameworks=["7 Habits: Seek First to Understand", "Bloom's: Understand"],
                target_audience="individual",
                time_estimate="Ongoing",
                activity_type="exercise",
                success_criteria="Clarifying questions asked before unclear orders executed",
                scout_connection="A Scout is Courteous - listening shows respect",
                habit_connection="Habit 5: Seek First to Understand, Then to Be Understood",
                bloom_level="Understand"
            ),
        ]

    def _generate_leadership_recommendations(self) -> List[TrainingRecommendation]:
        """Generate leadership-focused recommendations."""
        return [
            TrainingRecommendation(
                title="Situational Leadership Development",
                description=(
                    "Practice adapting leadership style to the situation: "
                    "Directive (tell) for emergencies, Coaching (sell) for learning, "
                    "Supporting (participate) for capable teams, Delegating (observe) "
                    "for experienced crews."
                ),
                priority=RecommendationPriority.MEDIUM,
                category=SkillCategory.LEADERSHIP,
                frameworks=["Kirkpatrick Level 3: Behavior", "NASA: Leadership"],
                target_audience="leadership",
                time_estimate="Ongoing development",
                activity_type="exercise",
                success_criteria="Leadership style matches situation requirements",
                scout_connection="A Scout is Friendly - adapts approach to others' needs",
                habit_connection="Habit 4: Think Win-Win - leadership that develops others",
                bloom_level="Evaluate"
            ),
            TrainingRecommendation(
                title="Decision-Making Under Pressure",
                description=(
                    "Use the OODA Loop: Observe (gather information), Orient (understand context), "
                    "Decide (choose action), Act (execute). Practice making faster, better decisions "
                    "by explicitly working through each step."
                ),
                priority=RecommendationPriority.HIGH,
                category=SkillCategory.DECISION_MAKING,
                frameworks=["Bloom's: Evaluate", "NASA: Adaptability"],
                target_audience="leadership",
                time_estimate="15 minutes training + practice",
                activity_type="drill",
                success_criteria="Decisions are faster and more consistent",
                scout_connection="A Scout is Brave - making decisions under pressure",
                habit_connection="Habit 3: Put First Things First - prioritized decision-making",
                bloom_level="Evaluate"
            ),
            TrainingRecommendation(
                title="Delegation and Empowerment",
                description=(
                    "Practice effective delegation: Clearly state the task, expected outcome, "
                    "authority level, and check-in points. Allow team members to accomplish "
                    "tasks their way while maintaining accountability."
                ),
                priority=RecommendationPriority.MEDIUM,
                category=SkillCategory.LEADERSHIP,
                frameworks=["7 Habits: Think Win-Win", "Kirkpatrick Level 4: Results"],
                target_audience="leadership",
                time_estimate="Ongoing",
                activity_type="exercise",
                success_criteria="Tasks delegated with clear expectations and appropriate autonomy",
                scout_connection="A Scout is Loyal - trusting team members with responsibility",
                habit_connection="Habit 4: Think Win-Win / Habit 6: Synergize",
                bloom_level="Create"
            ),
        ]

    def _generate_teamwork_recommendations(self) -> List[TrainingRecommendation]:
        """Generate teamwork-focused recommendations."""
        return [
            TrainingRecommendation(
                title="Cross-Training Rotation",
                description=(
                    "Rotate crew through different stations to build understanding "
                    "of each role's challenges. Each person should experience at least "
                    "three different stations over multiple missions."
                ),
                priority=RecommendationPriority.MEDIUM,
                category=SkillCategory.TEAMWORK,
                frameworks=["TeamSTEPPS: Team Structure", "7 Habits: Synergize"],
                target_audience="team",
                time_estimate="Multiple sessions",
                activity_type="exercise",
                success_criteria="Each team member can perform basics at 3+ stations",
                scout_connection="A Scout is Helpful - understanding others' roles",
                habit_connection="Habit 6: Synergize - understanding enables collaboration",
                bloom_level="Apply"
            ),
            TrainingRecommendation(
                title="Backup Behavior Protocol",
                description=(
                    "Establish explicit backup assignments: Each station has a designated "
                    "backup person. Practice scenarios where primary must hand off to backup "
                    "smoothly. Normalize asking for and offering help."
                ),
                priority=RecommendationPriority.HIGH,
                category=SkillCategory.TEAMWORK,
                frameworks=["TeamSTEPPS: Mutual Support", "NASA: Coordination"],
                target_audience="team",
                time_estimate="10 minutes setup + practice",
                activity_type="drill",
                success_criteria="Smooth handoffs during simulated overload scenarios",
                scout_connection="A Scout is Helpful - supporting teammates proactively",
                habit_connection="Habit 4: Think Win-Win - team success over individual glory",
                bloom_level="Apply"
            ),
            TrainingRecommendation(
                title="Team Huddle Protocol",
                description=(
                    "Implement brief team huddles at mission start, mid-mission, and end: "
                    "Start huddle: Goals and concerns. Mid-mission: Status and adjustments. "
                    "End huddle: What worked, what to improve."
                ),
                priority=RecommendationPriority.MEDIUM,
                category=SkillCategory.TEAMWORK,
                frameworks=["TeamSTEPPS: Situation Monitoring", "7 Habits: Sharpen the Saw"],
                target_audience="team",
                time_estimate="2-3 minutes each huddle",
                activity_type="exercise",
                success_criteria="Huddles happen consistently with participation from all",
                scout_connection="A Scout is Friendly - building team connections",
                habit_connection="Habit 7: Sharpen the Saw - continuous team improvement",
                bloom_level="Analyze"
            ),
        ]

    def _generate_personal_growth_recommendations(self) -> List[TrainingRecommendation]:
        """Generate personal growth recommendations."""
        return [
            TrainingRecommendation(
                title="Personal Mission Statement",
                description=(
                    "Have each crew member write a personal mission statement for their role: "
                    "What do they want to be known for? How do they want to contribute? "
                    "Share these with the team to build understanding and accountability."
                ),
                priority=RecommendationPriority.LOW,
                category=SkillCategory.PERSONAL_DEVELOPMENT,
                frameworks=["7 Habits: Begin with the End in Mind", "Kirkpatrick Level 2: Learning"],
                target_audience="individual",
                time_estimate="15 minutes reflection + sharing",
                activity_type="reflection",
                success_criteria="Each crew member has a written mission statement",
                scout_connection="Scout Oath - 'On my honor I will do my best'",
                habit_connection="Habit 2: Begin with the End in Mind",
                bloom_level="Create"
            ),
            TrainingRecommendation(
                title="Weekly Habit Focus",
                description=(
                    "Each week, focus on one of the 7 Habits. Post it at the bridge, "
                    "discuss examples before missions, and recognize demonstrations "
                    "of that habit during debrief."
                ),
                priority=RecommendationPriority.MEDIUM,
                category=SkillCategory.PERSONAL_DEVELOPMENT,
                frameworks=["7 Habits: All", "Kirkpatrick Level 3: Behavior"],
                target_audience="team",
                time_estimate="5 minutes before/after each mission",
                activity_type="discussion",
                success_criteria="Team can identify and demonstrate the weekly habit",
                scout_connection="Scout Law points align with 7 Habits",
                habit_connection="All 7 Habits - systematic development",
                bloom_level="Apply"
            ),
            TrainingRecommendation(
                title="Learning Journal",
                description=(
                    "Each crew member keeps a brief learning journal. After each mission, "
                    "write: 1) What went well, 2) What I learned, 3) What I'll do differently. "
                    "Review journals periodically to track growth."
                ),
                priority=RecommendationPriority.LOW,
                category=SkillCategory.PERSONAL_DEVELOPMENT,
                frameworks=["7 Habits: Sharpen the Saw", "Bloom's: Evaluate"],
                target_audience="individual",
                time_estimate="5 minutes after each mission",
                activity_type="reflection",
                success_criteria="Journals show growth trends over time",
                scout_connection="A Scout is Reverent - reflection and self-improvement",
                habit_connection="Habit 7: Sharpen the Saw - continuous improvement",
                bloom_level="Evaluate"
            ),
        ]

    def _generate_drills(self) -> List[DrillActivity]:
        """Generate specific drill activities."""
        return [
            DrillActivity(
                name="Rapid Fire Communications",
                purpose="Improve communication speed and clarity under pressure",
                duration="10 minutes",
                participants="full team",
                materials_needed=["Scenario cards", "Timer"],
                setup_instructions="Prepare cards with scenarios requiring quick responses",
                activity_steps=[
                    "Leader reads scenario card",
                    "Target station must respond within 5 seconds",
                    "Response must include acknowledgment and action",
                    "Team evaluates clarity (thumbs up/down)",
                    "Repeat with increasing complexity"
                ],
                debrief_questions=[
                    "Which responses were clearest? Why?",
                    "What made some responses difficult?",
                    "How can we improve response time without sacrificing clarity?"
                ],
                variations=[
                    "Add background noise/distractions",
                    "Require responses from multiple stations",
                    "Include priority conflicts"
                ],
                frameworks_addressed=["TeamSTEPPS: Communication", "Kirkpatrick Level 3"]
            ),
            DrillActivity(
                name="Station Swap Challenge",
                purpose="Build cross-functional understanding and flexibility",
                duration="20 minutes",
                participants="full team",
                materials_needed=["Timer", "Station checklists"],
                setup_instructions="Pair up stations for swap (Helm-Tactical, Science-Engineering, etc.)",
                activity_steps=[
                    "Original operator teaches partner the basics (5 min)",
                    "Partners swap stations",
                    "Run simple scenario with swapped roles (10 min)",
                    "Original operators can coach but not take over",
                    "Debrief as team"
                ],
                debrief_questions=[
                    "What was harder than expected about the other role?",
                    "What did you learn that will help you support that station?",
                    "How does understanding other roles improve teamwork?"
                ],
                variations=[
                    "Extend scenario difficulty",
                    "Three-way rotation",
                    "Complete mission with swapped roles"
                ],
                frameworks_addressed=["TeamSTEPPS: Team Structure", "7 Habits: Synergize"]
            ),
            DrillActivity(
                name="The Silent Bridge",
                purpose="Practice non-verbal communication and situational awareness",
                duration="15 minutes",
                participants="full team",
                materials_needed=["Hand signal guide", "Whiteboard for written communication"],
                setup_instructions="Establish basic hand signals for common commands",
                activity_steps=[
                    "Review hand signals as a team",
                    "Run scenario with no verbal communication",
                    "All communication through hand signals, pointing, or written notes",
                    "Debrief challenges and successes"
                ],
                debrief_questions=[
                    "What information was hardest to convey silently?",
                    "How did you adapt your communication?",
                    "What does this teach about clear verbal communication?"
                ],
                variations=[
                    "Limited to hand signals only",
                    "Captain can speak but crew cannot",
                    "One person translates between silent crew and speaking captain"
                ],
                frameworks_addressed=["TeamSTEPPS: Communication", "Bloom's: Apply"]
            ),
            DrillActivity(
                name="Proactive Initiative Round",
                purpose="Build Habit 1: Be Proactive - taking initiative",
                duration="15 minutes",
                participants="full team",
                materials_needed=["Initiative tracking sheet"],
                setup_instructions="Each person has a sheet to track proactive actions",
                activity_steps=[
                    "Run normal scenario",
                    "Each person tracks when they: volunteered information, offered help, anticipated needs",
                    "Cannot wait to be asked - must proactively contribute",
                    "Count initiatives at end",
                    "Discuss quality and impact of initiatives"
                ],
                debrief_questions=[
                    "What proactive actions had the biggest impact?",
                    "When did you hesitate? Why?",
                    "How does being proactive change the team dynamic?"
                ],
                variations=[
                    "Competition for most valuable initiative",
                    "Specific focus areas (safety, efficiency, communication)",
                    "Leadership role rotation during drill"
                ],
                frameworks_addressed=["7 Habits: Be Proactive", "NASA: Adaptability"]
            ),
        ]

    def _generate_discussion_topics(self) -> List[Dict[str, Any]]:
        """Generate discussion topics for team development."""
        return [
            {
                'topic': "The Win-Win Mission",
                'question': "How can we ensure that everyone on the crew 'wins' during a mission - not just completing objectives, but growing and contributing?",
                'frameworks': ["7 Habits: Think Win-Win"],
                'discussion_points': [
                    "What does 'winning' look like for each role?",
                    "How can we celebrate individual contributions while focusing on team success?",
                    "What happens when roles are in conflict? How do we find win-win?"
                ],
                'scout_connection': "A Scout is Helpful and Friendly - helping others succeed",
            },
            {
                'topic': "First Things First Under Pressure",
                'question': "When multiple emergencies happen at once, how do we decide what to do first?",
                'frameworks': ["7 Habits: Put First Things First", "Bloom's: Evaluate"],
                'discussion_points': [
                    "What makes something 'urgent' vs 'important'?",
                    "How do we communicate priorities to the team?",
                    "Practice scenario: Three alerts at once - how do you prioritize?"
                ],
                'scout_connection': "A Scout is Brave - making tough decisions under pressure",
            },
            {
                'topic': "Learning from Failure",
                'question': "What's the most valuable thing we learned from a mission that didn't go as planned?",
                'frameworks': ["7 Habits: Sharpen the Saw", "Kirkpatrick Level 2: Learning"],
                'discussion_points': [
                    "What's the difference between a mistake and a failure?",
                    "How can we create an environment where it's safe to fail and learn?",
                    "What would you do differently knowing what you know now?"
                ],
                'scout_connection': "A Scout is Trustworthy and Cheerful - honest about mistakes, positive about learning",
            },
        ]

    def _generate_reflection_activities(self) -> List[Dict[str, Any]]:
        """Generate reflection activities for personal development."""
        return [
            {
                'activity': "Circle of Influence Exercise",
                'description': "Identify what you can control (inner circle) vs what you can only influence (outer circle) vs what is outside your control. Focus energy on inner circle.",
                'framework': "7 Habits: Be Proactive",
                'time': "10 minutes",
                'questions': [
                    "What aspects of your bridge role are in your circle of control?",
                    "What can you influence but not directly control?",
                    "Where have you been wasting energy on things outside your control?"
                ],
            },
            {
                'activity': "Personal Mission Review",
                'description': "Review your personal mission statement and assess how well your recent performance aligns with it.",
                'framework': "7 Habits: Begin with the End in Mind",
                'time': "5 minutes",
                'questions': [
                    "Did my actions today align with my mission?",
                    "What would my best self have done differently?",
                    "What one thing will I focus on next time?"
                ],
            },
            {
                'activity': "Synergy Moments",
                'description': "Identify moments where the team achieved more together than they could have individually.",
                'framework': "7 Habits: Synergize",
                'time': "5 minutes team discussion",
                'questions': [
                    "When did we create something better together?",
                    "Whose idea built on someone else's?",
                    "How can we create more moments like this?"
                ],
            },
        ]

    def _generate_framework_alignment(self) -> Dict[str, List[str]]:
        """Generate framework alignment mapping."""
        return {
            "7 Habits → Scout Law": [
                "Be Proactive → Trustworthy (take responsibility)",
                "Begin with End in Mind → Obedient (understand purpose)",
                "Put First Things First → Thrifty (prioritize resources)",
                "Think Win-Win → Friendly, Helpful (mutual success)",
                "Seek First to Understand → Courteous (listen with respect)",
                "Synergize → Loyal (commit to team success)",
                "Sharpen the Saw → Brave (pursue continuous growth)",
            ],
            "7 Habits → TeamSTEPPS": [
                "Be Proactive → Situation Monitoring (proactive awareness)",
                "Think Win-Win → Mutual Support (team success focus)",
                "Seek First to Understand → Communication (clear exchange)",
                "Synergize → Team Structure (effective collaboration)",
            ],
            "7 Habits → Bloom's Taxonomy": [
                "Habit 1-3 (Personal): Remember, Understand, Apply",
                "Habit 4-6 (Interpersonal): Analyze, Evaluate",
                "Habit 7 (Renewal): Create (continuous improvement)",
            ],
            "Scout EDGE Method → Kirkpatrick": [
                "Explain → Level 1: Reaction (engagement)",
                "Demonstrate → Level 2: Learning (knowledge)",
                "Guide → Level 3: Behavior (application)",
                "Enable → Level 4: Results (independence)",
            ],
        }

    def generate_recommendations_section(self) -> str:
        """
        Generate complete training recommendations section for reports.

        Returns:
            Markdown formatted recommendations section
        """
        recommendations = self.generate_all_recommendations()

        lines = [
            "## Training Recommendations",
            "",
            "*Aligned with educational frameworks including the 7 Habits of Highly Effective People, "
            "TeamSTEPPS, Bloom's Taxonomy, and Scout Law*",
            "",
        ]

        # Immediate Actions
        lines.append("### Immediate Actions (This Session or Next)")
        lines.append("")
        for i, rec in enumerate(recommendations['immediate_actions'], 1):
            lines.append(f"**{i}. {rec.title}** ({rec.category.value})")
            lines.append(f"- {rec.description}")
            lines.append(f"- *Framework Alignment:* {', '.join(rec.frameworks)}")
            if rec.scout_connection:
                lines.append(f"- *Scout Connection:* {rec.scout_connection}")
            if rec.habit_connection:
                lines.append(f"- *7 Habits Connection:* {rec.habit_connection}")
            lines.append(f"- *Success Criteria:* {rec.success_criteria}")
            lines.append("")

        # Communication Improvements
        lines.append("### Communication Skill Development")
        lines.append("")
        for rec in recommendations['communication_improvements']:
            lines.append(f"**{rec.title}**")
            lines.append(f"- {rec.description}")
            lines.append(f"- *Time:* {rec.time_estimate} | *Audience:* {rec.target_audience}")
            if rec.habit_connection:
                lines.append(f"- *7 Habits:* {rec.habit_connection}")
            lines.append("")

        # Leadership Development
        lines.append("### Leadership Development")
        lines.append("")
        for rec in recommendations['leadership_development']:
            lines.append(f"**{rec.title}**")
            lines.append(f"- {rec.description}")
            lines.append(f"- *Frameworks:* {', '.join(rec.frameworks)}")
            if rec.habit_connection:
                lines.append(f"- *7 Habits:* {rec.habit_connection}")
            lines.append("")

        # Teamwork Enhancements
        lines.append("### Teamwork Enhancement")
        lines.append("")
        for rec in recommendations['teamwork_enhancements']:
            lines.append(f"**{rec.title}**")
            lines.append(f"- {rec.description}")
            if rec.scout_connection:
                lines.append(f"- *Scout Connection:* {rec.scout_connection}")
            lines.append("")

        # Personal Growth
        lines.append("### Personal Growth Opportunities")
        lines.append("")
        for rec in recommendations['personal_growth']:
            lines.append(f"**{rec.title}**")
            lines.append(f"- {rec.description}")
            if rec.habit_connection:
                lines.append(f"- *7 Habits:* {rec.habit_connection}")
            lines.append("")

        # Drills
        lines.append("### Recommended Training Drills")
        lines.append("")
        for drill in recommendations['drills_and_exercises'][:3]:
            lines.append(f"**{drill.name}** ({drill.duration})")
            lines.append(f"- *Purpose:* {drill.purpose}")
            lines.append(f"- *Participants:* {drill.participants}")
            lines.append(f"- *Frameworks:* {', '.join(drill.frameworks_addressed)}")
            lines.append("")

        # Discussion Topics
        lines.append("### Team Discussion Topics")
        lines.append("")
        for topic in recommendations['discussion_topics']:
            lines.append(f"**{topic['topic']}**")
            lines.append(f"- *Question:* {topic['question']}")
            if topic.get('scout_connection'):
                lines.append(f"- *Scout Connection:* {topic['scout_connection']}")
            lines.append("")

        # Framework Alignment
        lines.append("### Framework Alignment Guide")
        lines.append("")
        for mapping, alignments in recommendations['framework_alignment'].items():
            lines.append(f"**{mapping}:**")
            for alignment in alignments[:3]:
                lines.append(f"- {alignment}")
            lines.append("")

        return "\n".join(lines)

    def get_structured_results(self) -> Dict[str, Any]:
        """
        Get structured results suitable for LLM prompts.

        Returns:
            Dictionary with all recommendation data
        """
        recommendations = self.generate_all_recommendations()

        return {
            'recommendations_section': self.generate_recommendations_section(),
            'immediate_actions': [
                {
                    'title': r.title,
                    'description': r.description,
                    'priority': r.priority.name,
                    'category': r.category.value,
                    'frameworks': r.frameworks,
                    'scout_connection': r.scout_connection,
                    'habit_connection': r.habit_connection,
                    'success_criteria': r.success_criteria,
                }
                for r in recommendations['immediate_actions']
            ],
            'drills': [
                {
                    'name': d.name,
                    'purpose': d.purpose,
                    'duration': d.duration,
                    'steps': d.activity_steps,
                    'debrief_questions': d.debrief_questions,
                }
                for d in recommendations['drills_and_exercises']
            ],
            'discussion_topics': recommendations['discussion_topics'],
            'framework_alignment': recommendations['framework_alignment'],
            'total_recommendations': sum(
                len(v) for v in recommendations.values()
                if isinstance(v, list)
            ),
        }
