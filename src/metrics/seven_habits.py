"""
Seven Habits of Highly Effective People/Leaders framework.

Based on Stephen Covey's framework, adapted for team performance analysis
in bridge simulator training contexts. Suitable for youth programs,
scout troops, and educational settings.

Reference: Covey, S. R. (1989). The 7 Habits of Highly Effective People.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class SevenHabit(Enum):
    """The Seven Habits of Highly Effective People."""
    BE_PROACTIVE = 1  # Take initiative, responsibility
    BEGIN_WITH_END_IN_MIND = 2  # Vision, planning, goals
    PUT_FIRST_THINGS_FIRST = 3  # Prioritization, time management
    THINK_WIN_WIN = 4  # Mutual benefit, cooperation
    SEEK_FIRST_TO_UNDERSTAND = 5  # Empathic listening
    SYNERGIZE = 6  # Creative cooperation, teamwork
    SHARPEN_THE_SAW = 7  # Continuous improvement, learning


@dataclass
class HabitIndicators:
    """Observable behavior indicators for each habit."""

    # Habit 1: Be Proactive - Take initiative without being asked
    # Note: "ready/standing by/prepared" removed — these indicate REACTIVE waiting,
    # not proactive initiative. Kept patterns that show self-directed action.
    BE_PROACTIVE_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)\b(i'll|i will|let me|i can|i've got)\b",
        r"(?i)\b(taking|handling|on it|got this)\b",
        r"(?i)\b(initiative|volunteer|step up)\b",
        r"(?i)\b(i'm going to|i'm gonna|i'll go ahead)\b",
        r"(?i)\b(checking|monitoring|watching)\b.{5,}",
    ])

    # Habit 2: Begin with the End in Mind - Goal-oriented communication
    # Note: Removed bare "first/then/next" — too common and non-specific.
    # Kept goal/plan/strategy vocabulary that signals forward-looking intent.
    BEGIN_WITH_END_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)\b(objective|goal|mission|target|purpose)\b",
        r"(?i)\b(plan|strategy|approach)\b",
        r"(?i)\b(outcome|result|success)\b.{3,}",
        r"(?i)\b(our (priority|objective|goal|mission))\b",
        r"(?i)\bwe need to .{5,}",
    ])

    # Habit 3: Put First Things First - Prioritization
    # Note: Removed bare "first/before/primary" — too common in bridge chatter.
    # Kept patterns showing explicit prioritization decisions.
    FIRST_THINGS_FIRST_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)\b(priority|urgent|critical)\b",
        r"(?i)\b(most important|top priority|main concern)\b",
        r"(?i)\b(focus on|concentrate on|attend to)\b",
        r"(?i)\b(right away|immediately|right now)\b.{3,}",
        r"(?i)\b(before we|first we need|first priority)\b",
    ])

    # Habit 4: Think Win-Win - Mutual benefit, cooperation
    # Note: Removed bare "we/us/our" — nearly every collaborative utterance
    # contains these, inflating scores to near-max. Kept words that signal
    # deliberate cooperation and mutual benefit.
    THINK_WIN_WIN_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)\b(together|as a team)\b",
        r"(?i)\b(help|assist|support|backup)\b",
        r"(?i)\b(share|collaborate|cooperate)\b",
        r"(?i)\b(everyone|all of us)\b",
        r"(?i)\b(fair|equal|balance)\b",
    ])

    # Habit 5: Seek First to Understand - Active listening, questions
    # Note: Tightened question pattern — bare "what?" interjections don't
    # indicate empathic listening. Require at least a few words after the
    # question word to signal a genuine clarifying question.
    SEEK_TO_UNDERSTAND_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)\b(what|why|how|where|when|who)\b.{4,}\?",
        r"(?i)\b(understand|clarify|explain|tell me)\b",
        r"(?i)\b(confirm|verify|repeat|say again)\b",
        r"(?i)\b(do you mean|can you explain|what do you think)\b",
        r"(?i)\b(listen|hear me out|let me understand)\b",
    ])

    # Habit 6: Synergize - Creative cooperation, building on ideas
    # Note: Removed "together" (overlaps Habit 4). Tightened to patterns
    # indicating creative brainstorming and building on others' ideas.
    SYNERGIZE_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)\b(idea|suggest|option|alternative)\b",
        r"(?i)\b(what if we|how about we|could we try|let's try)\b",
        r"(?i)\b(build on|add to|combine)\b",
        r"(?i)\b(solution|solve|figure out)\b",
        r"(?i)\b(coordinate|work with|teamwork)\b",
    ])

    # Habit 7: Sharpen the Saw - Learning, improvement, reflection
    SHARPEN_SAW_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)\b(learn|learned|learning)\b",
        r"(?i)\b(improve|better|enhance)\b",
        r"(?i)\b(practice|train|drill)\b",
        r"(?i)\b(review|reflect|debrief)\b",
        r"(?i)\b(next time|in future|going forward)\b",
    ])


@dataclass
class HabitAssessment:
    """Assessment results for a single habit."""
    habit: SevenHabit
    score: int  # 1-5
    count: int
    frequency: float
    examples: List[Dict[str, Any]]
    interpretation: str
    youth_friendly_name: str
    development_tip: str
    pattern_breakdown: Dict[str, int] = field(default_factory=dict)  # Count per pattern
    speaker_contributions: Dict[str, int] = field(default_factory=dict)  # Count per speaker
    gap_to_next_score: str = ""  # What's needed to reach next score level


class SevenHabitsAnalyzer:
    """
    Analyzes team communications using the 7 Habits framework.

    Designed for educational contexts including scout troops and school programs.
    Provides youth-friendly language and actionable development tips.
    """

    # Youth-friendly names for each habit
    YOUTH_NAMES = {
        SevenHabit.BE_PROACTIVE: "Take Initiative",
        SevenHabit.BEGIN_WITH_END_IN_MIND: "Know Your Goals",
        SevenHabit.PUT_FIRST_THINGS_FIRST: "Prioritize What Matters",
        SevenHabit.THINK_WIN_WIN: "Work Together for Success",
        SevenHabit.SEEK_FIRST_TO_UNDERSTAND: "Listen Before Speaking",
        SevenHabit.SYNERGIZE: "Create Better Solutions Together",
        SevenHabit.SHARPEN_THE_SAW: "Keep Learning and Growing",
    }

    # Development tips for each habit
    DEVELOPMENT_TIPS = {
        SevenHabit.BE_PROACTIVE: (
            "Practice saying 'I will' instead of 'Someone should.' "
            "Look for opportunities to help before being asked."
        ),
        SevenHabit.BEGIN_WITH_END_IN_MIND: (
            "Before starting any task, ask yourself: 'What does success look like?' "
            "State the goal clearly before taking action."
        ),
        SevenHabit.PUT_FIRST_THINGS_FIRST: (
            "When multiple things need attention, identify what's most important. "
            "Do the hard or urgent things first, not just the easy ones."
        ),
        SevenHabit.THINK_WIN_WIN: (
            "Ask 'How can we both succeed?' instead of 'How can I win?' "
            "Look for ways to help teammates while accomplishing your tasks."
        ),
        SevenHabit.SEEK_FIRST_TO_UNDERSTAND: (
            "Practice repeating back what someone said before responding. "
            "Ask clarifying questions: 'Do you mean...?' or 'Can you explain...?'"
        ),
        SevenHabit.SYNERGIZE: (
            "When facing a challenge, ask teammates for their ideas. "
            "Build on others' suggestions with 'Yes, and...' instead of 'No, but...'"
        ),
        SevenHabit.SHARPEN_THE_SAW: (
            "After each mission, ask 'What did I learn?' and 'What would I do differently?' "
            "Practice weak areas during training, not just what you're already good at."
        ),
    }

    def __init__(
        self,
        transcripts: List[Dict[str, Any]],
        indicators: Optional[HabitIndicators] = None
    ):
        """
        Initialize the 7 Habits analyzer.

        Args:
            transcripts: List of transcript dictionaries
            indicators: Optional custom indicators
        """
        self.transcripts = transcripts
        self.indicators = indicators or HabitIndicators()
        self._habit_pattern_map = self._build_pattern_map()

    def _build_pattern_map(self) -> Dict[SevenHabit, List[str]]:
        """Build mapping of habits to detection patterns."""
        return {
            SevenHabit.BE_PROACTIVE: self.indicators.BE_PROACTIVE_PATTERNS,
            SevenHabit.BEGIN_WITH_END_IN_MIND: self.indicators.BEGIN_WITH_END_PATTERNS,
            SevenHabit.PUT_FIRST_THINGS_FIRST: self.indicators.FIRST_THINGS_FIRST_PATTERNS,
            SevenHabit.THINK_WIN_WIN: self.indicators.THINK_WIN_WIN_PATTERNS,
            SevenHabit.SEEK_FIRST_TO_UNDERSTAND: self.indicators.SEEK_TO_UNDERSTAND_PATTERNS,
            SevenHabit.SYNERGIZE: self.indicators.SYNERGIZE_PATTERNS,
            SevenHabit.SHARPEN_THE_SAW: self.indicators.SHARPEN_SAW_PATTERNS,
        }

    def analyze_all_habits(self) -> Dict[SevenHabit, HabitAssessment]:
        """
        Analyze transcripts for all 7 Habits.

        Returns:
            Dictionary mapping habits to their assessments
        """
        results = {}
        total_utterances = len(self.transcripts) if self.transcripts else 1

        # Track which utterance indices matched ANY habit for dedup
        self._matched_utterance_indices: set = set()

        for habit in SevenHabit:
            patterns = self._habit_pattern_map[habit]
            count = 0
            examples = []
            pattern_breakdown = {f"pattern_{i}": 0 for i in range(len(patterns))}
            speaker_contributions = defaultdict(int)

            for idx, t in enumerate(self.transcripts):
                text = t.get('text', '')
                speaker = t.get('speaker') or t.get('speaker_id') or 'unknown'
                timestamp = t.get('timestamp', '')

                for i, pattern in enumerate(patterns):
                    if re.search(pattern, text):
                        count += 1
                        pattern_breakdown[f"pattern_{i}"] += 1
                        speaker_contributions[speaker] += 1
                        self._matched_utterance_indices.add(idx)
                        if len(examples) < 5:
                            examples.append({
                                'timestamp': timestamp,
                                'speaker': speaker,
                                'text': text,
                            })
                        break  # Count each utterance once per habit

            frequency = count / total_utterances
            score = self._calculate_score(frequency)
            interpretation = self._interpret_score(habit, score, count)

            # Calculate gap to next score
            gap_to_next_score = self._calculate_gap_to_next(score, frequency, total_utterances)

            results[habit] = HabitAssessment(
                habit=habit,
                score=score,
                count=count,
                frequency=round(frequency * 100, 1),
                examples=examples,
                interpretation=interpretation,
                youth_friendly_name=self.YOUTH_NAMES[habit],
                development_tip=self.DEVELOPMENT_TIPS[habit],
                pattern_breakdown={k: v for k, v in pattern_breakdown.items() if v > 0},
                speaker_contributions=dict(speaker_contributions),
                gap_to_next_score=gap_to_next_score,
            )

        return results

    def _calculate_gap_to_next(self, current_score: int, frequency: float, total: int) -> str:
        """Calculate what's needed to reach the next score level."""
        thresholds = {
            1: 0.05,  # To reach score 2
            2: 0.10,  # To reach score 3
            3: 0.20,  # To reach score 4
            4: 0.30,  # To reach score 5
        }

        if current_score >= 5:
            return "Maximum score achieved"

        target_freq = thresholds.get(current_score, 0.30)
        current_count = int(frequency * total)
        needed_count = int(target_freq * total) + 1
        additional_needed = max(0, needed_count - current_count)

        return (
            f"Need {additional_needed} more observations to reach score {current_score + 1} "
            f"(target: {target_freq*100:.0f}% frequency)"
        )

    def _calculate_score(self, frequency: float) -> int:
        """Convert frequency to 1-5 score."""
        if frequency >= 0.30:
            return 5
        elif frequency >= 0.20:
            return 4
        elif frequency >= 0.10:
            return 3
        elif frequency >= 0.05:
            return 2
        else:
            return 1

    def _interpret_score(self, habit: SevenHabit, score: int, count: int) -> str:
        """Generate interpretation of habit score."""
        name = self.YOUTH_NAMES[habit]

        if score >= 4:
            return f"Excellent demonstration of '{name}' ({count} examples observed)"
        elif score >= 3:
            return f"Good practice of '{name}' ({count} examples) - continue developing"
        elif score >= 2:
            return f"Some evidence of '{name}' ({count} examples) - focus area for growth"
        else:
            return f"'{name}' needs attention ({count} examples) - priority development area"

    def get_strongest_habits(self, top_n: int = 3) -> List[HabitAssessment]:
        """Get the top N strongest habits."""
        results = self.analyze_all_habits()
        sorted_habits = sorted(results.values(), key=lambda x: (x.score, x.count), reverse=True)
        return sorted_habits[:top_n]

    def get_growth_areas(self, bottom_n: int = 3) -> List[HabitAssessment]:
        """Get the bottom N habits needing development."""
        results = self.analyze_all_habits()
        sorted_habits = sorted(results.values(), key=lambda x: (x.score, x.count))
        return sorted_habits[:bottom_n]

    def generate_habits_section(self) -> str:
        """
        Generate markdown section for 7 Habits analysis.

        Returns:
            Markdown formatted analysis
        """
        results = self.analyze_all_habits()

        lines = [
            "## Leadership Development: 7 Habits Assessment",
            "",
            "*Based on Stephen Covey's \"The 7 Habits of Highly Effective People\"*",
            "",
            "| Habit | Youth-Friendly Name | Score | Observations | Assessment |",
            "| --- | --- | --- | --- | --- |",
        ]

        for habit in SevenHabit:
            assessment = results[habit]
            lines.append(
                f"| Habit {habit.value} | {assessment.youth_friendly_name} | "
                f"{assessment.score}/5 | {assessment.count} | {assessment.interpretation.split(' - ')[0]} |"
            )

        lines.append("")

        # Strengths section
        strengths = self.get_strongest_habits(3)
        lines.append("### Team Strengths")
        lines.append("")
        for s in strengths:
            if s.score >= 3:
                lines.append(f"**{s.youth_friendly_name}** (Score: {s.score}/5)")
                lines.append(f"- {s.interpretation}")
                if s.examples:
                    ex = s.examples[0]
                    lines.append(f"- Example: \"{ex['text'][:80]}...\"" if len(ex['text']) > 80 else f"- Example: \"{ex['text']}\"")
                lines.append("")

        # Growth areas section
        growth = self.get_growth_areas(3)
        lines.append("### Growth Opportunities")
        lines.append("")
        for g in growth:
            if g.score <= 3:
                lines.append(f"**{g.youth_friendly_name}** (Score: {g.score}/5)")
                lines.append(f"- {g.interpretation}")
                lines.append(f"- **Development Tip:** {g.development_tip}")
                lines.append("")

        return "\n".join(lines)

    def generate_personal_development_plan(self) -> str:
        """
        Generate a personal development plan based on habit analysis.

        Returns:
            Markdown formatted development plan
        """
        results = self.analyze_all_habits()
        growth_areas = self.get_growth_areas(3)

        lines = [
            "## Personal Development Plan",
            "",
            "Based on the 7 Habits analysis, here are specific actions for improvement:",
            "",
        ]

        for i, area in enumerate(growth_areas, 1):
            if area.score <= 3:
                lines.append(f"### Priority {i}: {area.youth_friendly_name}")
                lines.append("")
                lines.append(f"**Current Level:** {area.score}/5 ({area.count} observations)")
                lines.append("")
                lines.append("**Action Steps:**")
                lines.append(f"1. {area.development_tip}")
                lines.append(self._get_specific_action(area.habit))
                lines.append(self._get_practice_activity(area.habit))
                lines.append("")

        return "\n".join(lines)

    def _get_specific_action(self, habit: SevenHabit) -> str:
        """Get specific action for a habit."""
        actions = {
            SevenHabit.BE_PROACTIVE: "2. During the next mission, volunteer for at least one task without being asked.",
            SevenHabit.BEGIN_WITH_END_IN_MIND: "2. Before undocking, state the mission objective out loud to the team.",
            SevenHabit.PUT_FIRST_THINGS_FIRST: "2. When multiple alerts occur, verbally identify which is most critical.",
            SevenHabit.THINK_WIN_WIN: "2. Offer to help a teammate with their station task at least once per mission.",
            SevenHabit.SEEK_FIRST_TO_UNDERSTAND: "2. Before responding to an order, repeat it back to confirm understanding.",
            SevenHabit.SYNERGIZE: "2. When facing a challenge, ask 'Does anyone have ideas?' before deciding alone.",
            SevenHabit.SHARPEN_THE_SAW: "2. At mission end, share one thing you learned with the team.",
        }
        return actions.get(habit, "2. Practice this habit consciously during the next mission.")

    def _get_practice_activity(self, habit: SevenHabit) -> str:
        """Get practice activity for a habit."""
        activities = {
            SevenHabit.BE_PROACTIVE: "3. **Practice:** In daily life, catch yourself saying 'I have to' and change it to 'I choose to.'",
            SevenHabit.BEGIN_WITH_END_IN_MIND: "3. **Practice:** Before starting homework, write down what 'done' looks like.",
            SevenHabit.PUT_FIRST_THINGS_FIRST: "3. **Practice:** Make a to-do list and mark the top 3 most important items.",
            SevenHabit.THINK_WIN_WIN: "3. **Practice:** Next time you disagree with someone, find one thing you both agree on first.",
            SevenHabit.SEEK_FIRST_TO_UNDERSTAND: "3. **Practice:** In your next conversation, ask two questions before sharing your opinion.",
            SevenHabit.SYNERGIZE: "3. **Practice:** On a group project, combine ideas from at least two people into your solution.",
            SevenHabit.SHARPEN_THE_SAW: "3. **Practice:** Keep a 'lessons learned' journal - write one thing after each activity.",
        }
        return activities.get(habit, "3. **Practice:** Look for opportunities to demonstrate this habit daily.")

    def get_structured_results(self) -> Dict[str, Any]:
        """
        Get structured results suitable for LLM prompts.

        Returns:
            Dictionary with all 7 Habits analysis data
        """
        results = self.analyze_all_habits()

        return {
            'habits_section': self.generate_habits_section(),
            'development_plan': self.generate_personal_development_plan(),
            'habits': {
                habit.name: {
                    'habit_number': habit.value,
                    'youth_name': assessment.youth_friendly_name,
                    'score': assessment.score,
                    'count': assessment.count,
                    'frequency': assessment.frequency,
                    'interpretation': assessment.interpretation,
                    'development_tip': assessment.development_tip,
                    'examples': assessment.examples,
                    'pattern_breakdown': assessment.pattern_breakdown,
                    'speaker_contributions': assessment.speaker_contributions,
                    'gap_to_next_score': assessment.gap_to_next_score,
                }
                for habit, assessment in results.items()
            },
            'strengths': [
                {
                    'habit': s.habit.name,
                    'name': s.youth_friendly_name,
                    'score': s.score,
                    'interpretation': s.interpretation,
                    'speaker_contributions': s.speaker_contributions,
                }
                for s in self.get_strongest_habits(3)
            ],
            'growth_areas': [
                {
                    'habit': g.habit.name,
                    'name': g.youth_friendly_name,
                    'score': g.score,
                    'development_tip': g.development_tip,
                    'gap_to_next_score': g.gap_to_next_score,
                }
                for g in self.get_growth_areas(3)
            ],
            'overall_effectiveness_score': round(
                sum(a.score for a in results.values()) / len(results), 1
            ),
            # Deduplicated engagement rate: what % of utterances matched
            # at least one habit (each utterance counted only once)
            'unique_habit_utterances': len(
                getattr(self, '_matched_utterance_indices', set())
            ),
            'total_utterances': len(self.transcripts),
            'score_thresholds': "Score 5: ≥30% | Score 4: ≥20% | Score 3: ≥10% | Score 2: ≥5% | Score 1: <5%",
        }
