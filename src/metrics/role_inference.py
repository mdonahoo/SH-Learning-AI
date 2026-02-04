"""
Role inference engine for crew member analysis.

This module provides keyword-frequency-based role detection for bridge crew members,
matching the methodology described in the example mission debrief report.

BALANCED SCORING APPROACH:
--------------------------
The role inference system uses a BALANCED scoring approach that combines:

1. **Keyword Density** (normalized within speaker)
   - Matches-per-utterance ratios instead of absolute counts
   - Prevents length dependency: same density = same score regardless of clip length

2. **Keyword Diversity** (quality of evidence)
   - Tracks distinct keyword types per role
   - More variety = stronger evidence (e.g., "course", "heading", "warp" > just "course")

3. **Speaker Prominence** (conversation share)
   - What percentage of total conversation does this speaker represent?
   - Critical for command roles: captains typically dominate conversation
   - A speaker with 40% conversation share + moderate keywords = likely captain
   - A speaker with 5% share + same keywords = likely crew member

ROLE-SPECIFIC WEIGHTING:
- **Command roles (Captain, XO)**: prominence matters more (30% weight)
  Score = density*35% + diversity*35% + prominence*30%

- **Crew roles (Helm, Tactical, etc.)**: keywords matter more (15% prominence)
  Score = density*45% + diversity*40% + prominence*15%

CONSISTENCY GUARANTEES:
- Short clips produce same results as full sessions (normalized metrics)
- Prominent speakers aren't ignored just because they have fewer keywords per utterance
- Command roles require both keywords AND conversation prominence
- Role conflicts resolved using balanced scores, not absolute counts
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum

logger = logging.getLogger(__name__)


class BridgeRole(Enum):
    """Standard bridge roles in Starship Horizons."""
    CAPTAIN = "Captain/Command"
    EXECUTIVE_OFFICER = "Executive Officer/Support"
    HELM = "Helm/Navigation"
    TACTICAL = "Tactical/Weapons"
    SCIENCE = "Science/Sensors"
    ENGINEERING = "Engineering/Systems"
    OPERATIONS = "Operations/Monitoring"
    COMMUNICATIONS = "Communications"
    UNKNOWN = "Crew Member"


@dataclass
class RolePatterns:
    """Keyword patterns for role detection."""

    # Patterns that indicate someone is ADDRESSING a superior (NOT being the captain)
    # These should REDUCE captain score - the speaker is talking TO the captain
    # IMPORTANT: Standalone acknowledgments like "Acknowledged" DON'T count - captains say this too
    # Only count as addressing authority when there's an explicit title (sir, captain, ma'am)
    ADDRESSING_AUTHORITY_PATTERNS: List[str] = field(default_factory=lambda: [
        # Starting sentence with title = clearly addressing (most reliable)
        r"(?i)^(captain|cap|sir|ma'am|commander|skipper)\b",

        # Acknowledgment WITH explicit title = crew responding to captain
        # Note: "Acknowledged" alone does NOT count - captains acknowledge too
        r"(?i)^(yes|aye|acknowledged|understood|copy|roger|affirmative)\s*,?\s*(sir|captain|cap|ma'am)\b",
        r"(?i)\baye\s+aye\b",  # "Aye aye" is specifically subordinate

        # Reporting status TO captain (requires title)
        r"(?i)\b(captain|cap|sir),?\s+(we have|we've got|there's|i'm reading|i'm detecting|i have)",
        r"(?i)\b(captain|cap|sir),?\s+(shields|weapons|sensors|engines|power|hull)",
        r"(?i)\b(captain|cap|sir),?\s+(enemy|target|contact|bogey|hostile)",
        r"(?i)\b(captain|cap|sir),?\s+(the|our|they|it)",

        # Asking/deferring to captain (requires title or explicit deference)
        r"(?i)\b(captain|cap|sir),?\s+(what|should|do you want|shall|orders)",
        r"(?i)\bpermission to\b",
        r"(?i)\bwaiting (for|on) (your|the captain|orders)",
        r"(?i)\b(your orders|awaiting orders)\b",

        # Deferring to/mentioning captain in third person
        r"(?i)\blet the captain\b",
        r"(?i)\btell the captain\b",
        r"(?i)\bask the captain\b",
        r"(?i)\bthe captain (said|wants|ordered|needs)\b",
    ])

    # Patterns that indicate COMMAND authority (being the captain)
    # These should be specific enough to not match crew members
    CAPTAIN_PATTERNS: List[str] = field(default_factory=lambda: [
        # Direct orders (high specificity)
        r"(?i)\b(set course|engage|execute|make it so|proceed|do it|go ahead)\b",
        r"(?i)\b(red alert|yellow alert|battle stations|stand down|at ease)\b",
        r"(?i)\b(all hands|attention crew|listen up everyone)\b",
        r"(?i)\bon screen\b",
        r"(?i)\bhail (them|the ship|that vessel)\b",
        r"(?i)\bopen (a )?channel\b",

        # Addressing specific stations by name (giving orders)
        r"(?i)\b(helm|tactical|science|engineering|ops|communications),?\s+(report|status|what|give me)",
        r"(?i)\b(helm|tactical|science|engineering),?\s+(set|target|scan|divert)",

        # Asking for status (command perspective)
        r"(?i)\bwhat('s| is) (the|our) (status|situation|position|eta)\b",
        r"(?i)\b(status report|damage report|give me a report)\b",
        r"(?i)\b(what do we (have|know|got)|what are we (looking at|dealing with))\b",

        # Command decisions
        r"(?i)\b(fire at will|weapons free|hold fire|cease fire)\b",
        r"(?i)\b(evasive maneuvers|evasive pattern)\b",
        r"(?i)\b(take us (in|out|to)|bring us about)\b",
        r"(?i)\b(approved|permission granted|denied|negative on that)\b",

        # Praise/feedback to crew (command perspective)
        r"(?i)\b(good work|well done|excellent work|nice (job|work|shot))\b",
        r"(?i)\b(good call|good thinking|that's (good|right|correct))\b",

        # Command phrases - decision making language
        r"(?i)^(alright|okay|right),?\s+(let's|we need|here's what)",
        r"(?i)^(everyone|all right everyone|okay everyone)\b",
        r"(?i)\b(let's|we('ll| will| need to| should| can))\s+(go|do|try|head|move|attack|defend|wait)\b",
        r"(?i)\b(i want|i need|get me|give me)\s+(a|the|those|some|more)\b",
        r"(?i)\b(our (mission|objective|goal|priority|target) is)\b",
        r"(?i)\b(here's (the|our) plan|the plan is|we're going to)\b",
        r"(?i)\b(focus on|concentrate on|prioritize)\b",

        # Captain-specific questions (command perspective)
        r"(?i)\b(can we|are we able to|do we have enough)\b",
        r"(?i)\b(how (long|far|much|many)|what's (the|our) (range|distance|time))\b",
        r"(?i)\b(options\??|what are our options|recommendations\??)\b",

        # Giving the order to start/stop
        r"(?i)^(go|stop|wait|hold|now|fire|launch)\b",
        r"(?i)\b(on my (mark|command|order|signal))\b",
        r"(?i)\b(that's (an order|enough)|belay that)\b",
    ])

    HELM_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)\b(course laid in|course set|heading|bearing)\b",
        r"(?i)\b(impulse|warp|full stop|all stop)\b",
        r"(?i)\b(eta|arrival|distance to|kilometers away)\b",
        r"(?i)\b(approach|approaching|orbit|docking)\b",
        r"(?i)\b(evasive|maneuver|turn|rotate)\b",
        r"(?i)\b(navigation|nav|plotting|plot course)\b",
        # Starship Horizons game-specific helm actions
        r"(?i)\b(head to|headed to|heading to|heading for)\b",
        r"(?i)\b(stick around|moving to|move to|fly to)\b",
        r"(?i)\b(outpost|waypoint|sector)\s+\w",
    ])

    TACTICAL_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)\b(targeting|target locked|acquiring|locked on)\b",
        r"(?i)\b(weapons|torpedoes|phasers|missiles)\b",
        r"(?i)\b(shields|shield status|shields at|hull)\b",
        r"(?i)\b(firing|fire|launch|launching)\b",
        r"(?i)\b(enemy|hostile|threat|contact|bogey)\b",
        r"(?i)\b(damage|hit|impact|taking fire)\b",
        # Starship Horizons game-specific tactical
        r"(?i)\b(turret|sentry|deploy)\b",
        r"(?i)\b(combat|attack|defend)\b",
    ])

    SCIENCE_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)\b(scanning|scan complete|sensors|detecting)\b",
        r"(?i)\b(reading|readings|analysis|analyzing)\b",
        r"(?i)\b(anomaly|signal|signature|emissions)\b",
        r"(?i)\b(life signs|life forms|biosigns)\b",
        r"(?i)\b(composition|spectrum|radiation)\b",
        r"(?i)\b(data|research|scientific)\b",
        # Starship Horizons game-specific science
        r"(?i)\b(scan|ended scan|boost.*(sensor|scan))\b",
        r"(?i)\b(sciences?\s+(has|have|granted|completed))\b",
    ])

    ENGINEERING_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)\b(reactor|power levels|power at|warp core)\b",
        r"(?i)\b(rerouting|diverting|transferring) power\b",
        r"(?i)\b(damage control|repairs|repairing)\b",
        r"(?i)\b(systems|subsystems|online|offline)\b",
        r"(?i)\b(coolant|overload|overheating|venting)\b",
        r"(?i)\b(efficiency|output|capacity)\b",
        # Starship Horizons game-specific engineering
        r"(?i)\b(warp core breach|restart|lattice|alignment)\b",
        r"(?i)\b(boost|power).*(engines?|shields?|sensors?)\b",
        r"(?i)\b(mini[- ]?game|puzzle)\b",
    ])

    OPERATIONS_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)\b(sector|quadrant|grid|coordinates)\b",
        r"(?i)\b(monitoring|tracking|observing)\b",
        r"(?i)\b(cargo|supplies|inventory|manifest)\b",
        r"(?i)\b(docking|dock|bay|hangar)\b",
        r"(?i)\b(schedule|timing|countdown)\b",
        # Starship Horizons game-specific operations
        r"(?i)\b(credits?|alliance|resources?)\b",
        r"(?i)\b(capture|captured|outpost)\b",
        r"(?i)\b(assist|granted)\b",
    ])

    COMMUNICATIONS_PATTERNS: List[str] = field(default_factory=lambda: [
        r"(?i)\b(hailing|hail|channel open|frequency)\b",
        r"(?i)\b(transmitting|transmission|receiving|signal)\b",
        r"(?i)\b(message|incoming|outgoing)\b",
        r"(?i)\b(audio|visual|subspace)\b",
        r"(?i)\b(broadcast|distress|mayday)\b",
    ])

    # Self-identification patterns: when a speaker claims or confirms their station
    # These get a strong boost because they're direct evidence of role assignment
    # IMPORTANT: Patterns must be specific to CLAIMING a station, not just mentioning it.
    # Exclude mini-game references ("take engineering as my mini game") and
    # questions about stations ("who's at flight?").
    SELF_IDENTIFICATION_PATTERNS: Dict[str, List[str]] = field(default_factory=lambda: {
        "Captain/Command": [
            r"(?i)\bi('m| am) (?:the )?(?:captain|commanding|in command)\b",
        ],
        "Helm/Navigation": [
            # "I'll be helm" / "I'll take flight" — but NOT "take X as my mini game"
            r"(?i)\bi('ll| will) (?:take |be )(?:the )?(?:helm|flight|navigation|nav)\b(?!.*mini[ -]?game)",
            # Confirming station: "Helm, yes" / "Flight, here"
            r"(?i)^(?:helm|flight|nav(?:igation)?),?\s*(?:yes|here|ready|aye)\b",
        ],
        "Tactical/Weapons": [
            r"(?i)\bi('ll| will) (?:take |be )(?:the )?(?:tactical|weapons|gunnery)\b(?!.*mini[ -]?game)",
            r"(?i)^(?:tactical|weapons),?\s*(?:yes|here|ready|aye)\b",
        ],
        "Science/Sensors": [
            r"(?i)\bi('ll| will) (?:take |be )(?:the )?(?:science|sciences|sensors?)\b(?!.*mini[ -]?game)",
            r"(?i)^(?:science|sciences?),?\s*(?:yes|here|ready|aye)\b",
        ],
        "Engineering/Systems": [
            r"(?i)\bi('ll| will) (?:take |be )(?:the )?(?:engineering|engineer)\b(?!.*mini[ -]?game)",
            r"(?i)^(?:engineering|engineer),?\s*(?:yes|here|ready|aye)\b",
        ],
        "Operations/Monitoring": [
            r"(?i)\bi('ll| will) (?:take |be )(?:the )?(?:operations?|ops)\b(?!.*mini[ -]?game)",
            r"(?i)^(?:operations?|ops),?\s*(?:yes|here|ready|aye)\b",
        ],
        "Communications": [
            r"(?i)\bi('ll| will) (?:take |be )(?:the )?(?:comms?|communications?)\b(?!.*mini[ -]?game)",
            r"(?i)^(?:comms?|communications?),?\s*(?:yes|here|ready|aye)\b",
        ],
    })

    # XO patterns should be SPECIFIC to XO role, not generic acknowledgments
    # Generic "aye", "copy", "roger" are used by everyone - don't count these
    # XO specifically: relays orders, coordinates stations, backs up captain
    EXECUTIVE_OFFICER_PATTERNS: List[str] = field(default_factory=lambda: [
        # Relaying/coordinating between stations (XO-specific)
        r"(?i)\b(helm|tactical|science|engineering),?\s+(you heard|the captain said|captain wants)",
        r"(?i)\b(relay|pass (it|that) on|inform|notify)\b",
        r"(?i)\b(coordinate|coordinating) with\b",

        # Supporting captain explicitly
        r"(?i)\b(i('ll| will) take|i have) (the bridge|command|the conn)\b",
        r"(?i)\b(backing you up|right behind you)\b",
        r"(?i)\b(i'll handle|taking care of|i've got) (that|it|this)\b",

        # Delegation from captain
        r"(?i)\b(you have the (bridge|conn)|take (the bridge|command))\b",
        r"(?i)\b(as you were|carry on)\b",

        # XO reporting to captain (different from crew reporting)
        r"(?i)\b(all stations report|stations report)\b",
        r"(?i)\b(crew is ready|we're ready captain)\b",
    ])


@dataclass
class SpeakerRoleAnalysis:
    """Analysis results for a single speaker's role."""
    speaker: str
    inferred_role: BridgeRole
    confidence: float
    utterance_count: int
    utterance_percentage: float
    keyword_matches: Dict[str, int]
    total_keyword_matches: int
    key_indicators: List[str]
    example_utterances: List[Dict[str, Any]]
    methodology_notes: str


class RoleInferenceEngine:
    """
    Engine for inferring bridge crew roles from transcript analysis.

    Uses BALANCED scoring combining keyword analysis AND speaker prominence
    to assign probable roles to speakers.

    Key Features:
    - Keyword density scoring (matches per utterance, not absolute counts)
    - Keyword diversity tracking (distinct keyword types per role)
    - Speaker prominence weighting (conversation share matters for command roles)
    - Role-specific weighting (captains need prominence, crew need keywords)
    - Addressing pattern detection (identifies when someone defers to authority)
    - Conflict resolution using balanced confidence scores

    The balanced approach ensures:
    - Consistent results regardless of audio recording length
    - Command roles require both keywords AND conversation dominance
    - Crew roles can be identified even with lower speaking volume
    - A 40% speaker with moderate keywords beats a 5% speaker with high density
    """

    def __init__(
        self,
        transcripts: List[Dict[str, Any]],
        patterns: Optional[RolePatterns] = None
    ):
        """
        Initialize the role inference engine.

        Args:
            transcripts: List of transcript dictionaries with 'speaker' and 'text'
            patterns: Optional custom role patterns
        """
        self.transcripts = transcripts
        self.patterns = patterns or RolePatterns()
        self._role_pattern_map = self._build_role_pattern_map()
        self._addressing_patterns = self._build_addressing_patterns()
        self._self_id_patterns = self._build_self_identification_patterns()

    def _build_role_pattern_map(self) -> Dict[BridgeRole, List[str]]:
        """Build mapping of roles to their detection patterns."""
        return {
            BridgeRole.CAPTAIN: self.patterns.CAPTAIN_PATTERNS,
            BridgeRole.HELM: self.patterns.HELM_PATTERNS,
            BridgeRole.TACTICAL: self.patterns.TACTICAL_PATTERNS,
            BridgeRole.SCIENCE: self.patterns.SCIENCE_PATTERNS,
            BridgeRole.ENGINEERING: self.patterns.ENGINEERING_PATTERNS,
            BridgeRole.OPERATIONS: self.patterns.OPERATIONS_PATTERNS,
            BridgeRole.COMMUNICATIONS: self.patterns.COMMUNICATIONS_PATTERNS,
            BridgeRole.EXECUTIVE_OFFICER: self.patterns.EXECUTIVE_OFFICER_PATTERNS,
        }

    def _build_addressing_patterns(self) -> List[re.Pattern]:
        """Build compiled patterns for detecting when someone is addressing authority."""
        return [re.compile(p) for p in self.patterns.ADDRESSING_AUTHORITY_PATTERNS]

    def _build_self_identification_patterns(self) -> Dict[BridgeRole, List[re.Pattern]]:
        """Build compiled patterns for detecting self-identification with a role."""
        role_name_map = {
            "Captain/Command": BridgeRole.CAPTAIN,
            "Helm/Navigation": BridgeRole.HELM,
            "Tactical/Weapons": BridgeRole.TACTICAL,
            "Science/Sensors": BridgeRole.SCIENCE,
            "Engineering/Systems": BridgeRole.ENGINEERING,
            "Operations/Monitoring": BridgeRole.OPERATIONS,
            "Communications": BridgeRole.COMMUNICATIONS,
        }
        result: Dict[BridgeRole, List[re.Pattern]] = {}
        for role_name, patterns in self.patterns.SELF_IDENTIFICATION_PATTERNS.items():
            bridge_role = role_name_map.get(role_name)
            if bridge_role:
                result[bridge_role] = [re.compile(p) for p in patterns]
        return result

    def analyze_all_speakers(self) -> Dict[str, SpeakerRoleAnalysis]:
        """
        Analyze all speakers and infer their roles.

        Returns:
            Dictionary mapping speaker IDs to their role analysis
        """
        # Count utterances per speaker
        speaker_utterances = defaultdict(list)
        for t in self.transcripts:
            speaker = t.get('speaker') or t.get('speaker_id') or 'unknown'
            speaker_utterances[speaker].append(t)

        total_utterances = len(self.transcripts)
        results = {}

        for speaker, utterances in speaker_utterances.items():
            results[speaker] = self._analyze_speaker(
                speaker, utterances, total_utterances
            )

        # Post-process to resolve role conflicts
        results = self._resolve_role_conflicts(results)

        # Apply addressing pattern penalties
        results = self._apply_addressing_penalties(results, speaker_utterances)

        return results

    # Minimum requirements for role assignment
    MIN_UTTERANCES_FOR_ROLE = 2  # Need at least 2 utterances (lowered for short clips)
    MIN_KEYWORD_TYPES = 1  # Need at least 1 keyword type (lowered, prominence can compensate)

    # Role-specific prominence expectations (what % of conversation we expect)
    ROLE_PROMINENCE_WEIGHTS = {
        BridgeRole.CAPTAIN: 0.4,           # Captains typically dominate conversation
        BridgeRole.EXECUTIVE_OFFICER: 0.3,  # XOs are also prominent
        BridgeRole.HELM: 0.15,
        BridgeRole.TACTICAL: 0.15,
        BridgeRole.SCIENCE: 0.15,
        BridgeRole.ENGINEERING: 0.15,
        BridgeRole.OPERATIONS: 0.1,
        BridgeRole.COMMUNICATIONS: 0.1,
    }

    def _analyze_speaker(
        self,
        speaker: str,
        utterances: List[Dict[str, Any]],
        total_utterances: int
    ) -> SpeakerRoleAnalysis:
        """
        Analyze a single speaker's communications.

        Uses BALANCED scoring combining:
        - Keyword density (normalized within speaker) - prevents length dependency
        - Speaker prominence (% of total conversation) - captures volume signal
        - Keyword diversity (distinct types) - quality of evidence

        This ensures consistent results while still recognizing that a captain
        who speaks 40% of the time is more likely than one who speaks 5%.
        """
        utterance_count = len(utterances)
        utterance_pct = (utterance_count / total_utterances * 100) if total_utterances > 0 else 0
        prominence = utterance_count / total_utterances if total_utterances > 0 else 0

        # Count keyword matches per role
        role_match_counts: Dict[BridgeRole, int] = defaultdict(int)
        keyword_matches: Dict[str, int] = defaultdict(int)
        matched_keywords: Dict[BridgeRole, set] = defaultdict(set)
        utterances_with_keywords: Dict[BridgeRole, set] = defaultdict(set)
        addressing_count = 0
        self_identified_roles: Dict[BridgeRole, List[str]] = defaultdict(list)

        for idx, utterance in enumerate(utterances):
            text = utterance.get('text', '')

            # Check if addressing authority
            is_addressing = False
            for pattern in self._addressing_patterns:
                if pattern.search(text):
                    is_addressing = True
                    addressing_count += 1
                    break

            # Check for self-identification ("I'll take engineering", "Operations, yes")
            for role, patterns in self._self_id_patterns.items():
                for pattern in patterns:
                    if pattern.search(text):
                        self_identified_roles[role].append(text)
                        # Self-ID also counts as keyword evidence for the role
                        role_match_counts[role] += 3  # Strong weight
                        utterances_with_keywords[role].add(idx)
                        matched_keywords[role].add(f"self-id: {role.value}")
                        break

            for role, patterns in self._role_pattern_map.items():
                for pattern in patterns:
                    matches = re.findall(pattern, text)
                    if matches:
                        if is_addressing and role == BridgeRole.CAPTAIN:
                            continue

                        role_match_counts[role] += len(matches)
                        utterances_with_keywords[role].add(idx)

                        for match in matches:
                            match_text = match if isinstance(match, str) else match[0]
                            keyword_matches[match_text.lower()] += 1
                            matched_keywords[role].add(match_text.lower())

        # Calculate BALANCED scores combining density AND prominence
        role_scores: Dict[BridgeRole, float] = defaultdict(float)

        for role in role_match_counts:
            distinct_keywords = len(matched_keywords[role])
            utterances_with_role = len(utterances_with_keywords[role])

            if utterance_count > 0:
                # Component 1: Keyword density (normalized within speaker)
                density = utterances_with_role / utterance_count

                # Component 2: Keyword diversity (distinct types, capped)
                diversity = min(1.0, distinct_keywords / 5)

                # Component 3: Prominence fit - how well does speaker prominence
                # match expected prominence for this role?
                expected_prominence = self.ROLE_PROMINENCE_WEIGHTS.get(role, 0.15)

                # Prominence score: bonus if speaker prominence matches or exceeds expectation
                # For command roles, high prominence is a strong signal
                # For crew roles, moderate prominence is fine
                if prominence >= expected_prominence:
                    prominence_score = 1.0
                elif prominence >= expected_prominence * 0.5:
                    prominence_score = 0.7
                else:
                    prominence_score = 0.4

                # Special handling for command roles - prominence matters more
                if role in (BridgeRole.CAPTAIN, BridgeRole.EXECUTIVE_OFFICER):
                    # Command roles: prominence is critical
                    # density=35%, diversity=35%, prominence=30%
                    role_scores[role] = (
                        density * 0.35 +
                        diversity * 0.35 +
                        prominence_score * 0.30
                    )
                    # Extra bonus for prominent speakers with command keywords
                    # Lowered thresholds: prominence > 20% and density > 10%
                    if prominence > 0.20 and density > 0.10:
                        role_scores[role] = min(1.0, role_scores[role] + 0.15)
                    # Strong boost for very prominent speakers (30%+) with any command evidence
                    # The person talking most is often the captain
                    if prominence > 0.30 and distinct_keywords >= 1:
                        role_scores[role] = min(1.0, role_scores[role] + 0.20)
                else:
                    # Crew roles: density and diversity matter more than prominence
                    # density=45%, diversity=40%, prominence=15%
                    role_scores[role] = (
                        density * 0.45 +
                        diversity * 0.40 +
                        prominence_score * 0.15
                    )

        # Apply self-identification boost — strong evidence of role assignment
        for role, id_texts in self_identified_roles.items():
            if role in role_scores:
                role_scores[role] = min(1.0, role_scores[role] + 0.35)
            else:
                # Self-ID alone is strong enough to create a score from nothing
                role_scores[role] = 0.50
            # Ensure the role has keyword evidence for minimum threshold checks
            matched_keywords[role].add(f"self-id: {role.value}")
            logger.info(
                f"Self-identification boost for {speaker} -> {role.value}: "
                f'"{id_texts[0][:60]}"'
            )

        # Determine primary role
        addressing_ratio = addressing_count / utterance_count if utterance_count > 0 else 0

        inferred_role = BridgeRole.UNKNOWN
        confidence = 0.0

        # Sort by balanced score
        sorted_roles = sorted(role_scores.items(), key=lambda x: -x[1])

        # IMPORTANT: For very prominent speakers, prefer Captain over XO
        # The most prominent speaker is usually the captain, not XO
        # XO patterns are generic acknowledgments that captains also use
        captain_keywords = len(matched_keywords.get(BridgeRole.CAPTAIN, set()))
        xo_score = role_scores.get(BridgeRole.EXECUTIVE_OFFICER, 0)
        captain_score = role_scores.get(BridgeRole.CAPTAIN, 0)

        if sorted_roles and sorted_roles[0][1] > 0:
            top_role = sorted_roles[0][0]
            top_score = sorted_roles[0][1]
            second_score = sorted_roles[1][1] if len(sorted_roles) > 1 else 0

            # Override: Prominent speaker with Captain keywords should be Captain, not XO
            # This handles the case where XO generic patterns inflate XO score
            if top_role == BridgeRole.EXECUTIVE_OFFICER and prominence > 0.25:
                if captain_keywords >= 1 and captain_score > 0:
                    # Swap to Captain - the most talkative person with command keywords is captain
                    top_role = BridgeRole.CAPTAIN
                    top_score = captain_score
                    logger.debug(
                        f"Promoted {speaker} from XO to Captain: "
                        f"prominence={prominence:.0%}, captain_keywords={captain_keywords}"
                    )

            # Also promote if speaker is THE dominant voice (30%+) and has any command evidence
            if top_role == BridgeRole.EXECUTIVE_OFFICER and prominence > 0.30:
                if captain_score > 0:
                    top_role = BridgeRole.CAPTAIN
                    top_score = max(captain_score, top_score)
                    logger.debug(
                        f"Promoted dominant speaker {speaker} to Captain: prominence={prominence:.0%}"
                    )

            # Check minimum evidence thresholds
            distinct_keyword_count = len(matched_keywords.get(top_role, set()))
            keyword_density = len(utterances_with_keywords.get(top_role, set())) / utterance_count if utterance_count > 0 else 0

            # Relaxed thresholds - prominence can compensate for fewer keywords
            meets_minimum = (
                utterance_count >= self.MIN_UTTERANCES_FOR_ROLE and
                distinct_keyword_count >= self.MIN_KEYWORD_TYPES
            )

            # For prominent speakers (15%+), be more lenient on keyword requirements
            if prominence > 0.15 and distinct_keyword_count >= 1:
                meets_minimum = True

            # For command roles with very high prominence (25%+), even 1 keyword is sufficient
            # The most talkative person is often the captain
            if top_role in (BridgeRole.CAPTAIN, BridgeRole.EXECUTIVE_OFFICER):
                if prominence > 0.25 and distinct_keyword_count >= 1:
                    meets_minimum = True

            # Self-identification always meets minimum — direct evidence of role
            if top_role in self_identified_roles:
                meets_minimum = True

            if meets_minimum:
                inferred_role = top_role

                # Calculate confidence based on balanced score and dominance
                if top_score > 0:
                    dominance = (top_score - second_score) / top_score if top_score > second_score else 0

                    # Base confidence on balanced score
                    confidence = min(0.95, top_score * 0.7 + dominance * 0.3)

                    # Boost for strong evidence
                    if distinct_keyword_count >= 4:
                        confidence = min(0.95, confidence + 0.08)
                    if keyword_density >= 0.4:
                        confidence = min(0.95, confidence + 0.08)
                    if prominence > 0.3 and top_role in (BridgeRole.CAPTAIN, BridgeRole.EXECUTIVE_OFFICER):
                        confidence = min(0.95, confidence + 0.1)
                    # Self-identification is strong direct evidence
                    if top_role in self_identified_roles:
                        confidence = min(0.95, confidence + 0.15)

                # Handle captain addressing pattern
                if addressing_ratio > 0.3 and inferred_role == BridgeRole.CAPTAIN:
                    if len(sorted_roles) > 1 and sorted_roles[1][1] > 0.1:
                        second_role = sorted_roles[1][0]
                        second_distinct = len(matched_keywords.get(second_role, set()))
                        if second_distinct >= self.MIN_KEYWORD_TYPES:
                            inferred_role = second_role
                            confidence *= 0.7
                    else:
                        inferred_role = BridgeRole.EXECUTIVE_OFFICER
                        confidence = 0.4
            else:
                inferred_role = BridgeRole.UNKNOWN
                confidence = 0.0

        # Build key indicators
        top_keywords = sorted(keyword_matches.items(), key=lambda x: -x[1])[:5]
        key_indicators = [f'"{kw}" ({count})' for kw, count in top_keywords]

        # Add self-identification as a key indicator
        for role, id_texts in self_identified_roles.items():
            snippet = id_texts[0][:40]
            key_indicators.insert(0, f'SELF-ID {role.value}: "{snippet}"')

        if addressing_count > 2:
            key_indicators.append(f"addresses authority ({addressing_count}x)")

        # Add evidence summary with prominence
        if inferred_role != BridgeRole.UNKNOWN:
            distinct_count = len(matched_keywords.get(inferred_role, set()))
            density_pct = len(utterances_with_keywords.get(inferred_role, set())) / utterance_count * 100 if utterance_count > 0 else 0
            key_indicators.append(f"{distinct_count} types, {density_pct:.0f}% density, {utterance_pct:.0f}% of convo")

        # Select example utterances
        example_utterances = []
        for u in sorted(utterances, key=lambda x: x.get('confidence', 0), reverse=True)[:5]:
            example_utterances.append({
                'timestamp': u.get('timestamp', ''),
                'text': u.get('text', ''),
                'confidence': u.get('confidence', 0)
            })

        # Generate methodology note
        methodology = self._generate_methodology_note(
            speaker, inferred_role, utterance_count, utterance_pct,
            {r: role_match_counts[r] for r in role_match_counts},
            sum(role_match_counts.values()),
            {r: list(kws) for r, kws in matched_keywords.items()},
            addressing_count,
            len(matched_keywords.get(inferred_role, set())) if inferred_role != BridgeRole.UNKNOWN else 0,
            len(utterances_with_keywords.get(inferred_role, set())) / utterance_count if utterance_count > 0 and inferred_role != BridgeRole.UNKNOWN else 0,
            prominence
        )

        return SpeakerRoleAnalysis(
            speaker=speaker,
            inferred_role=inferred_role,
            confidence=round(confidence, 2),
            utterance_count=utterance_count,
            utterance_percentage=round(utterance_pct, 1),
            keyword_matches=dict(keyword_matches),
            total_keyword_matches=sum(role_match_counts.values()),
            key_indicators=key_indicators,
            example_utterances=example_utterances,
            methodology_notes=methodology
        )

    def _generate_methodology_note(
        self,
        speaker: str,
        role: BridgeRole,
        utterance_count: int,
        utterance_pct: float,
        role_scores: Dict[BridgeRole, int],
        total_matches: int,
        matched_keywords: Dict[BridgeRole, List[str]],
        addressing_count: int = 0,
        distinct_keyword_count: int = 0,
        keyword_density: float = 0.0,
        prominence: float = 0.0
    ) -> str:
        """
        Generate explanation of role assignment methodology.

        Explains how keyword density, diversity, AND speaker prominence
        contributed to the role assignment.
        """
        if role == BridgeRole.UNKNOWN:
            reasons = []
            if utterance_count < self.MIN_UTTERANCES_FOR_ROLE:
                reasons.append(f"only {utterance_count} utterances (minimum {self.MIN_UTTERANCES_FOR_ROLE} required)")
            if distinct_keyword_count < self.MIN_KEYWORD_TYPES:
                reasons.append(f"no clear role keywords detected")

            if reasons:
                return (f"{speaker} had {utterance_count} utterances ({utterance_pct:.1f}% of conversation) "
                       f"but insufficient evidence: {'; '.join(reasons)}.")
            else:
                return (f"{speaker} had {utterance_count} utterances ({utterance_pct:.1f}% of conversation) "
                       f"but no clear role indicators were detected.")

        role_keywords = matched_keywords.get(role, [])
        keyword_sample = ', '.join(f'"{kw}"' for kw in role_keywords[:5])

        # Build explanation with all three factors
        factors = []

        # Factor 1: Keywords
        if distinct_keyword_count > 0:
            factors.append(f"{distinct_keyword_count} keyword types ({keyword_sample})")

        # Factor 2: Density
        if keyword_density > 0:
            factors.append(f"{keyword_density:.0%} keyword density")

        # Factor 3: Prominence
        if prominence > 0.25:
            factors.append(f"high speaking volume ({utterance_pct:.0f}% of conversation)")
        elif prominence > 0.15:
            factors.append(f"moderate speaking volume ({utterance_pct:.0f}%)")

        factors_str = ", ".join(factors) if factors else "pattern analysis"
        note = f"{speaker} assigned {role.value} based on {factors_str}."

        # Add context for command roles and prominence
        if role in (BridgeRole.CAPTAIN, BridgeRole.EXECUTIVE_OFFICER):
            if prominence > 0.3:
                note += " High conversation share strongly supports command role."
            elif prominence > 0.2:
                note += " Conversation share consistent with command role."

        # Add context for addressing patterns
        if addressing_count > 0 and utterance_count > 0:
            addressing_pct = addressing_count / utterance_count * 100
            if addressing_pct > 30:
                note += f" Frequently addresses authority ({addressing_pct:.0f}%), suggesting crew member."
            elif addressing_pct > 10:
                note += f" Sometimes addresses superiors ({addressing_pct:.0f}%)."

        # Evidence strength summary
        evidence_strength = 0
        if distinct_keyword_count >= 3:
            evidence_strength += 1
        if keyword_density >= 0.3:
            evidence_strength += 1
        if prominence > 0.2:
            evidence_strength += 1

        if evidence_strength >= 3:
            note += " Strong multi-factor evidence."
        elif evidence_strength >= 2:
            note += " Good supporting evidence."

        return note

    def _resolve_role_conflicts(
        self,
        results: Dict[str, SpeakerRoleAnalysis]
    ) -> Dict[str, SpeakerRoleAnalysis]:
        """
        Resolve conflicts when multiple speakers are assigned the same role.

        Uses BALANCED scoring (confidence + prominence) for consistent resolution.
        For command roles, prominence is weighted more heavily in tie-breakers.

        The speaker with the highest combined score keeps the role; others are
        reassigned to their next best role or marked as support.
        """
        # Group by inferred role
        role_assignments: Dict[BridgeRole, List[SpeakerRoleAnalysis]] = defaultdict(list)
        for analysis in results.values():
            role_assignments[analysis.inferred_role].append(analysis)

        # For roles with multiple speakers, keep the strongest match
        reassignments = {}
        for role, speakers in role_assignments.items():
            if role == BridgeRole.UNKNOWN:
                continue

            if len(speakers) > 1:
                # For command roles, weight prominence more heavily in conflict resolution
                if role in (BridgeRole.CAPTAIN, BridgeRole.EXECUTIVE_OFFICER):
                    # Combined score: confidence (60%) + prominence (40%)
                    sorted_speakers = sorted(
                        speakers,
                        key=lambda x: (x.confidence * 0.6 + (x.utterance_percentage / 100) * 0.4),
                        reverse=True
                    )
                else:
                    # Crew roles: confidence matters more
                    sorted_speakers = sorted(
                        speakers,
                        key=lambda x: (x.confidence * 0.8 + (x.utterance_percentage / 100) * 0.2),
                        reverse=True
                    )

                # First speaker keeps the role (highest combined score)
                primary = sorted_speakers[0]
                logger.debug(
                    f"Role conflict for {role.value}: {primary.speaker} keeps role "
                    f"(confidence: {primary.confidence:.0%}, prominence: {primary.utterance_percentage:.0f}%)"
                )

                # Others become support roles or unknown
                for secondary in sorted_speakers[1:]:
                    confidence_gap = primary.confidence - secondary.confidence
                    prominence_gap = primary.utterance_percentage - secondary.utterance_percentage

                    # Close match with good prominence -> XO
                    if confidence_gap < 0.15 and secondary.utterance_percentage > 15:
                        reassignments[secondary.speaker] = BridgeRole.EXECUTIVE_OFFICER
                        logger.debug(
                            f"  {secondary.speaker} -> XO (close confidence, good prominence)"
                        )
                    # High prominence but lower confidence -> XO
                    elif secondary.utterance_percentage > 20:
                        reassignments[secondary.speaker] = BridgeRole.EXECUTIVE_OFFICER
                        logger.debug(
                            f"  {secondary.speaker} -> XO (high prominence: {secondary.utterance_percentage:.0f}%)"
                        )
                    # Moderate prominence -> crew member
                    elif secondary.utterance_percentage > 10:
                        reassignments[secondary.speaker] = BridgeRole.UNKNOWN
                        logger.debug(
                            f"  {secondary.speaker} -> Unknown (moderate evidence)"
                        )
                    else:
                        # Low volume, lower confidence
                        reassignments[secondary.speaker] = BridgeRole.UNKNOWN

        # Apply reassignments with adjusted confidence
        for speaker, new_role in reassignments.items():
            old_analysis = results[speaker]

            # Calculate new confidence based on reassignment type and original strength
            if new_role == BridgeRole.EXECUTIVE_OFFICER:
                # XO gets decent confidence if they were close to primary
                new_confidence = min(0.7, old_analysis.confidence * 0.8)
            else:
                new_confidence = 0.0

            results[speaker] = SpeakerRoleAnalysis(
                speaker=old_analysis.speaker,
                inferred_role=new_role,
                confidence=round(new_confidence, 2),
                utterance_count=old_analysis.utterance_count,
                utterance_percentage=old_analysis.utterance_percentage,
                keyword_matches=old_analysis.keyword_matches,
                total_keyword_matches=old_analysis.total_keyword_matches,
                key_indicators=old_analysis.key_indicators,
                example_utterances=old_analysis.example_utterances,
                methodology_notes=old_analysis.methodology_notes +
                    f" (Reassigned from {old_analysis.inferred_role.value} - another speaker had stronger combined evidence.)"
            )

        return results

    # Patterns that indicate a speaker is ADDRESSING someone (not being that role)
    ADDRESSING_PATTERNS = {
        BridgeRole.CAPTAIN: [
            r'^captain[,\s]',
            r'^cap[,\s]',
            r'^sir[,\s]',
            r'\bcaptain[,\s]+(?:we|i|the|shields|weapons|engines)',
        ],
        BridgeRole.HELM: [
            r'^helm[,\s]',
            r'^navigation[,\s]',
        ],
        BridgeRole.TACTICAL: [
            r'^tactical[,\s]',
            r'^weapons[,\s]',
        ],
        BridgeRole.ENGINEERING: [
            r'^engineer(?:ing)?[,\s]',
        ],
        BridgeRole.SCIENCE: [
            r'^science[,\s]',
        ],
        BridgeRole.COMMUNICATIONS: [
            r'^comms?[,\s]',
            r'^communications[,\s]',
        ],
    }

    def _apply_addressing_penalties(
        self,
        results: Dict[str, SpeakerRoleAnalysis],
        speaker_utterances: Dict[str, List]
    ) -> Dict[str, SpeakerRoleAnalysis]:
        """
        Detect when speakers address specific roles and penalize misassignment.

        If a speaker frequently says "Captain, ..." they are NOT the Captain.
        This fixes misattributions like "Captain, I cannot fire" being labeled Captain.

        Args:
            results: Current role analysis results
            speaker_utterances: Mapping of speaker to their utterances

        Returns:
            Updated results with addressing penalties applied
        """
        import re

        for speaker, analysis in results.items():
            if analysis.inferred_role == BridgeRole.UNKNOWN:
                continue

            utterances = speaker_utterances.get(speaker, [])
            if not utterances:
                continue

            # Count how many times this speaker addresses their assigned role
            addressing_count = 0
            total_count = len(utterances)

            role_patterns = self.ADDRESSING_PATTERNS.get(analysis.inferred_role, [])
            if not role_patterns:
                continue

            for utt in utterances:
                text = (utt.get('text') or '').lower().strip()
                for pattern in role_patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        addressing_count += 1
                        break  # Count once per utterance

            # If speaker frequently addresses their assigned role, reassign them
            # Threshold: if >10% of utterances address the role, it's suspicious
            if total_count >= 3 and addressing_count / total_count > 0.10:
                old_role = analysis.inferred_role
                new_confidence = analysis.confidence * 0.5  # Halve confidence

                # Reassign to Crew Member
                results[speaker] = SpeakerRoleAnalysis(
                    speaker=analysis.speaker,
                    inferred_role=BridgeRole.UNKNOWN,
                    confidence=round(new_confidence, 2),
                    utterance_count=analysis.utterance_count,
                    utterance_percentage=analysis.utterance_percentage,
                    keyword_matches=analysis.keyword_matches,
                    total_keyword_matches=analysis.total_keyword_matches,
                    key_indicators=analysis.key_indicators,
                    example_utterances=analysis.example_utterances,
                    methodology_notes=analysis.methodology_notes +
                        f" (Reassigned from {old_role.value}: speaker frequently addresses {old_role.value} role, suggesting they are NOT that role.)"
                )

                logger.info(
                    f"Addressing penalty: {speaker} reassigned from {old_role.value} -> Crew Member "
                    f"({addressing_count}/{total_count} utterances address {old_role.value})"
                )

        return results

    def generate_role_analysis_table(self) -> str:
        """
        Generate a markdown table of role assignments.

        Returns:
            Markdown formatted table string
        """
        results = self.analyze_all_speakers()

        # Sort by utterance count (descending)
        sorted_results = sorted(
            results.values(),
            key=lambda x: x.utterance_count,
            reverse=True
        )

        lines = [
            "| Speaker | Utterances | Likely Role | Key Indicators |",
            "| --- | --- | --- | --- |"
        ]

        for analysis in sorted_results:
            indicators = ", ".join(analysis.key_indicators[:3]) if analysis.key_indicators else "No clear indicators"
            lines.append(
                f"| {analysis.speaker} | {analysis.utterance_count} | "
                f"{analysis.inferred_role.value} | {indicators} |"
            )

        return "\n".join(lines)

    def generate_methodology_section(self) -> str:
        """
        Generate the Role Assignment Methodology section for reports.

        Returns:
            Markdown formatted methodology explanation
        """
        results = self.analyze_all_speakers()

        # Sort by utterance count
        sorted_results = sorted(
            results.values(),
            key=lambda x: x.utterance_count,
            reverse=True
        )

        lines = ["### Role Assignment Methodology", ""]
        lines.append(
            "Role assignments are based on keyword frequency analysis across all utterances. "
            "Each speaker's communications were analyzed for patterns indicating specific bridge roles."
        )
        lines.append("")

        for analysis in sorted_results:
            if analysis.utterance_count > 0:
                lines.append(analysis.methodology_notes)
                lines.append("")

        return "\n".join(lines)

    def get_structured_results(self) -> Dict[str, Any]:
        """
        Get structured results suitable for LLM prompts.

        Returns:
            Dictionary with all role analysis data
        """
        results = self.analyze_all_speakers()

        return {
            'role_table': self.generate_role_analysis_table(),
            'methodology': self.generate_methodology_section(),
            'speaker_roles': {
                speaker: {
                    'role': analysis.inferred_role.value,
                    'confidence': analysis.confidence,
                    'utterance_count': analysis.utterance_count,
                    'utterance_percentage': analysis.utterance_percentage,
                    'keyword_matches': analysis.total_keyword_matches,
                    'key_indicators': analysis.key_indicators,
                    'methodology_note': analysis.methodology_notes
                }
                for speaker, analysis in results.items()
            }
        }


@dataclass
class VoicePatternMetrics:
    """Metrics derived from speaking patterns."""
    speaker: str
    avg_words_per_utterance: float
    utterance_count: int
    speaking_percentage: float
    command_ratio: float  # Ratio of imperative sentences
    question_ratio: float  # Ratio of questions asked
    avg_utterance_duration: float
    is_dominant_speaker: bool
    response_pattern: str  # 'initiator', 'responder', 'balanced'


class VoicePatternAnalyzer:
    """
    Analyzes speaking patterns to help identify bridge roles.

    Uses speech characteristics like command frequency, question ratio,
    and speaking dominance to infer roles without requiring keywords.
    """

    # Role-specific voice pattern expectations
    ROLE_PATTERNS = {
        BridgeRole.CAPTAIN: {
            'min_utterance_pct': 20,  # Captains speak a lot
            'command_ratio_weight': 0.8,  # High command ratio expected
            'question_ratio_weight': 0.3,  # Moderate questions
            'dominant_bonus': 0.2,  # Bonus for being dominant speaker
            'initiator_bonus': 0.15,  # Bonus for initiating conversations
        },
        BridgeRole.HELM: {
            'min_utterance_pct': 10,
            'command_ratio_weight': 0.2,  # Few commands
            'question_ratio_weight': 0.2,  # Few questions
            'confirmation_weight': 0.6,  # High confirmation rate
        },
        BridgeRole.TACTICAL: {
            'min_utterance_pct': 8,
            'command_ratio_weight': 0.4,  # Some commands
            'short_utterance_weight': 0.5,  # Short, decisive
            'alert_pattern_weight': 0.6,
        },
        BridgeRole.SCIENCE: {
            'min_utterance_pct': 8,
            'long_utterance_weight': 0.5,  # Detailed explanations
            'question_ratio_weight': 0.4,  # Asks about data
            'report_pattern_weight': 0.6,
        },
        BridgeRole.ENGINEERING: {
            'min_utterance_pct': 8,
            'report_pattern_weight': 0.6,  # Status reports
            'technical_weight': 0.5,
        },
    }

    # Patterns for command detection (imperative sentences)
    COMMAND_PATTERNS = [
        r"^(?:set|engage|fire|launch|raise|lower|divert|transfer|scan|target|lock|hold|stop|go|execute|initiate)\b",
        r"^(?:all hands|attention|battle stations|red alert|yellow alert)\b",
        r"\b(?:now|immediately|at once)$",
    ]

    # Patterns for confirmation/acknowledgment
    CONFIRMATION_PATTERNS = [
        r"^(?:aye|yes|affirmative|confirmed|acknowledged|copy|roger|understood)\b",
        r"^(?:course|heading|target|shields?|weapons?|power)\s+(?:set|locked|ready|online)\b",
    ]

    # Patterns for status reports
    REPORT_PATTERNS = [
        r"^(?:captain|sir|ma'am),?\s",
        r"\b(?:reading|detecting|showing|at|levels?)\s+\d",
        r"\b(?:status|report|analysis|scan)\s+(?:complete|ready|shows)\b",
    ]

    def __init__(self, transcripts: List[Dict[str, Any]]):
        """
        Initialize voice pattern analyzer.

        Args:
            transcripts: List of transcript dictionaries with speaker and text
        """
        self.transcripts = transcripts
        self._compiled_commands = [re.compile(p, re.IGNORECASE) for p in self.COMMAND_PATTERNS]
        self._compiled_confirms = [re.compile(p, re.IGNORECASE) for p in self.CONFIRMATION_PATTERNS]
        self._compiled_reports = [re.compile(p, re.IGNORECASE) for p in self.REPORT_PATTERNS]

    def analyze_speaker_patterns(self) -> Dict[str, VoicePatternMetrics]:
        """
        Analyze speaking patterns for all speakers.

        Returns:
            Dict mapping speaker ID to VoicePatternMetrics
        """
        speaker_utterances = defaultdict(list)
        for t in self.transcripts:
            speaker = t.get('speaker') or t.get('speaker_id') or 'unknown'
            speaker_utterances[speaker].append(t)

        total_utterances = len(self.transcripts)
        total_speakers = len(speaker_utterances)

        # Find dominant speaker
        max_utterances = max(len(u) for u in speaker_utterances.values()) if speaker_utterances else 0

        results = {}
        for speaker, utterances in speaker_utterances.items():
            results[speaker] = self._analyze_single_speaker(
                speaker, utterances, total_utterances, max_utterances
            )

        return results

    def _analyze_single_speaker(
        self,
        speaker: str,
        utterances: List[Dict[str, Any]],
        total_utterances: int,
        max_utterances: int
    ) -> VoicePatternMetrics:
        """Analyze patterns for a single speaker."""
        texts = [u.get('text', '') for u in utterances]
        utterance_count = len(utterances)

        # Calculate average words per utterance
        word_counts = [len(t.split()) for t in texts]
        avg_words = sum(word_counts) / len(word_counts) if word_counts else 0

        # Calculate speaking percentage
        speaking_pct = (utterance_count / total_utterances * 100) if total_utterances > 0 else 0

        # Count commands and questions
        command_count = 0
        question_count = 0
        confirm_count = 0
        report_count = 0

        for text in texts:
            # Check for commands
            for pattern in self._compiled_commands:
                if pattern.search(text):
                    command_count += 1
                    break

            # Check for questions
            if '?' in text or text.lower().startswith(('what', 'where', 'when', 'how', 'why', 'is', 'are', 'do', 'does', 'can', 'could', 'should')):
                question_count += 1

            # Check for confirmations
            for pattern in self._compiled_confirms:
                if pattern.search(text):
                    confirm_count += 1
                    break

            # Check for reports
            for pattern in self._compiled_reports:
                if pattern.search(text):
                    report_count += 1
                    break

        command_ratio = command_count / utterance_count if utterance_count > 0 else 0
        question_ratio = question_count / utterance_count if utterance_count > 0 else 0

        # Calculate average utterance duration if available
        total_duration = 0
        duration_count = 0
        for u in utterances:
            start = u.get('start_time', 0)
            end = u.get('end_time', 0)
            if end > start:
                total_duration += end - start
                duration_count += 1
        avg_duration = total_duration / duration_count if duration_count > 0 else 0

        # Determine if dominant speaker
        is_dominant = utterance_count == max_utterances and speaking_pct > 25

        # Determine response pattern based on conversation position
        # (This is simplified - a full analysis would look at timing)
        if command_ratio > 0.3:
            response_pattern = 'initiator'
        elif confirm_count / utterance_count > 0.3 if utterance_count > 0 else False:
            response_pattern = 'responder'
        else:
            response_pattern = 'balanced'

        return VoicePatternMetrics(
            speaker=speaker,
            avg_words_per_utterance=round(avg_words, 1),
            utterance_count=utterance_count,
            speaking_percentage=round(speaking_pct, 1),
            command_ratio=round(command_ratio, 2),
            question_ratio=round(question_ratio, 2),
            avg_utterance_duration=round(avg_duration, 2),
            is_dominant_speaker=is_dominant,
            response_pattern=response_pattern
        )

    def get_role_hints(self) -> Dict[str, List[Tuple[BridgeRole, float]]]:
        """
        Get role hints based on voice patterns.

        Returns:
            Dict mapping speaker ID to list of (role, confidence) tuples
        """
        patterns = self.analyze_speaker_patterns()
        hints = {}

        for speaker, metrics in patterns.items():
            role_scores = []

            # Captain hints
            captain_score = 0.0
            if metrics.is_dominant_speaker:
                captain_score += 0.2
            if metrics.command_ratio > 0.2:
                captain_score += min(0.3, metrics.command_ratio)
            if metrics.speaking_percentage > 25:
                captain_score += 0.15
            if metrics.response_pattern == 'initiator':
                captain_score += 0.1
            if captain_score > 0.2:
                role_scores.append((BridgeRole.CAPTAIN, min(0.6, captain_score)))

            # Helm hints
            helm_score = 0.0
            if metrics.speaking_percentage > 8 and metrics.command_ratio < 0.2:
                helm_score += 0.2
            if metrics.avg_words_per_utterance < 10:
                helm_score += 0.1
            if metrics.response_pattern == 'responder':
                helm_score += 0.15
            if helm_score > 0.2:
                role_scores.append((BridgeRole.HELM, min(0.5, helm_score)))

            # Tactical hints
            tactical_score = 0.0
            if metrics.avg_words_per_utterance < 8:  # Short, decisive
                tactical_score += 0.15
            if metrics.command_ratio > 0.1 and metrics.command_ratio < 0.3:
                tactical_score += 0.15
            if tactical_score > 0.2:
                role_scores.append((BridgeRole.TACTICAL, min(0.5, tactical_score)))

            # Science hints
            science_score = 0.0
            if metrics.avg_words_per_utterance > 12:  # Longer explanations
                science_score += 0.2
            if metrics.question_ratio > 0.15:
                science_score += 0.15
            if science_score > 0.2:
                role_scores.append((BridgeRole.SCIENCE, min(0.5, science_score)))

            # Engineering hints
            eng_score = 0.0
            if metrics.speaking_percentage > 5 and metrics.speaking_percentage < 20:
                eng_score += 0.1
            if metrics.response_pattern == 'responder':
                eng_score += 0.1
            if eng_score > 0.15:
                role_scores.append((BridgeRole.ENGINEERING, min(0.4, eng_score)))

            # Sort by score descending
            role_scores.sort(key=lambda x: -x[1])
            hints[speaker] = role_scores

        return hints


class EnhancedRoleInferenceEngine(RoleInferenceEngine):
    """
    Enhanced role inference using both keywords and voice patterns.

    Combines keyword-based role detection with voice pattern analysis
    for more accurate role identification from audio alone.
    """

    def __init__(
        self,
        transcripts: List[Dict[str, Any]],
        patterns: Optional[RolePatterns] = None,
        use_voice_patterns: bool = True
    ):
        """
        Initialize enhanced role inference engine.

        Args:
            transcripts: List of transcript dictionaries
            patterns: Optional custom role patterns
            use_voice_patterns: Whether to use voice pattern analysis
        """
        super().__init__(transcripts, patterns)
        self.use_voice_patterns = use_voice_patterns
        self._voice_analyzer = VoicePatternAnalyzer(transcripts) if use_voice_patterns else None

    def analyze_all_speakers(self) -> Dict[str, SpeakerRoleAnalysis]:
        """
        Analyze all speakers using keywords and voice patterns.

        Returns:
            Dictionary mapping speaker IDs to their role analysis
        """
        # Get base keyword analysis
        results = super().analyze_all_speakers()

        # Enhance with voice patterns if enabled
        if self.use_voice_patterns and self._voice_analyzer:
            voice_hints = self._voice_analyzer.get_role_hints()
            voice_metrics = self._voice_analyzer.analyze_speaker_patterns()

            for speaker, analysis in results.items():
                hints = voice_hints.get(speaker, [])
                metrics = voice_metrics.get(speaker)

                if hints and metrics:
                    # Boost confidence if voice patterns agree with keyword role
                    keyword_role = analysis.inferred_role
                    for hint_role, hint_score in hints:
                        if hint_role == keyword_role:
                            # Voice pattern agrees - boost confidence
                            boost = min(0.15, hint_score * 0.3)
                            new_confidence = min(1.0, analysis.confidence + boost)

                            # Update methodology note
                            pattern_note = self._generate_voice_pattern_note(metrics)
                            new_methodology = (
                                f"{analysis.methodology_notes} "
                                f"Voice pattern analysis (+{boost:.0%}): {pattern_note}"
                            )

                            # Create updated analysis
                            results[speaker] = SpeakerRoleAnalysis(
                                speaker=analysis.speaker,
                                inferred_role=analysis.inferred_role,
                                confidence=round(new_confidence, 2),
                                utterance_count=analysis.utterance_count,
                                utterance_percentage=analysis.utterance_percentage,
                                keyword_matches=analysis.keyword_matches,
                                total_keyword_matches=analysis.total_keyword_matches,
                                key_indicators=analysis.key_indicators,
                                example_utterances=analysis.example_utterances,
                                methodology_notes=new_methodology
                            )
                            break

                    # If keyword analysis found UNKNOWN but voice patterns suggest a role
                    if analysis.inferred_role == BridgeRole.UNKNOWN and hints:
                        top_hint = hints[0]
                        if top_hint[1] >= 0.3:  # Reasonable voice pattern confidence
                            new_methodology = (
                                f"{analysis.methodology_notes} "
                                f"Voice pattern suggests {top_hint[0].value} "
                                f"(confidence: {top_hint[1]:.0%})."
                            )
                            results[speaker] = SpeakerRoleAnalysis(
                                speaker=analysis.speaker,
                                inferred_role=top_hint[0],
                                confidence=round(top_hint[1] * 0.7, 2),  # Reduced since no keyword support
                                utterance_count=analysis.utterance_count,
                                utterance_percentage=analysis.utterance_percentage,
                                keyword_matches=analysis.keyword_matches,
                                total_keyword_matches=analysis.total_keyword_matches,
                                key_indicators=analysis.key_indicators + [f"voice:{top_hint[0].value.split('/')[0].lower()}"],
                                example_utterances=analysis.example_utterances,
                                methodology_notes=new_methodology
                            )

        return results

    def _generate_voice_pattern_note(self, metrics: VoicePatternMetrics) -> str:
        """Generate a note describing the voice patterns observed."""
        parts = []

        if metrics.is_dominant_speaker:
            parts.append("dominant speaker")

        if metrics.command_ratio > 0.2:
            parts.append(f"{metrics.command_ratio:.0%} commands")
        elif metrics.command_ratio < 0.1:
            parts.append("few commands")

        if metrics.avg_words_per_utterance > 12:
            parts.append("detailed explanations")
        elif metrics.avg_words_per_utterance < 6:
            parts.append("brief responses")

        if metrics.response_pattern == 'initiator':
            parts.append("initiates conversations")
        elif metrics.response_pattern == 'responder':
            parts.append("responds to others")

        return ", ".join(parts) if parts else "standard speaking pattern"

    def get_voice_pattern_summary(self) -> Dict[str, Any]:
        """
        Get a summary of voice patterns for all speakers.

        Returns:
            Dictionary with voice pattern analysis
        """
        if not self._voice_analyzer:
            return {}

        metrics = self._voice_analyzer.analyze_speaker_patterns()
        hints = self._voice_analyzer.get_role_hints()

        return {
            'speaker_patterns': {
                speaker: {
                    'avg_words': m.avg_words_per_utterance,
                    'speaking_pct': m.speaking_percentage,
                    'command_ratio': m.command_ratio,
                    'question_ratio': m.question_ratio,
                    'is_dominant': m.is_dominant_speaker,
                    'pattern_type': m.response_pattern,
                    'role_hints': [(r.value, s) for r, s in hints.get(speaker, [])]
                }
                for speaker, m in metrics.items()
            }
        }
