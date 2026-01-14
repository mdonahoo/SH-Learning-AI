#!/usr/bin/env python3
"""
Mission Summarizer for Starship Horizons Learning AI
Generates comprehensive mission summaries from events and transcripts.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class MissionSummarizer:
    """Generates mission summaries and analyses from recorded data."""

    def __init__(self, mission_id: str, mission_name: str = "", llm_model: str = None,
                 bridge_id: str = None):
        """
        Initialize mission summarizer.

        Args:
            mission_id: Unique mission identifier
            mission_name: Human-readable mission name
            llm_model: LLM model to use for summary generation (default from env)
            bridge_id: Identifier for the recording bridge/station
        """
        import os
        self.mission_id = mission_id
        self.mission_name = mission_name
        self.bridge_id = bridge_id
        self.llm_model = llm_model or os.getenv('OLLAMA_MODEL', 'qwen2.5:14b')

        self.events = []
        self.transcripts = []
        self._timeline = []

    def load_events(self, events: List[Dict[str, Any]]) -> None:
        """Load event data for processing."""
        self.events = events.copy()

    def load_transcripts(self, transcripts: List[Dict[str, Any]]) -> None:
        """Load transcript data for processing."""
        self.transcripts = transcripts.copy()

    def generate_timeline(self) -> List[Dict[str, Any]]:
        """
        Generate chronological mission timeline.

        Returns:
            List of timeline entries with elapsed time
        """
        if not self.events:
            return []

        # Sort events by timestamp
        sorted_events = sorted(self.events, key=lambda x: x["timestamp"])

        # Get mission start time
        start_time = sorted_events[0]["timestamp"]

        timeline = []
        for event in sorted_events:
            # Calculate elapsed time
            if isinstance(event["timestamp"], str):
                event_time = datetime.fromisoformat(event["timestamp"])
            else:
                event_time = event["timestamp"]

            if isinstance(start_time, str):
                start_time_dt = datetime.fromisoformat(start_time)
            else:
                start_time_dt = start_time

            elapsed = event_time - start_time_dt

            # Format elapsed time consistently as HH:MM:SS
            total_seconds = int(elapsed.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            timeline_entry = {
                "timestamp": event_time,
                "elapsed_time": elapsed_str,
                "type": event.get("type") or event.get("event_type"),
                "data": event.get("data", {})
            }

            timeline.append(timeline_entry)

        self._timeline = timeline
        return timeline

    def identify_key_moments(self) -> List[Dict[str, Any]]:
        """
        Identify key moments in the mission.

        Returns:
            List of important events
        """
        key_event_types = ["alert", "critical", "combat", "damage", "emergency"]
        key_categories = ["combat", "damage", "system_failure"]

        key_moments = []

        for event in self.events:
            event_type = event.get("type") or event.get("event_type", "")
            category = event.get("category", "")

            # Check if this is a key event
            data = event.get("data", {})
            # Handle alert level as either string or int (4 or 5 = red alert)
            if isinstance(data, (int, str)):
                alert_level = data
            elif isinstance(data, dict):
                alert_level = data.get("alert_level")
            else:
                alert_level = None
            is_red_alert = alert_level in ["red", 4, 5] if alert_level else False

            is_key = (
                event_type in key_event_types or
                category in key_categories or
                is_red_alert or
                (isinstance(data, dict) and data.get("hull_breach")) or
                event_type == "critical"
            )

            if is_key:
                key_moments.append(event)

        return key_moments

    def analyze_crew_performance(self) -> Dict[str, Any]:
        """
        Analyze crew performance metrics.

        Returns:
            Performance metrics by crew member
        """
        performance = defaultdict(lambda: {
            "action_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "communication_count": 0,
            "success_rate": 0.0
        })

        # Analyze events
        for event in self.events:
            if event.get("type") == "crew_action" or event.get("event_type") == "crew_action":
                data = event.get("data", {})
                crew_member = data.get("crew")

                if crew_member:
                    performance[crew_member]["action_count"] += 1

                    if data.get("success"):
                        performance[crew_member]["success_count"] += 1
                    else:
                        performance[crew_member]["failure_count"] += 1

        # Analyze transcripts
        for transcript in self.transcripts:
            speaker = transcript.get("speaker")
            if speaker:
                performance[speaker]["communication_count"] += 1

        # Calculate success rates
        for crew_member, metrics in performance.items():
            if metrics["action_count"] > 0:
                metrics["success_rate"] = metrics["success_count"] / metrics["action_count"]

        return dict(performance)

    def generate_narrative_summary(self) -> Dict[str, Any]:
        """
        Generate narrative mission summary.

        Returns:
            Comprehensive mission summary
        """
        # Calculate mission duration
        if self.events:
            start_time = min(e.get("timestamp") for e in self.events)
            end_time = max(e.get("timestamp") for e in self.events)

            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time)
            if isinstance(end_time, str):
                end_time = datetime.fromisoformat(end_time)

            duration = str(end_time - start_time)
        else:
            duration = "0:00:00"

        # Find mission outcome
        outcome = "unknown"
        for event in self.events:
            if event.get("type") == "mission_complete" or event.get("event_type") == "mission_complete":
                outcome = event.get("data", {}).get("status", "completed")
                break

        # Generate narrative text
        narrative_parts = []

        # Opening
        narrative_parts.append(f"Mission {self.mission_name or self.mission_id} Summary:")

        # Mission objective
        for event in self.events:
            if event.get("type") == "mission_start" or event.get("event_type") == "mission_start":
                objective = event.get("data", {}).get("objective")
                if objective:
                    narrative_parts.append(f"Objective: {objective}")
                break

        # Key events
        key_moments = self.identify_key_moments()
        if key_moments:
            narrative_parts.append(f"The mission encountered {len(key_moments)} critical events.")

        # Combat encounters
        combat_events = [e for e in self.events if "combat" in str(e.get("type", "")).lower()]
        if combat_events:
            narrative_parts.append(f"Combat was engaged {len(combat_events)} times during the mission.")

        # Crew communications
        if self.transcripts:
            narrative_parts.append(f"The crew exchanged {len(self.transcripts)} communications.")

        # Mission outcome
        narrative_parts.append(f"Mission outcome: {outcome}")

        narrative = " ".join(narrative_parts)

        return {
            "mission_name": self.mission_name or self.mission_id,
            "duration": duration,
            "narrative": narrative,
            "key_events": len(key_moments),
            "crew_performance": self.analyze_crew_performance(),
            "outcome": outcome
        }

    def analyze_tactical_performance(self) -> Dict[str, Any]:
        """
        Analyze tactical combat performance.

        Returns:
            Tactical metrics and analysis
        """
        tactical = {
            "total_engagements": 0,
            "weapons_fired": 0,
            "hits": 0,
            "misses": 0,
            "hit_rate": 0.0,
            "weapons_used": [],
            "damage_taken": {},
            "combat_effectiveness": "unknown"
        }

        combat_started = False
        weapons_used = set()

        for event in self.events:
            event_type = event.get("type") or event.get("event_type", "")
            data = event.get("data", {})

            if event_type == "combat_start":
                combat_started = True
                tactical["total_engagements"] += 1

            elif event_type == "combat_end":
                combat_started = False

            elif event_type == "weapon_fired":
                tactical["weapons_fired"] += 1
                weapon = data.get("weapon")
                if weapon:
                    weapons_used.add(weapon)

                if data.get("hit"):
                    tactical["hits"] += 1
                else:
                    tactical["misses"] += 1

            elif event_type == "damage_taken":
                system = data.get("system")
                severity = data.get("severity")
                if system:
                    tactical["damage_taken"][system] = severity

        # Calculate hit rate
        if tactical["weapons_fired"] > 0:
            tactical["hit_rate"] = tactical["hits"] / tactical["weapons_fired"]

        tactical["weapons_used"] = list(weapons_used)

        # Determine combat effectiveness
        if tactical["hit_rate"] >= 0.8:
            tactical["combat_effectiveness"] = "high"
        elif tactical["hit_rate"] >= 0.5:
            tactical["combat_effectiveness"] = "moderate"
        elif tactical["weapons_fired"] > 0:
            tactical["combat_effectiveness"] = "low"

        return tactical

    def assess_learning_objectives(self, objectives: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess completion of learning objectives.

        Args:
            objectives: List of learning objectives to assess

        Returns:
            Assessment results
        """
        assessment = {}
        completed_count = 0

        for objective in objectives:
            obj_id = objective["id"]
            category = objective.get("category", "")

            # Check if objective was met based on events
            completed = False

            if category == "navigation":
                # Check for navigation events
                nav_events = [e for e in self.events if "navigation" in str(e.get("type", "")).lower()]
                arrival_events = [e for e in self.events if "arrival" in str(e.get("type", "")).lower()]
                completed = len(nav_events) > 0 and len(arrival_events) > 0

            elif category == "communication":
                # Check for communications
                completed = len(self.transcripts) > 0

            elif category == "combat":
                # Check for combat events
                combat_events = [e for e in self.events if "combat" in str(e.get("type", "")).lower()]
                completed = len(combat_events) > 0

            assessment[obj_id] = {
                "completed": completed,
                "description": objective.get("description", "")
            }

            if completed:
                completed_count += 1

        # Calculate completion rate
        completion_rate = completed_count / len(objectives) if objectives else 0

        assessment["completion_rate"] = completion_rate

        return assessment

    def export_report(self, filepath: Path, format: str = "json") -> None:
        """
        Export mission report in specified format.

        Args:
            filepath: Export file path
            format: Export format (json, markdown, html)
        """
        # Generate comprehensive report data
        report_data = {
            "mission_id": self.mission_id,
            "mission_name": self.mission_name,
            "bridge_id": self.bridge_id,
            "timeline": self.generate_timeline(),
            "key_moments": self.identify_key_moments(),
            "crew_performance": self.analyze_crew_performance(),
            "narrative_summary": self.generate_narrative_summary(),
            "tactical_analysis": self.analyze_tactical_performance()
        }

        filepath.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)

        elif format == "markdown":
            markdown = self._generate_markdown_report(report_data)
            with open(filepath, 'w') as f:
                f.write(markdown)

        elif format == "html":
            html = self._generate_html_report(report_data)
            with open(filepath, 'w') as f:
                f.write(html)

    def _generate_markdown_report(self, data: Dict[str, Any]) -> str:
        """Generate Markdown format report."""
        md = []
        md.append(f"# Mission Report: {data['mission_name']}")
        md.append(f"\n## Mission ID: {data['mission_id']}")
        if data.get('bridge_id'):
            md.append(f"\n## Bridge: {data['bridge_id']}")

        md.append("\n## Executive Summary")
        summary = data.get("narrative_summary", {})
        md.append(summary.get("narrative", ""))

        md.append("\n## Key Moments")
        for moment in data.get("key_moments", [])[:5]:
            md.append(f"- {moment.get('type', 'Unknown event')}")

        md.append("\n## Crew Performance")
        for crew, perf in data.get("crew_performance", {}).items():
            md.append(f"- **{crew}**: {perf['action_count']} actions, "
                     f"{perf['success_rate']:.0%} success rate")

        return "\n".join(md)

    def _generate_html_report(self, data: Dict[str, Any]) -> str:
        """Generate HTML format report."""
        html = []
        html.append("<!DOCTYPE html><html><head>")
        html.append("<title>Mission Report</title>")
        html.append("<style>body { font-family: Arial; margin: 20px; }</style>")
        html.append("</head><body>")

        html.append(f"<h1>Mission Report: {data['mission_name']}</h1>")
        html.append(f"<h2>Mission ID: {data['mission_id']}</h2>")

        html.append("<h2>Executive Summary</h2>")
        summary = data.get("narrative_summary", {})
        html.append(f"<p>{summary.get('narrative', '')}</p>")

        html.append("<h2>Crew Performance</h2><ul>")
        for crew, perf in data.get("crew_performance", {}).items():
            html.append(f"<li><strong>{crew}</strong>: {perf['action_count']} actions, "
                       f"{perf['success_rate']:.0%} success rate</li>")
        html.append("</ul>")

        html.append("</body></html>")
        return "".join(html)

    @staticmethod
    def compare_missions(missions: List['MissionSummarizer']) -> Dict[str, Any]:
        """
        Compare multiple mission performances.

        Args:
            missions: List of mission summarizers to compare

        Returns:
            Comparison analysis
        """
        comparison = {
            "missions": [],
            "best_score": 0,
            "worst_score": 100,
            "average_score": 0
        }

        scores = []

        for mission in missions:
            # Extract score from events
            score = 0
            for event in mission.events:
                if event.get("type") == "mission_complete" or event.get("event_type") == "mission_complete":
                    score = event.get("data", {}).get("score", 0)
                    break

            comparison["missions"].append({
                "mission_id": mission.mission_id,
                "score": score
            })

            scores.append(score)

        if scores:
            comparison["best_score"] = max(scores)
            comparison["worst_score"] = min(scores)
            comparison["average_score"] = sum(scores) / len(scores)

        return comparison

    def generate_llm_summary(self, include_recommendations: bool = False,
                             include_learning_points: bool = False,
                             style: str = "professional") -> Dict[str, Any]:
        """
        Generate LLM-powered mission summary using Ollama.

        Args:
            include_recommendations: Include performance recommendations
            include_learning_points: Include learning points
            style: Summary style (entertaining, professional, technical, casual)

        Returns:
            LLM-generated summary
        """
        try:
            from src.llm.ollama_client import OllamaClient

            # Check if LLM is enabled
            import os
            if os.getenv('ENABLE_LLM_REPORTS', 'true').lower() != 'true':
                logger.info("LLM reports disabled, using fallback")
                return self._generate_fallback_summary(include_recommendations, include_learning_points)

            # Initialize Ollama client
            client = OllamaClient(model=self.llm_model)

            # Check connection
            if not client.check_connection():
                logger.warning("Ollama server not available, using fallback")
                return self._generate_fallback_summary(include_recommendations, include_learning_points)

            # Prepare mission data
            mission_data = {
                'mission_id': self.mission_id,
                'mission_name': self.mission_name,
                'events': self.events,
                'transcripts': self.transcripts,
                'duration': self._calculate_duration()
            }

            # Generate summary
            logger.info(f"Generating LLM summary (style={style})...")
            summary_text = client.generate_mission_summary(mission_data, style=style)

            if not summary_text:
                logger.warning("LLM generation failed, using fallback")
                return self._generate_fallback_summary(include_recommendations, include_learning_points)

            result = {
                "summary": summary_text.strip(),
                "generated_by": f"ollama/{client.model}"
            }

            # Generate additional sections if requested
            if include_recommendations or include_learning_points:
                crew_analysis = client.generate_crew_analysis(self.transcripts, self.events)
                if crew_analysis:
                    result["crew_analysis"] = crew_analysis

            logger.info("✓ LLM summary generated successfully")
            return result

        except ImportError:
            logger.error("Ollama client not available")
            return self._generate_fallback_summary(include_recommendations, include_learning_points)
        except Exception as e:
            logger.error(f"Error generating LLM summary: {e}")
            return self._generate_fallback_summary(include_recommendations, include_learning_points)

    def _generate_fallback_summary(self, include_recommendations: bool = False,
                                   include_learning_points: bool = False) -> Dict[str, Any]:
        """
        Generate fallback summary when LLM is unavailable.

        Args:
            include_recommendations: Include performance recommendations
            include_learning_points: Include learning points

        Returns:
            Basic summary
        """
        summary_text = f"""
        Mission {self.mission_name} was successfully completed. The crew demonstrated
        effective teamwork and tactical proficiency throughout the mission. Key events
        were handled with appropriate responses, and all primary objectives were achieved.
        The mission lasted {len(self.events)} recorded events with {len(self.transcripts)}
        crew communications.
        """

        result = {
            "summary": summary_text.strip(),
            "generated_by": "fallback"
        }

        if include_recommendations:
            result["recommendations"] = [
                "Continue practicing emergency response procedures",
                "Improve communication clarity during high-stress situations",
                "Review tactical engagement protocols"
            ]

        if include_learning_points:
            result["learning_points"] = [
                "Effective use of defensive patterns during combat",
                "Quick decision-making under pressure",
                "Successful crew coordination"
            ]

        return result

    def _calculate_duration(self) -> str:
        """Calculate mission duration from events."""
        if not self.events:
            return "0:00:00"

        start_time = min(e.get("timestamp") for e in self.events)
        end_time = max(e.get("timestamp") for e in self.events)

        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time)

        return str(end_time - start_time)

    def generate_llm_report(self, style: str = "entertaining", output_file: Optional[Path] = None) -> str:
        """
        Generate complete mission report using LLM.

        Args:
            style: Report style (entertaining, professional, technical, casual)
            output_file: Optional file path to save the report

        Returns:
            Markdown formatted report
        """
        try:
            from src.llm.ollama_client import OllamaClient
            import os

            # Check if LLM is enabled
            if os.getenv('ENABLE_LLM_REPORTS', 'true').lower() != 'true':
                logger.info("LLM reports disabled")
                return ""

            # Initialize Ollama client
            client = OllamaClient(model=self.llm_model)

            if not client.check_connection():
                logger.warning("Ollama server not available")
                return ""

            # Prepare complete mission data
            mission_data = {
                'mission_id': self.mission_id,
                'mission_name': self.mission_name,
                'events': self.events,
                'transcripts': self.transcripts,
                'duration': self._calculate_duration()
            }

            # Generate full report
            logger.info(f"Generating full LLM report (style={style})...")
            report_markdown = client.generate_full_report(mission_data, style=style)

            if report_markdown:
                logger.info("✓ Full report generated successfully")

                # Save to file if specified
                if output_file:
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_file, 'w') as f:
                        f.write(report_markdown)
                    logger.info(f"✓ Report saved to {output_file}")

                return report_markdown
            else:
                logger.warning("Report generation returned empty")
                return ""

        except Exception as e:
            logger.error(f"Error generating LLM report: {e}")
            return ""

    def generate_hybrid_report(self, style: str = "professional", output_file: Optional[Path] = None) -> str:
        """
        Generate mission report using hybrid approach:
        - Python calculates ALL facts and metrics
        - LLM only formats pre-computed data into narrative

        This prevents hallucination by removing LLM's ability to invent facts.

        Args:
            style: Report style (professional, technical, educational)
            output_file: Optional file path to save the report

        Returns:
            Markdown formatted report
        """
        try:
            from src.llm.ollama_client import OllamaClient
            from src.metrics.learning_evaluator import LearningEvaluator
            import os

            # Check if LLM is enabled
            if os.getenv('ENABLE_LLM_REPORTS', 'true').lower() != 'true':
                logger.info("LLM reports disabled")
                return ""

            # Step 1: Calculate all facts programmatically
            logger.info("Step 1: Computing facts and metrics...")
            evaluator = LearningEvaluator(self.events, self.transcripts)
            structured_data = evaluator.generate_structured_report()

            # Add mission metadata
            structured_data['mission_id'] = self.mission_id
            structured_data['mission_name'] = self.mission_name

            logger.info(f"✓ Computed {len(structured_data['evaluations'])} evaluation frameworks")
            logger.info(f"  - Kirkpatrick: 4 levels")
            logger.info(f"  - Bloom's Taxonomy: {structured_data['evaluations']['blooms_taxonomy']['total_cognitive_indicators']} cognitive indicators")
            logger.info(f"  - NASA Teamwork: {structured_data['evaluations']['nasa_teamwork']['overall_teamwork_score']}/100")

            # Step 2: Use LLM only for formatting
            logger.info("Step 2: Formatting with LLM (facts locked, narrative only)...")
            client = OllamaClient(model=self.llm_model)

            if not client.check_connection():
                logger.warning("Ollama server not available")
                return ""

            report_markdown = client.generate_hybrid_report(structured_data, style=style)

            if report_markdown:
                logger.info("✓ Hybrid report generated successfully")

                # Validate the generated report against known facts
                from src.metrics.report_validator import ReportValidator
                validator = ReportValidator(structured_data)
                issues = validator.validate_report(report_markdown)

                summary = validator.get_summary()
                if summary['errors'] > 0:
                    logger.warning(f"⚠️ Report validation found {summary['errors']} errors, {summary['warnings']} warnings")
                    logger.warning("The LLM may have invented or miscounted data")
                    # Log the validation report for debugging
                    for issue in issues:
                        if issue.severity.value == 'error':
                            logger.warning(f"  - {issue.category}: expected {issue.expected}, found {issue.found}")
                elif summary['warnings'] > 0:
                    logger.info(f"Report validation: {summary['warnings']} warnings (may be acceptable)")
                else:
                    logger.info("✓ Report validation passed - all facts verified")

                # Save to file if specified
                if output_file:
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_file, 'w') as f:
                        f.write(report_markdown)
                    logger.info(f"✓ Report saved to {output_file}")

                return report_markdown
            else:
                logger.warning("Report generation returned empty")
                return ""

        except Exception as e:
            logger.error(f"Error generating hybrid report: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""

    def generate_mission_story(self, output_file: Optional[Path] = None) -> str:
        """
        Generate creative mission story based on actual events and dialogue.

        Uses hybrid approach:
        - Python extracts actual dialogue and facts
        - LLM creates narrative around those facts

        Args:
            output_file: Optional file path to save the story

        Returns:
            Mission story
        """
        try:
            from src.llm.ollama_client import OllamaClient
            from src.metrics.learning_evaluator import LearningEvaluator
            import os

            # Check if LLM is enabled
            if os.getenv('ENABLE_LLM_REPORTS', 'true').lower() != 'true':
                logger.info("LLM reports disabled")
                return ""

            # Get structured data with actual dialogue
            logger.info("Extracting mission facts and dialogue...")
            evaluator = LearningEvaluator(self.events, self.transcripts)
            structured_data = evaluator.generate_structured_report()

            # Add mission metadata
            structured_data['mission_id'] = self.mission_id
            structured_data['mission_name'] = self.mission_name

            logger.info(f"✓ Extracted {len(self.transcripts)} actual dialogue lines")

            # Generate story
            client = OllamaClient(model=self.llm_model)

            if not client.check_connection():
                logger.warning("Ollama server not available")
                return ""

            logger.info("Generating creative story (this may take 2-4 minutes)...")
            story = client.generate_mission_story(structured_data)

            if story:
                logger.info("✓ Mission story generated successfully")

                # Save to file if specified
                if output_file:
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_file, 'w') as f:
                        f.write(f"# Mission Story: {self.mission_name}\n\n")
                        f.write(f"*Based on actual mission {self.mission_id}*\n\n")
                        f.write("---\n\n")
                        f.write(story)
                    logger.info(f"✓ Story saved to {output_file}")

                return story
            else:
                logger.warning("Story generation returned empty")
                return ""

        except Exception as e:
            logger.error(f"Error generating mission story: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""