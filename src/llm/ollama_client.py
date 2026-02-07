"""
Ollama client for LLM-powered mission analysis.

This module provides integration with a local Ollama server for generating
mission summaries, crew performance analysis, and narrative reports.
"""

import logging
import os
import sys
import time
import threading
from typing import Dict, Any, List, Optional, Callable, Tuple
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Import hallucination prevention utilities
try:
    from src.llm.hallucination_prevention import (
        ConstrainedContextBuilder,
        OutputValidator,
        clean_hallucinations,
        ANTI_HALLUCINATION_PARAMS,
        STORY_PARAMS,
    )
    HALLUCINATION_PREVENTION_AVAILABLE = True
except ImportError:
    HALLUCINATION_PREVENTION_AVAILABLE = False
    logger.warning("Hallucination prevention module not available")


class ProgressDisplay:
    """Real-time progress display for LLM generation."""

    # Time estimation based on empirical measurements
    # Baseline: 4-min session (55KB) = 220 seconds
    SECONDS_PER_KB = 4.0  # ~4 seconds per KB of transcript

    def __init__(
        self,
        transcript_size_kb: float = 0,
        show_spinner: bool = True,
        output_stream=None
    ):
        """
        Initialize progress display.

        Args:
            transcript_size_kb: Size of transcript in KB for time estimation
            show_spinner: Whether to show spinner animation
            output_stream: Output stream (default: sys.stderr)
        """
        self.transcript_size_kb = transcript_size_kb
        self.show_spinner = show_spinner
        self.output = output_stream or sys.stderr
        self.start_time = None
        self.chars_generated = 0
        self.tokens_generated = 0
        self._stop_event = threading.Event()
        self._spinner_thread = None
        self._spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self._spinner_idx = 0

        # Estimate time based on transcript size
        if transcript_size_kb > 0:
            self.estimated_seconds = transcript_size_kb * self.SECONDS_PER_KB
        else:
            self.estimated_seconds = 60  # Default 1 minute

    def start(self) -> None:
        """Start progress display."""
        self.start_time = time.time()
        self.chars_generated = 0
        self.tokens_generated = 0
        self._stop_event.clear()

        if self.show_spinner:
            self._spinner_thread = threading.Thread(target=self._run_spinner, daemon=True)
            self._spinner_thread.start()

    def _run_spinner(self) -> None:
        """Run spinner animation in background thread."""
        while not self._stop_event.is_set():
            elapsed = time.time() - self.start_time
            remaining = max(0, self.estimated_seconds - elapsed)

            # Build status line
            spinner = self._spinner_chars[self._spinner_idx % len(self._spinner_chars)]
            self._spinner_idx += 1

            if self.chars_generated > 0:
                status = (
                    f"\r{spinner} Generating report... "
                    f"[{self._format_time(elapsed)} elapsed, "
                    f"~{self._format_time(remaining)} remaining] "
                    f"({self.chars_generated:,} chars)"
                )
            else:
                status = (
                    f"\r{spinner} Generating report... "
                    f"[{self._format_time(elapsed)} elapsed, "
                    f"~{self._format_time(remaining)} remaining]"
                )

            self.output.write(f"{status}    ")
            self.output.flush()

            self._stop_event.wait(0.1)

    def update(self, chars: int = 0, tokens: int = 0) -> None:
        """Update progress with new characters/tokens."""
        self.chars_generated += chars
        self.tokens_generated += tokens

    def stop(self, success: bool = True) -> None:
        """Stop progress display."""
        self._stop_event.set()
        if self._spinner_thread:
            self._spinner_thread.join(timeout=0.5)

        elapsed = time.time() - self.start_time if self.start_time else 0

        # Clear line and show final status
        if success:
            self.output.write(
                f"\r✓ Report generated in {self._format_time(elapsed)} "
                f"({self.chars_generated:,} chars)          \n"
            )
        else:
            self.output.write(
                f"\r✗ Generation failed after {self._format_time(elapsed)}          \n"
            )
        self.output.flush()

    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS or HH:MM:SS."""
        seconds = int(seconds)
        if seconds >= 3600:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}:{secs:02d}"


class OllamaClient:
    """Client for interacting with Ollama LLM server."""

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None
    ):
        """
        Initialize Ollama client.

        Args:
            host: Ollama server URL (default from OLLAMA_HOST env var)
            model: Model name to use (default from OLLAMA_MODEL env var)
            timeout: Request timeout in seconds (default from OLLAMA_TIMEOUT env var)
        """
        self.host = host or os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.model = model or os.getenv('OLLAMA_MODEL', 'llama3.2')
        self.timeout = timeout or int(os.getenv('OLLAMA_TIMEOUT', '120'))
        self.num_ctx = int(os.getenv('OLLAMA_NUM_CTX', '32768'))

        # Ensure host doesn't end with slash
        self.host = self.host.rstrip('/')

        logger.info(f"Ollama client initialized: {self.host}, model={self.model}")

    def check_connection(self) -> bool:
        """
        Check if Ollama server is accessible.

        Returns:
            True if server is accessible, False otherwise
        """
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False

    def list_models(self) -> List[str]:
        """
        List available models on Ollama server.

        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        metrics_out: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Generate text using Ollama.

        Args:
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-1.0), lower = more deterministic
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling threshold (0.0-1.0)
            top_k: Top-k sampling limit
            repeat_penalty: Penalty for repeating tokens (1.0 = no penalty)
            metrics_out: Optional dict to populate with Ollama response metrics
                (prompt_eval_count, eval_count, total_duration, etc.)

        Returns:
            Generated text or None if generation failed
        """
        try:
            options = {
                "temperature": temperature,
                "num_ctx": self.num_ctx,
            }

            # Add optional sampling parameters for hallucination reduction
            if top_p is not None:
                options["top_p"] = top_p
            if top_k is not None:
                options["top_k"] = top_k
            if repeat_penalty is not None:
                options["repeat_penalty"] = repeat_penalty
            if max_tokens:
                options["num_predict"] = max_tokens

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": options
            }

            if system:
                payload["system"] = system

            logger.info(f"Sending request to Ollama (model={self.model}, temp={temperature})")

            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            generated_text = result.get('response', '')

            # Populate caller-provided metrics dict with Ollama response stats
            if metrics_out is not None:
                metrics_out['model'] = self.model
                for key in (
                    'prompt_eval_count', 'eval_count', 'total_duration',
                    'load_duration', 'prompt_eval_duration', 'eval_duration'
                ):
                    metrics_out[key] = result.get(key, 0)

            logger.info(f"✓ Generated {len(generated_text)} characters")
            return generated_text

        except requests.exceptions.Timeout:
            logger.error(f"Ollama request timed out after {self.timeout}s")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during generation: {e}")
            return None

    def generate_streaming(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        callback: Optional[Callable[[str], None]] = None
    ) -> Optional[str]:
        """
        Generate text with streaming response.

        Args:
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature
            callback: Function to call with each chunk (optional)

        Returns:
            Complete generated text or None if failed
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_ctx": self.num_ctx,
                }
            }

            if system:
                payload["system"] = system

            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()

            full_response = []

            for line in response.iter_lines():
                if line:
                    import json
                    chunk_data = json.loads(line)
                    chunk_text = chunk_data.get('response', '')

                    if chunk_text:
                        full_response.append(chunk_text)
                        if callback:
                            callback(chunk_text)

            complete_text = ''.join(full_response)
            logger.info(f"✓ Streaming complete: {len(complete_text)} characters")
            return complete_text

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            return None

    def generate_with_progress(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        transcript_size_kb: float = 0,
        show_progress: bool = True
    ) -> Optional[str]:
        """
        Generate text with progress display.

        Args:
            prompt: User prompt
            system: System prompt (optional)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            transcript_size_kb: Size of transcript in KB for time estimation
            show_progress: Whether to show progress indicator

        Returns:
            Generated text or None if generation failed
        """
        progress = None
        if show_progress:
            progress = ProgressDisplay(
                transcript_size_kb=transcript_size_kb,
                show_spinner=True
            )
            progress.start()

        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_ctx": self.num_ctx,
                }
            }

            if system:
                payload["system"] = system

            if max_tokens:
                payload["options"]["num_predict"] = max_tokens

            logger.info(f"Sending request to Ollama (model={self.model}, temp={temperature})")

            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()

            full_response = []

            for line in response.iter_lines():
                if line:
                    import json
                    chunk_data = json.loads(line)
                    chunk_text = chunk_data.get('response', '')

                    if chunk_text:
                        full_response.append(chunk_text)
                        if progress:
                            progress.update(chars=len(chunk_text))

            complete_text = ''.join(full_response)
            logger.info(f"✓ Generated {len(complete_text)} characters")

            if progress:
                progress.stop(success=True)

            return complete_text

        except requests.exceptions.Timeout:
            logger.error(f"Ollama request timed out after {self.timeout}s")
            if progress:
                progress.stop(success=False)
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            if progress:
                progress.stop(success=False)
            return None
        except Exception as e:
            logger.error(f"Unexpected error during generation: {e}")
            if progress:
                progress.stop(success=False)
            return None

    def generate_mission_summary(
        self,
        mission_data: Dict[str, Any],
        style: str = "entertaining"
    ) -> Optional[str]:
        """
        Generate mission summary from recorded data.

        Args:
            mission_data: Dictionary containing mission events and transcripts
            style: Summary style (entertaining, professional, technical)

        Returns:
            Generated summary or None if failed
        """
        from src.llm.prompt_templates import build_mission_summary_prompt

        prompt = build_mission_summary_prompt(mission_data, style)
        system_prompt = "You are an expert mission analyst for bridge simulator training sessions."

        return self.generate(prompt, system=system_prompt, temperature=0.8)

    def generate_crew_analysis(
        self,
        transcripts: List[Dict[str, Any]],
        events: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Generate crew performance analysis.

        Args:
            transcripts: List of crew communications
            events: List of mission events

        Returns:
            Crew analysis or None if failed
        """
        from src.llm.prompt_templates import build_crew_analysis_prompt

        prompt = build_crew_analysis_prompt(transcripts, events)
        system_prompt = "You are an expert in crew coordination and team performance analysis."

        return self.generate(prompt, system=system_prompt, temperature=0.6)

    def generate_full_report(
        self,
        mission_data: Dict[str, Any],
        style: str = "entertaining",
        show_progress: bool = True
    ) -> Optional[str]:
        """
        Generate complete mission report in markdown format.

        Args:
            mission_data: Complete mission data including events and transcripts
            style: Report style
            show_progress: Whether to show progress indicator

        Returns:
            Markdown formatted report or None if failed
        """
        from src.llm.prompt_templates import build_full_report_prompt

        prompt = build_full_report_prompt(mission_data, style)
        system_prompt = """You are an expert mission analyst who creates comprehensive,
        entertaining, and insightful mission reports for bridge simulator training sessions."""

        # Calculate transcript size for time estimation
        import json
        transcript_size_kb = len(json.dumps(mission_data.get('transcripts', []))) / 1024

        return self.generate_with_progress(
            prompt,
            system=system_prompt,
            temperature=0.7,
            max_tokens=4096,
            transcript_size_kb=transcript_size_kb,
            show_progress=show_progress
        )

    def generate_hybrid_report(
        self,
        structured_data: Dict[str, Any],
        style: str = "professional"
    ) -> Optional[str]:
        """
        Generate mission report from pre-computed structured data.

        This hybrid approach calculates all facts programmatically,
        then uses LLM ONLY for narrative formatting.

        Args:
            structured_data: Pre-computed analysis from LearningEvaluator
            style: Narrative style (professional, technical, educational)

        Returns:
            Markdown formatted report or None if failed
        """
        from src.llm.hybrid_prompts import build_hybrid_narrative_prompt

        prompt = build_hybrid_narrative_prompt(structured_data, style)

        system_prompt = """You are a training assessment formatter. You format pre-computed
        data into readable narratives. You NEVER calculate, modify, or invent data.
        You ONLY format provided facts into clear, professional prose."""

        logger.info(f"Generating hybrid report with pre-computed facts (style={style})")

        return self.generate(
            prompt,
            system=system_prompt,
            temperature=0.3,  # Lower temperature for factual formatting
            max_tokens=4096
        )

    def generate_mission_story(
        self,
        structured_data: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate creative mission story from pre-computed facts.

        Uses hybrid approach: Python provides facts and actual dialogue,
        LLM creates narrative around those facts.

        Args:
            structured_data: Pre-computed analysis from LearningEvaluator

        Returns:
            Short story or None if failed
        """
        from src.llm.story_prompts import build_mission_story_prompt

        prompt = build_mission_story_prompt(structured_data)

        system_prompt = """You are a creative writer specializing in military sci-fi
        and bridge simulation narratives. You craft engaging stories that bring
        mission data to life while staying 100% faithful to actual events and dialogue."""

        logger.info("Generating mission story from actual dialogue and events...")

        # Use story-specific parameters if available
        if HALLUCINATION_PREVENTION_AVAILABLE:
            return self.generate(
                prompt,
                system=system_prompt,
                temperature=STORY_PARAMS["temperature"],
                max_tokens=STORY_PARAMS["num_predict"],
                top_p=STORY_PARAMS["top_p"],
                top_k=STORY_PARAMS["top_k"],
                repeat_penalty=STORY_PARAMS["repeat_penalty"]
            )
        else:
            return self.generate(
                prompt,
                system=system_prompt,
                temperature=0.7,
                max_tokens=4096
            )

    def generate_constrained_debrief(
        self,
        transcripts: List[Dict[str, Any]],
        analysis_results: Dict[str, Any],
        validate_output: bool = True
    ) -> Tuple[Optional[str], List[Any]]:
        """
        Generate mission debrief with hallucination prevention.

        Uses constrained context (only verified data) and validates output.

        Args:
            transcripts: Raw transcript data
            analysis_results: Pre-computed analysis from other modules
            validate_output: Whether to validate generated content

        Returns:
            Tuple of (generated debrief, list of validation issues)
        """
        if not HALLUCINATION_PREVENTION_AVAILABLE:
            logger.warning("Hallucination prevention not available, using standard generation")
            # Fall back to standard generation
            from src.llm.hybrid_prompts import build_concise_debrief_prompt
            prompt = build_concise_debrief_prompt({'seven_habits': analysis_results.get('seven_habits', {}),
                                                   'communication_quality': analysis_results.get('communication_quality', {}),
                                                   'top_communications': []})
            result = self.generate(prompt, temperature=0.3, max_tokens=500)
            return result, []

        # Build constrained context
        context_builder = ConstrainedContextBuilder(transcripts, analysis_results)
        context = context_builder.build_debrief_context()

        # Build prompt with constrained data
        from src.llm.hybrid_prompts import MISSION_DEBRIEF_PROMPT

        # Format key moments for the prompt
        key_moments_text = "\n".join([
            f"[{m['timestamp']}] {m['speaker']}: \"{m['text']}\""
            for m in context['key_moments']
        ])

        prompt = MISSION_DEBRIEF_PROMPT.format(
            habits_score=context['habits_score'],
            top_habit=context['top_habit']['name'],
            top_score=context['top_habit']['score'],
            lowest_habit=context['lowest_habit']['name'],
            lowest_score=context['lowest_habit']['score'],
            comm_score=context['communication_score'],
            key_moments=key_moments_text
        )

        system_prompt = """You are a flight instructor reviewing a training mission.
        Use ONLY the data provided. Do not invent quotes, statistics, or events.
        Keep your response under 300 words. Write for middle school students (ages 11-14)."""

        logger.info("Generating constrained debrief with anti-hallucination parameters...")

        # Generate with anti-hallucination parameters
        generated = self.generate(
            prompt,
            system=system_prompt,
            temperature=ANTI_HALLUCINATION_PARAMS["temperature"],
            max_tokens=ANTI_HALLUCINATION_PARAMS["num_predict"],
            top_p=ANTI_HALLUCINATION_PARAMS["top_p"],
            top_k=ANTI_HALLUCINATION_PARAMS["top_k"],
            repeat_penalty=ANTI_HALLUCINATION_PARAMS["repeat_penalty"]
        )

        if not generated:
            return None, []

        # Validate output if requested
        issues = []
        if validate_output:
            generated, issues = clean_hallucinations(
                generated,
                transcripts,
                analysis_results,
                add_warning=True
            )

            if issues:
                logger.warning(f"Validation found {len(issues)} issues in generated debrief")

        return generated, issues

    def generate_validated_story(
        self,
        structured_data: Dict[str, Any],
        validate_output: bool = True
    ) -> Tuple[Optional[str], List[Any]]:
        """
        Generate mission story with post-generation validation.

        Args:
            structured_data: Pre-computed analysis from LearningEvaluator
            validate_output: Whether to validate generated content

        Returns:
            Tuple of (generated story, list of validation issues)
        """
        from src.llm.story_prompts import build_mission_story_prompt

        prompt = build_mission_story_prompt(structured_data)

        system_prompt = """You are a creative writer specializing in military sci-fi
        and bridge simulation narratives. You craft engaging stories that bring
        mission data to life while staying 100% faithful to actual events and dialogue.
        Use ONLY the dialogue provided - do not invent new quotes."""

        logger.info("Generating validated mission story...")

        # Use story parameters
        if HALLUCINATION_PREVENTION_AVAILABLE:
            generated = self.generate(
                prompt,
                system=system_prompt,
                temperature=STORY_PARAMS["temperature"],
                max_tokens=STORY_PARAMS["num_predict"],
                top_p=STORY_PARAMS["top_p"],
                top_k=STORY_PARAMS["top_k"],
                repeat_penalty=STORY_PARAMS["repeat_penalty"]
            )
        else:
            generated = self.generate(
                prompt,
                system=system_prompt,
                temperature=0.5,
                max_tokens=2500
            )

        if not generated:
            return None, []

        # Validate output if requested
        issues = []
        if validate_output and HALLUCINATION_PREVENTION_AVAILABLE:
            raw_transcripts = structured_data.get('raw_transcripts', [])
            generated, issues = clean_hallucinations(
                generated,
                raw_transcripts,
                structured_data,
                add_warning=True
            )

            if issues:
                logger.warning(f"Validation found {len(issues)} issues in generated story")

        return generated, issues
