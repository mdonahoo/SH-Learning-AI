"""
Live GM analysis for real-time crew support during missions.

Provides lightweight LLM-based analysis of streaming transcript segments
to help the Game Master identify when crew members need help with game
controls, objectives, or communication. Designed for low-latency use
with GPU-accelerated inference.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
LIVE_ANALYSIS_INTERVAL = int(os.getenv('LIVE_ANALYSIS_INTERVAL', '30'))
LIVE_ANALYSIS_MAX_TOKENS = int(os.getenv('LIVE_ANALYSIS_MAX_TOKENS', '250'))
LIVE_ANALYSIS_WINDOW = int(os.getenv('LIVE_ANALYSIS_WINDOW', '15'))

GM_SYSTEM_PROMPT = """You are a Game Master assistant for Starship Horizons, \
a bridge simulator where a crew of 2-6 players operates a starship together. \
Each player mans a station: Captain, Helm, Tactical, Science, Engineering, \
or Communications.

Your job is to monitor crew dialogue and alert the GM when players need help. \
Be concise and actionable. Only report issues you actually observe in the \
dialogue — never speculate or invent problems."""

GM_ANALYSIS_PROMPT = """\
Analyze this recent crew dialogue from a live Starship Horizons mission.

RECENT DIALOGUE (last {window_seconds}s):
{transcript_block}

CURRENT METRICS:
- Stress: {stress_label} ({stress_pct}%)
- Effective communications: {effective_count} | Needs work: {improvement_count}
- Words per second: {wps}

Identify any issues the GM should address RIGHT NOW. Look for:
1. CONTROLS — confusion about how to operate their station
2. OBJECTIVES — unsure what to do next or where to go
3. MECHANICS — misunderstanding game rules or ship systems
4. COMMUNICATION — crew talking past each other, no one taking charge
5. FRUSTRATION — players getting stuck or upset

Respond with a short JSON object:
{{"needs_help": true/false, "urgency": "low"|"medium"|"high", \
"insights": ["one-line insight", ...], "suggestion": "what GM should do", \
"mission_summary": "1-2 sentence summary of what the crew is currently doing"}}

If the crew is doing fine, respond: {{"needs_help": false, "insights": [], \
"suggestion": "", "mission_summary": "brief summary of current activity"}}

JSON only, no markdown fences:"""


class LiveGMAnalyzer:
    """
    Analyzes streaming transcript segments for GM situational awareness.

    Maintains a cooldown timer to avoid overwhelming the LLM. Uses a
    sliding window of recent segments to keep prompts small and focused.

    Attributes:
        _client: LLMClient instance for generation.
        _last_analysis_time: Monotonic timestamp of last analysis run.
        _min_interval: Minimum seconds between analyses.
        _last_result: Most recent analysis result for deduplication.
    """

    def __init__(
        self,
        min_interval: Optional[int] = None,
        max_tokens: Optional[int] = None,
        window_size: Optional[int] = None,
    ) -> None:
        """
        Initialize the live GM analyzer.

        Args:
            min_interval: Minimum seconds between LLM calls.
            max_tokens: Maximum tokens for LLM response.
            window_size: Number of recent segments to include.
        """
        self._client: Any = None
        self._client_available: Optional[bool] = None
        self._last_analysis_time: float = 0.0
        self._min_interval = min_interval or LIVE_ANALYSIS_INTERVAL
        self._max_tokens = max_tokens or LIVE_ANALYSIS_MAX_TOKENS
        self._window_size = window_size or LIVE_ANALYSIS_WINDOW
        self._last_result: Optional[Dict[str, Any]] = None
        self._analysis_count: int = 0
        self._last_client_retry: float = 0.0
        self._client_retry_interval: float = 60.0

    def _ensure_client(self) -> bool:
        """
        Lazy-initialize the LLM client with periodic retry on failure.

        Retries every 60 seconds if the initial connection fails,
        so that starting the LLM server after the web session still works.

        Returns:
            True if client is available, False otherwise.
        """
        if self._client_available is True:
            return True

        # If previously failed, retry after cooldown
        if self._client_available is False:
            now = time.monotonic()
            if (now - self._last_client_retry) < self._client_retry_interval:
                return False
            logger.info("LiveGMAnalyzer: retrying LLM client connection...")

        try:
            from src.llm.llm_client import get_default_client
            self._client = get_default_client()
            self._client_available = True
            logger.info(
                f"LiveGMAnalyzer: LLM client ready "
                f"(model={self._client.model})"
            )
        except Exception as e:
            logger.warning(f"LiveGMAnalyzer: LLM not available: {e}")
            self._client_available = False
            self._last_client_retry = time.monotonic()

        return self._client_available

    def should_analyze(self) -> bool:
        """
        Check if enough time has elapsed for a new analysis.

        Returns:
            True if the cooldown has expired.
        """
        now = time.monotonic()
        return (now - self._last_analysis_time) >= self._min_interval

    def analyze(
        self,
        segments: List[Dict[str, Any]],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Run GM analysis on recent transcript segments.

        Takes the last N segments (window_size), builds a lightweight
        prompt, and calls the LLM synchronously. Designed to be run
        in a thread executor from async code.

        Args:
            segments: All accumulated transcript segments.
            metrics: Current live metrics dict (from LiveMetricsComputer).

        Returns:
            Dict with analysis result, or None if skipped/failed.
            Keys: needs_help, urgency, insights, suggestion,
                  generation_time, tokens_per_second.
        """
        if not self._ensure_client():
            return None

        if not self.should_analyze():
            return None

        if not segments:
            return None

        self._last_analysis_time = time.monotonic()

        # Take recent window
        window = segments[-self._window_size:]
        if not window:
            return None

        # Build transcript block
        lines = []
        for seg in window:
            t = seg.get('start', 0.0)
            mins = int(t // 60)
            secs = int(t % 60)
            text = seg.get('text', '').strip()
            if text:
                lines.append(f"  [{mins}:{secs:02d}] {text}")

        if not lines:
            return None

        transcript_block = '\n'.join(lines)

        # Extract metrics for prompt
        stress = (metrics or {}).get('stress', {})
        comm = (metrics or {}).get('communication', {})
        speech = (metrics or {}).get('speech', {})

        # Calculate window duration
        window_start = window[0].get('start', 0.0)
        window_end = window[-1].get('end', window[-1].get('start', 0.0))
        window_seconds = max(1, int(window_end - window_start))

        prompt = GM_ANALYSIS_PROMPT.format(
            window_seconds=window_seconds,
            transcript_block=transcript_block,
            stress_label=stress.get('label', 'unknown'),
            stress_pct=round(stress.get('avg', 0) * 100),
            effective_count=comm.get('effective_count', 0),
            improvement_count=comm.get('improvement_count', 0),
            wps=speech.get('avg_wps', 0.0),
        )

        try:
            start = time.monotonic()
            response = self._client.generate(
                prompt=prompt,
                system=GM_SYSTEM_PROMPT,
                temperature=0.3,
                max_tokens=self._max_tokens,
                top_p=0.9,
            )
            elapsed = time.monotonic() - start

            if not response or not response.text:
                logger.warning("LiveGMAnalyzer: empty LLM response")
                return None

            result = self._parse_response(response.text)
            result['generation_time'] = round(elapsed, 2)
            result['tokens_per_second'] = response.tokens_per_second
            result['model'] = response.model
            self._analysis_count += 1
            self._last_result = result

            logger.info(
                f"GM analysis #{self._analysis_count}: "
                f"needs_help={result.get('needs_help')}, "
                f"urgency={result.get('urgency')}, "
                f"{elapsed:.1f}s ({response.tokens_per_second} tok/s)"
            )

            return result

        except Exception as e:
            logger.error(f"LiveGMAnalyzer: analysis failed: {e}")
            return None

    def _parse_response(self, text: str) -> Dict[str, Any]:
        """
        Parse the LLM JSON response, with fallback for malformed output.

        Args:
            text: Raw LLM response text.

        Returns:
            Parsed result dict.
        """
        import json

        cleaned = text.strip()
        # Strip markdown fences if present
        if cleaned.startswith('```'):
            lines = cleaned.split('\n')
            lines = [
                ln for ln in lines
                if not ln.strip().startswith('```')
            ]
            cleaned = '\n'.join(lines).strip()

        try:
            parsed = json.loads(cleaned)
            return {
                'needs_help': bool(parsed.get('needs_help', False)),
                'urgency': str(parsed.get('urgency', 'low')),
                'insights': list(parsed.get('insights', [])),
                'suggestion': str(parsed.get('suggestion', '')),
                'mission_summary': str(
                    parsed.get('mission_summary', '')
                ),
            }
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse GM analysis JSON: {e}")
            # Fallback: treat the whole response as a single insight
            return {
                'needs_help': True,
                'urgency': 'low',
                'insights': [cleaned[:200]] if cleaned else [],
                'suggestion': '',
                'mission_summary': '',
            }

    @property
    def last_result(self) -> Optional[Dict[str, Any]]:
        """Get the most recent analysis result."""
        return self._last_result

    @property
    def analysis_count(self) -> int:
        """Get the total number of analyses performed."""
        return self._analysis_count
