"""
Azure Application Insights telemetry integration.

Provides optional telemetry tracking for the web application.
Gracefully degrades to no-ops when Application Insights is not configured
or the azure-monitor-opentelemetry package is not installed.
"""

import logging
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
APPINSIGHTS_CONNECTION_STRING = os.getenv('APPINSIGHTS_CONNECTION_STRING', '')
APPINSIGHTS_SAMPLING_PERCENTAGE = int(os.getenv('APPINSIGHTS_SAMPLING_PERCENTAGE', '100'))
APPINSIGHTS_CLOUD_ROLE = os.getenv('APPINSIGHTS_CLOUD_ROLE', 'sh-learning-ai')

# Module-level state
_telemetry_initialized = False
_tracer = None

try:
    from azure.monitor.opentelemetry import configure_azure_monitor
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode
    AZURE_MONITOR_AVAILABLE = True
except ImportError:
    AZURE_MONITOR_AVAILABLE = False


class NoOpTelemetryClient:
    """Fallback telemetry client that does nothing.

    Used when Application Insights is not configured or the SDK
    is not installed, ensuring all telemetry calls are safe no-ops.
    """

    def track_event(
        self,
        name: str,
        properties: Optional[Dict[str, str]] = None,
        measurements: Optional[Dict[str, float]] = None
    ) -> None:
        """No-op event tracking."""

    def track_exception(
        self,
        exception: Optional[BaseException] = None,
        properties: Optional[Dict[str, str]] = None
    ) -> None:
        """No-op exception tracking."""


_noop_client = NoOpTelemetryClient()


def is_telemetry_enabled() -> bool:
    """Check whether Application Insights telemetry is enabled.

    Returns:
        True if a connection string is configured and the SDK is available.
    """
    return bool(APPINSIGHTS_CONNECTION_STRING) and AZURE_MONITOR_AVAILABLE


def initialize_telemetry(app: Any) -> None:
    """Initialize Azure Application Insights for the FastAPI application.

    Sets up OpenTelemetry with the Azure Monitor exporter and adds
    workspace tracking middleware. No-ops if the connection string
    is empty or the SDK is not installed.

    Args:
        app: The FastAPI application instance.
    """
    global _telemetry_initialized, _tracer

    if not APPINSIGHTS_CONNECTION_STRING:
        logger.info("Application Insights not configured (APPINSIGHTS_CONNECTION_STRING is empty)")
        return

    if not AZURE_MONITOR_AVAILABLE:
        logger.warning(
            "azure-monitor-opentelemetry package not installed. "
            "Install with: pip install azure-monitor-opentelemetry"
        )
        return

    try:
        configure_azure_monitor(
            connection_string=APPINSIGHTS_CONNECTION_STRING,
            enable_live_metrics=True,
            sampling_ratio=APPINSIGHTS_SAMPLING_PERCENTAGE / 100.0,
            resource_attributes={
                "service.name": APPINSIGHTS_CLOUD_ROLE,
            }
        )

        _tracer = trace.get_tracer(__name__)
        _telemetry_initialized = True

        # Add workspace tracking middleware
        from starlette.middleware.base import BaseHTTPMiddleware
        app.add_middleware(WorkspaceTrackingMiddleware)

        logger.info(
            "Application Insights initialized (sampling=%d%%)",
            APPINSIGHTS_SAMPLING_PERCENTAGE
        )
    except Exception as e:
        logger.error("Failed to initialize Application Insights: %s", e)
        _telemetry_initialized = False


class WorkspaceTrackingMiddleware:
    """ASGI middleware that adds workspace ID to OpenTelemetry spans.

    Reads the X-Workspace-ID header from each request and sets it
    as a custom attribute on the current span for cross-correlation
    in Application Insights.
    """

    def __init__(self, app: Any) -> None:
        """Initialize middleware.

        Args:
            app: The ASGI application.
        """
        self.app = app

    async def __call__(self, scope: Dict, receive: Any, send: Any) -> None:
        """Process request and add workspace ID to span.

        Args:
            scope: ASGI connection scope.
            receive: ASGI receive callable.
            send: ASGI send callable.
        """
        if scope["type"] == "http" and _telemetry_initialized:
            headers = dict(scope.get("headers", []))
            workspace_id = headers.get(b"x-workspace-id", b"").decode("utf-8", errors="ignore")
            if workspace_id:
                span = trace.get_current_span()
                if span and span.is_recording():
                    span.set_attribute("workspace.id", workspace_id)

        await self.app(scope, receive, send)


def track_event(
    name: str,
    properties: Optional[Dict[str, str]] = None,
    measurements: Optional[Dict[str, float]] = None
) -> None:
    """Track a custom event in Application Insights.

    No-ops if telemetry is not initialized.

    Args:
        name: Event name (e.g. 'analysis_start').
        properties: String key-value pairs for the event.
        measurements: Numeric measurements for the event.
    """
    if not _telemetry_initialized or not _tracer:
        return

    try:
        with _tracer.start_as_current_span(name) as span:
            if properties:
                for key, value in properties.items():
                    span.set_attribute(f"custom.{key}", str(value))
            if measurements:
                for key, value in measurements.items():
                    span.set_attribute(f"metric.{key}", float(value))
    except Exception as e:
        logger.debug("Failed to track event '%s': %s", name, e)


def track_exception(
    exception: Optional[BaseException] = None,
    properties: Optional[Dict[str, str]] = None
) -> None:
    """Track an exception in Application Insights.

    No-ops if telemetry is not initialized.

    Args:
        exception: The exception to track.
        properties: Additional context properties.
    """
    if not _telemetry_initialized or not _tracer:
        return

    try:
        span = trace.get_current_span()
        if span and span.is_recording():
            if exception:
                span.record_exception(exception)
                span.set_status(StatusCode.ERROR, str(exception))
            if properties:
                for key, value in properties.items():
                    span.set_attribute(f"custom.{key}", str(value))
    except Exception as e:
        logger.debug("Failed to track exception: %s", e)
