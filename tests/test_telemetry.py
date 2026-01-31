"""Tests for Azure Application Insights telemetry module."""

import pytest
from unittest.mock import patch, MagicMock

import src.web.telemetry as telemetry_module
from src.web.telemetry import (
    NoOpTelemetryClient,
    is_telemetry_enabled,
    track_event,
    track_exception,
    initialize_telemetry,
)


class TestNoOpTelemetryClient:
    """Test suite for NoOpTelemetryClient."""

    @pytest.fixture
    def client(self) -> NoOpTelemetryClient:
        """Create a NoOpTelemetryClient instance."""
        return NoOpTelemetryClient()

    def test_track_event_no_args(self, client: NoOpTelemetryClient) -> None:
        """track_event with no optional args does not raise."""
        client.track_event("test_event")

    def test_track_event_with_properties(self, client: NoOpTelemetryClient) -> None:
        """track_event with properties does not raise."""
        client.track_event("test_event", properties={"key": "value"})

    def test_track_event_with_measurements(self, client: NoOpTelemetryClient) -> None:
        """track_event with measurements does not raise."""
        client.track_event("test_event", measurements={"duration": 1.5})

    def test_track_event_with_all(self, client: NoOpTelemetryClient) -> None:
        """track_event with both properties and measurements does not raise."""
        client.track_event(
            "test_event",
            properties={"key": "value"},
            measurements={"duration": 1.5}
        )

    def test_track_exception_no_args(self, client: NoOpTelemetryClient) -> None:
        """track_exception with no args does not raise."""
        client.track_exception()

    def test_track_exception_with_exception(self, client: NoOpTelemetryClient) -> None:
        """track_exception with an exception does not raise."""
        client.track_exception(exception=ValueError("test"))

    def test_track_exception_with_properties(self, client: NoOpTelemetryClient) -> None:
        """track_exception with properties does not raise."""
        client.track_exception(
            exception=RuntimeError("test"),
            properties={"context": "unit_test"}
        )


class TestIsTelemetryEnabled:
    """Test suite for is_telemetry_enabled."""

    @patch('src.web.telemetry.APPINSIGHTS_CONNECTION_STRING', '')
    @patch('src.web.telemetry.AZURE_MONITOR_AVAILABLE', True)
    def test_disabled_when_no_connection_string(self) -> None:
        """Returns False when connection string is empty."""
        assert is_telemetry_enabled() is False

    @patch('src.web.telemetry.APPINSIGHTS_CONNECTION_STRING', 'InstrumentationKey=test')
    @patch('src.web.telemetry.AZURE_MONITOR_AVAILABLE', False)
    def test_disabled_when_sdk_not_available(self) -> None:
        """Returns False when azure-monitor-opentelemetry is not installed."""
        assert is_telemetry_enabled() is False

    @patch('src.web.telemetry.APPINSIGHTS_CONNECTION_STRING', '')
    @patch('src.web.telemetry.AZURE_MONITOR_AVAILABLE', False)
    def test_disabled_when_both_missing(self) -> None:
        """Returns False when both connection string and SDK are missing."""
        assert is_telemetry_enabled() is False

    @patch('src.web.telemetry.APPINSIGHTS_CONNECTION_STRING', 'InstrumentationKey=test')
    @patch('src.web.telemetry.AZURE_MONITOR_AVAILABLE', True)
    def test_enabled_when_configured(self) -> None:
        """Returns True when connection string is set and SDK is available."""
        assert is_telemetry_enabled() is True


class TestTrackEventBeforeInit:
    """Test that track_event does not raise before initialization."""

    @patch('src.web.telemetry._telemetry_initialized', False)
    def test_track_event_before_init(self) -> None:
        """track_event does not raise when telemetry is not initialized."""
        track_event("test_event", properties={"key": "value"})

    @patch('src.web.telemetry._telemetry_initialized', False)
    def test_track_exception_before_init(self) -> None:
        """track_exception does not raise when telemetry is not initialized."""
        track_exception(exception=ValueError("test"), properties={"key": "value"})


class TestInitializeTelemetry:
    """Test suite for initialize_telemetry."""

    @patch('src.web.telemetry.APPINSIGHTS_CONNECTION_STRING', '')
    def test_noop_when_no_connection_string(self) -> None:
        """initialize_telemetry does nothing when connection string is empty."""
        mock_app = MagicMock()
        initialize_telemetry(mock_app)
        mock_app.add_middleware.assert_not_called()

    @patch('src.web.telemetry.APPINSIGHTS_CONNECTION_STRING', 'InstrumentationKey=test')
    @patch('src.web.telemetry.AZURE_MONITOR_AVAILABLE', False)
    def test_noop_when_sdk_not_available(self) -> None:
        """initialize_telemetry does nothing when SDK is not installed."""
        mock_app = MagicMock()
        initialize_telemetry(mock_app)
        mock_app.add_middleware.assert_not_called()

    @patch('src.web.telemetry.APPINSIGHTS_CONNECTION_STRING', 'InstrumentationKey=test')
    @patch('src.web.telemetry.AZURE_MONITOR_AVAILABLE', True)
    def test_initializes_with_valid_config(self) -> None:
        """initialize_telemetry configures Azure Monitor when properly configured."""
        mock_configure = MagicMock()
        mock_trace = MagicMock()
        mock_trace.get_tracer.return_value = MagicMock()

        # Inject mocks for SDK symbols that may not exist when package is missing
        telemetry_module.configure_azure_monitor = mock_configure
        telemetry_module.trace = mock_trace

        try:
            mock_app = MagicMock()
            initialize_telemetry(mock_app)

            mock_configure.assert_called_once()
            mock_app.add_middleware.assert_called_once()
        finally:
            # Reset module-level init state
            telemetry_module._telemetry_initialized = False
            telemetry_module._tracer = None
            if not hasattr(telemetry_module, '_original_configure'):
                delattr(telemetry_module, 'configure_azure_monitor')

    @patch('src.web.telemetry.APPINSIGHTS_CONNECTION_STRING', 'InstrumentationKey=test')
    @patch('src.web.telemetry.AZURE_MONITOR_AVAILABLE', True)
    def test_handles_init_error(self) -> None:
        """initialize_telemetry handles SDK initialization errors gracefully."""
        mock_configure = MagicMock(side_effect=Exception("SDK error"))
        telemetry_module.configure_azure_monitor = mock_configure

        try:
            mock_app = MagicMock()
            # Should not raise
            initialize_telemetry(mock_app)
            mock_app.add_middleware.assert_not_called()
        finally:
            telemetry_module._telemetry_initialized = False
            telemetry_module._tracer = None
