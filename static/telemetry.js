/**
 * Frontend telemetry module for Azure Application Insights.
 *
 * Exposes a global SHTelemetry object with safe no-op methods
 * when Application Insights is not configured.
 */

/* global Microsoft */

const SHTelemetry = (function () {
    let _appInsights = null;
    let _initialized = false;
    let _eventQueue = [];

    /**
     * Initialize Application Insights with the given connection string.
     *
     * Dynamically loads the App Insights JS SDK from the Microsoft CDN,
     * then flushes any events buffered before initialization.
     *
     * @param {string} connectionString - The App Insights connection string.
     * @param {string} [workspaceId] - Optional workspace ID for user context.
     */
    function initialize(connectionString, workspaceId) {
        if (!connectionString || _initialized) return;

        try {
            var script = document.createElement('script');
            script.src = 'https://js.monitor.azure.com/scripts/b/ai.3.gbl.min.js';
            script.crossOrigin = 'anonymous';

            script.onload = function () {
                try {
                    var sdkConfig = {
                        connectionString: connectionString,
                        enableAutoRouteTracking: false,
                        disableFetchTracking: true,
                        disableAjaxTracking: false,
                    };

                    // Check if the global SDK loaded
                    if (typeof Microsoft !== 'undefined' &&
                        Microsoft.ApplicationInsights &&
                        Microsoft.ApplicationInsights.ApplicationInsights) {

                        var appInsightsInstance = new Microsoft.ApplicationInsights.ApplicationInsights({
                            config: sdkConfig
                        });
                        _appInsights = appInsightsInstance.loadAppInsights();

                        // Set workspace ID as authenticated user context
                        if (workspaceId && _appInsights.context && _appInsights.context.user) {
                            _appInsights.context.user.authenticatedId = workspaceId;
                        }

                        _initialized = true;

                        // Flush queued events
                        _eventQueue.forEach(function (entry) {
                            _trackEventInternal(entry.name, entry.properties, entry.measurements);
                        });
                        _eventQueue = [];

                        // Track initial page view
                        trackPageView();
                    }
                } catch (e) {
                    // SDK init failed -- silent no-op
                }
            };

            script.onerror = function () {
                // CDN unreachable -- silent no-op
            };

            document.head.appendChild(script);

            // Register global error handlers
            _registerErrorHandlers();
        } catch (e) {
            // Initialization error -- silent no-op
        }
    }

    /**
     * Register global error handlers to capture unhandled exceptions.
     */
    function _registerErrorHandlers() {
        window.addEventListener('error', function (event) {
            trackError(event.error || event.message, {
                source: event.filename || '',
                lineno: String(event.lineno || ''),
                colno: String(event.colno || ''),
            });
        });

        window.addEventListener('unhandledrejection', function (event) {
            trackError(event.reason || 'Unhandled promise rejection', {
                type: 'unhandledrejection',
            });
        });
    }

    /**
     * Internal event tracking (only called when SDK is ready).
     */
    function _trackEventInternal(name, properties, measurements) {
        if (!_appInsights) return;
        try {
            _appInsights.trackEvent({
                name: name,
                properties: properties || {},
                measurements: measurements || {},
            });
        } catch (e) {
            // Tracking error -- silent no-op
        }
    }

    /**
     * Track a custom event.
     *
     * If the SDK is not yet loaded, events are buffered and flushed
     * once initialization completes.
     *
     * @param {string} name - Event name (e.g. 'tab_switch').
     * @param {Object} [properties] - String key-value pairs.
     * @param {Object} [measurements] - Numeric measurements.
     */
    function trackEvent(name, properties, measurements) {
        if (_initialized) {
            _trackEventInternal(name, properties, measurements);
        } else if (_eventQueue.length < 100) {
            _eventQueue.push({ name: name, properties: properties, measurements: measurements });
        }
    }

    /**
     * Track a page view.
     *
     * @param {string} [pageName] - Optional page name override.
     */
    function trackPageView(pageName) {
        if (!_appInsights) return;
        try {
            _appInsights.trackPageView({ name: pageName || document.title });
        } catch (e) {
            // Tracking error -- silent no-op
        }
    }

    /**
     * Track an error/exception.
     *
     * @param {Error|string} error - The error object or message.
     * @param {Object} [properties] - Additional context properties.
     */
    function trackError(error, properties) {
        if (!_appInsights) return;
        try {
            var exception = error instanceof Error ? error : new Error(String(error));
            _appInsights.trackException({
                exception: exception,
                properties: properties || {},
            });
        } catch (e) {
            // Tracking error -- silent no-op
        }
    }

    return {
        initialize: initialize,
        trackEvent: trackEvent,
        trackPageView: trackPageView,
        trackError: trackError,
    };
})();
