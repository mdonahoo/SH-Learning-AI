/**
 * Audio Analyzer Frontend Application
 *
 * Handles audio recording, file upload, and results display.
 */

// ============================================================================
// Audio Recorder
// ============================================================================

class AudioRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.startTime = null;
        this.timerInterval = null;
    }

    async start() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 16000
                }
            });

            // Try WebM first, fall back to other formats
            const mimeTypes = [
                'audio/webm;codecs=opus',
                'audio/webm',
                'audio/ogg;codecs=opus',
                'audio/mp4'
            ];

            let selectedMimeType = '';
            for (const mimeType of mimeTypes) {
                if (MediaRecorder.isTypeSupported(mimeType)) {
                    selectedMimeType = mimeType;
                    break;
                }
            }

            const options = selectedMimeType ? { mimeType: selectedMimeType } : {};
            this.mediaRecorder = new MediaRecorder(stream, options);
            this.audioChunks = [];

            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };

            this.mediaRecorder.start(1000); // Collect data every second
            this.isRecording = true;
            this.startTime = Date.now();

            return true;
        } catch (error) {
            console.error('Failed to start recording:', error);
            throw error;
        }
    }

    stop() {
        return new Promise((resolve) => {
            if (!this.mediaRecorder) {
                resolve(null);
                return;
            }

            this.mediaRecorder.onstop = () => {
                const mimeType = this.mediaRecorder.mimeType || 'audio/webm';
                const blob = new Blob(this.audioChunks, { type: mimeType });
                this.isRecording = false;

                // Stop all tracks
                this.mediaRecorder.stream.getTracks().forEach(track => track.stop());

                resolve(blob);
            };

            this.mediaRecorder.stop();
        });
    }

    getElapsedTime() {
        if (!this.startTime) return 0;
        return Math.floor((Date.now() - this.startTime) / 1000);
    }
}


// ============================================================================
// API Client
// ============================================================================

class ApiClient {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
    }

    async analyze(file, options = {}) {
        const formData = new FormData();
        formData.append('file', file);

        const params = new URLSearchParams();
        if (options.includeDiarization !== undefined) {
            params.append('include_diarization', options.includeDiarization);
        }
        if (options.includeQuality !== undefined) {
            params.append('include_quality', options.includeQuality);
        }

        const url = `${this.baseUrl}/api/analyze?${params.toString()}`;

        const response = await fetch(url, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'Unknown error' }));
            throw new Error(error.detail || error.error || `HTTP ${response.status}`);
        }

        return response.json();
    }

    async transcribe(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${this.baseUrl}/api/transcribe`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'Unknown error' }));
            throw new Error(error.detail || error.error || `HTTP ${response.status}`);
        }

        return response.json();
    }

    analyzeWithProgress(file, options = {}, onProgress) {
        return new Promise((resolve, reject) => {
            const formData = new FormData();
            formData.append('file', file);

            // Build query string from options
            const params = new URLSearchParams();
            if (options.includeNarrative !== undefined) {
                params.append('include_narrative', options.includeNarrative);
            }
            if (options.includeStory !== undefined) {
                params.append('include_story', options.includeStory);
            }
            if (options.telemetrySessionId) {
                params.append('telemetry_session_id', options.telemetrySessionId);
            }
            const queryString = params.toString();
            const url = queryString
                ? `${this.baseUrl}/api/analyze-stream?${queryString}`
                : `${this.baseUrl}/api/analyze-stream`;

            fetch(url, {
                method: 'POST',
                body: formData
            }).then(response => {
                if (!response.ok) {
                    reject(new Error(`HTTP ${response.status}`));
                    return;
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                const processChunk = ({ done, value }) => {
                    if (done) {
                        reject(new Error('Stream ended unexpectedly'));
                        return;
                    }

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || '';

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                if (data.type === 'progress') {
                                    if (onProgress) {
                                        onProgress(data.step, data.label, data.progress);
                                    }
                                } else if (data.type === 'result') {
                                    resolve(data.data);
                                    return;
                                } else if (data.type === 'error') {
                                    reject(new Error(data.message));
                                    return;
                                }
                            } catch (e) {
                                console.warn('Failed to parse SSE data:', line);
                            }
                        }
                    }

                    reader.read().then(processChunk).catch(reject);
                };

                reader.read().then(processChunk).catch(reject);
            }).catch(reject);
        });
    }

    async health() {
        const response = await fetch(`${this.baseUrl}/api/health`);
        return response.json();
    }

    async listAnalyses() {
        const response = await fetch(`${this.baseUrl}/api/analyses`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        return response.json();
    }

    async getArchiveIndex(options = {}) {
        const params = new URLSearchParams();
        if (options.starredOnly) params.append('starred_only', 'true');
        if (options.tag) params.append('tag', options.tag);
        if (options.search) params.append('search', options.search);

        const response = await fetch(`${this.baseUrl}/api/archive-index?${params.toString()}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        return response.json();
    }

    async getServicesStatus() {
        const response = await fetch(`${this.baseUrl}/api/services-status`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        return response.json();
    }

    async getAnalysis(filename) {
        const response = await fetch(`${this.baseUrl}/api/analyses/${filename}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        return response.json();
    }

    async deleteAnalysis(filename) {
        const response = await fetch(`${this.baseUrl}/api/analyses/${filename}`, {
            method: 'DELETE'
        });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        return response.json();
    }

    async updateMetadata(filename, metadata) {
        const response = await fetch(`${this.baseUrl}/api/analyses/${filename}/metadata`, {
            method: 'PATCH',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(metadata)
        });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        return response.json();
    }

    async startTelemetry() {
        const response = await fetch(`${this.baseUrl}/api/telemetry/start`, {
            method: 'POST'
        });
        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'Unknown error' }));
            throw new Error(error.detail || error.error || `HTTP ${response.status}`);
        }
        return response.json();
    }

    async stopTelemetry(sessionId) {
        const response = await fetch(`${this.baseUrl}/api/telemetry/stop`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ session_id: sessionId })
        });
        if (!response.ok) {
            const error = await response.json().catch(() => ({ error: 'Unknown error' }));
            throw new Error(error.detail || error.error || `HTTP ${response.status}`);
        }
        return response.json();
    }

    async getTelemetryStatus(sessionId) {
        const response = await fetch(`${this.baseUrl}/api/telemetry/status/${sessionId}`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        return response.json();
    }
}


// ============================================================================
// Results Renderer
// ============================================================================

class ResultsRenderer {
    constructor(container) {
        this.container = container;
        // Speaker-to-role mapping for displaying roles instead of speaker IDs
        this.speakerRoles = {};
        // Chart.js instances for cleanup
        this.chartInstances = [];
    }

    /**
     * Build speaker-to-role mapping from role assignments.
     * This allows all render methods to display roles instead of "speaker_X".
     */
    buildSpeakerRoleMap(roleAssignments) {
        this.speakerRoles = {};
        if (!roleAssignments) return;

        roleAssignments.forEach(ra => {
            if (ra.speaker_id && ra.role) {
                // Extract just the role name (e.g., "Captain" from "Captain/Command")
                const roleName = ra.role.split('/')[0];
                this.speakerRoles[ra.speaker_id] = {
                    role: roleName,
                    fullRole: ra.role,
                    confidence: ra.confidence || 0
                };
            }
        });
    }

    /**
     * Get display name for a speaker - prefers role name over speaker_id.
     * @param {string} speakerId - The speaker ID (e.g., "speaker_1")
     * @param {boolean} includeId - Whether to include speaker ID as suffix for disambiguation
     * @returns {string} Display name (e.g., "Captain" or "Captain (1)")
     */
    getSpeakerDisplayName(speakerId, includeId = false) {
        if (!speakerId) return 'Unknown';

        const roleInfo = this.speakerRoles[speakerId];
        if (roleInfo && roleInfo.role && roleInfo.role !== 'Unknown') {
            if (includeId) {
                // Extract number from speaker_id for disambiguation
                const num = speakerId.replace(/\D/g, '');
                return `${roleInfo.role} (${num})`;
            }
            return roleInfo.role;
        }

        // Fallback: format speaker_id more nicely
        const num = speakerId.replace(/\D/g, '');
        return `Crew ${num}`;
    }

    /**
     * Get role badge HTML for a speaker.
     */
    getSpeakerRoleBadge(speakerId) {
        const roleInfo = this.speakerRoles[speakerId];
        if (!roleInfo) return '';

        const confClass = roleInfo.confidence >= 0.7 ? 'high' : roleInfo.confidence >= 0.4 ? 'medium' : 'low';
        return `<span class="role-badge ${confClass}">${roleInfo.role}</span>`;
    }

    render(results) {
        // Build speaker-to-role mapping first (before any rendering)
        this.buildSpeakerRoleMap(results.role_assignments);

        // Update stats
        document.getElementById('duration').textContent =
            this.formatDuration(results.duration_seconds);
        document.getElementById('segment-count').textContent =
            results.transcription.length;
        document.getElementById('proc-time').textContent =
            `${results.processing_time_seconds.toFixed(1)}s`;

        // Destroy old Chart.js instances
        this.destroyCharts();

        // Render summary tab
        this.renderSummary(results);

        // Render bridge station map
        this.renderBridgeMap(results.role_assignments, results.speakers);

        // Render transcript
        this.renderTranscript(results.transcription, results.full_text);

        // Render mission timeline (with sentiment if available)
        this.renderTimeline(results.transcription, results.duration_seconds, results.sentiment_summary);

        // Render waveform amplitude
        this.renderWaveform(results.waveform_data, results.duration_seconds);

        // Render speakers (with stress data if available)
        this.renderSpeakers(results.speakers, results.role_assignments, results.sentiment_summary?.speaker_stress);

        // Render scorecards + radar charts
        this.renderScorecards(results.speaker_scorecards);
        this.renderRadarCharts(results.speaker_scorecards);

        // Render stress timeline chart
        this.renderStressTimeline(results.sentiment_summary);

        // Render quality
        this.renderQuality(results.communication_quality, results.confidence_distribution);

        // Render learning
        this.renderLearning(results.learning_evaluation);

        // Render 7 Habits
        this.renderHabits(results.seven_habits);

        // Render Training recommendations
        this.renderTraining(results.training_recommendations);

        // Render AI Narrative (pass llm_skipped_reason for long recordings)
        this.renderNarrative(results.narrative_summary, results.llm_skipped_reason);

        // Render AI Story (pass llm_skipped_reason for long recordings)
        this.renderStory(results.story_narrative, results.llm_skipped_reason);
    }

    renderSummary(results) {
        // Update summary cards
        document.getElementById('summary-duration').textContent =
            this.formatDuration(results.duration_seconds);
        document.getElementById('summary-speakers').textContent =
            results.speakers?.length || 0;
        document.getElementById('summary-segments').textContent =
            results.transcription?.length || 0;

        // Effective communication percentage
        const effectivePct = results.communication_quality?.effective_percentage || 0;
        document.getElementById('summary-effective').textContent =
            `${effectivePct.toFixed(0)}%`;

        // Stress metric card
        const stressCard = document.getElementById('stress-metric-card');
        const stressValueEl = document.getElementById('summary-stress');
        const avgStress = results.sentiment_summary?.average_stress;
        if (avgStress != null && stressCard && stressValueEl) {
            // Backend uses 0-1 scale, convert to percentage
            const stressPct = Math.round(avgStress * 100);
            stressValueEl.textContent = `${stressPct}%`;
            stressCard.classList.remove('hidden', 'stress-low', 'stress-moderate', 'stress-high');
            if (stressPct < 40) {
                stressCard.classList.add('stress-low');
            } else if (stressPct < 70) {
                stressCard.classList.add('stress-moderate');
            } else {
                stressCard.classList.add('stress-high');
            }
        } else if (stressCard) {
            stressCard.classList.add('hidden');
        }

        // Key metrics bars
        const habitsScore = results.seven_habits?.overall_score || 0;
        const habitsPercent = (habitsScore / 5) * 100;
        document.getElementById('metric-habits-bar').style.width = `${habitsPercent}%`;
        document.getElementById('metric-habits-value').textContent = `${habitsScore.toFixed(1)}/5`;

        document.getElementById('metric-quality-bar').style.width = `${effectivePct}%`;
        document.getElementById('metric-quality-value').textContent = `${effectivePct.toFixed(0)}%`;

        const avgConfidence = results.confidence_distribution?.average_confidence || 0;
        this.lastAvgConfidence = avgConfidence;
        const confidencePct = avgConfidence * 100;
        document.getElementById('metric-confidence-bar').style.width = `${confidencePct}%`;
        const confValueEl = document.getElementById('metric-confidence-value');
        confValueEl.textContent = `${confidencePct.toFixed(0)}%`;

        // Remove any existing provisional badges before adding new one
        const existingBadges = confValueEl.parentElement.querySelectorAll('.provisional-badge');
        existingBadges.forEach(badge => badge.remove());

        if (avgConfidence < 0.5) {
            confValueEl.insertAdjacentHTML('afterend', '<span class="provisional-badge" data-tooltip="Scores are estimates due to low audio quality. Results may be less accurate than higher-confidence analyses.">Provisional</span>');
        }

        // Top recommendations preview
        const topRecs = document.getElementById('top-recommendations');
        const recsPreview = document.getElementById('recommendations-preview');

        if (results.training_recommendations?.immediate_actions?.length > 0) {
            const actions = results.training_recommendations.immediate_actions.slice(0, 3);
            const getPriorityIcon = (priority) => {
                switch(priority.toUpperCase()) {
                    case 'CRITICAL': return '<svg class="icon icon-sm"><use href="#icon-shield"></use></svg>';
                    case 'HIGH': return '<svg class="icon icon-sm"><use href="#icon-delta"></use></svg>';
                    case 'MEDIUM': return '<svg class="icon icon-sm"><use href="#icon-crosshair"></use></svg>';
                    default: return '<svg class="icon icon-sm"><use href="#icon-star"></use></svg>';
                }
            };
            recsPreview.innerHTML = actions.map(action => `
                <div class="rec-preview-item priority-${action.priority.toLowerCase()}">
                    <div class="rec-priority-indicator">
                        <span class="rec-priority-icon">${getPriorityIcon(action.priority)}</span>
                        <span class="rec-priority-badge priority-${action.priority.toLowerCase()}">${action.priority.toUpperCase() === 'CRITICAL' ? 'MISSION CRITICAL' : action.priority}</span>
                    </div>
                    <div class="rec-content">
                        <span class="rec-preview-title">${action.title}</span>
                        ${action.description ? `<span class="rec-preview-desc">${action.description.substring(0, 100)}${action.description.length > 100 ? '...' : ''}</span>` : ''}
                    </div>
                </div>
            `).join('');
            topRecs.classList.remove('hidden');
        } else {
            topRecs.classList.add('hidden');
        }
    }

    highlightKeywords(text) {
        const keywords = /\b(shields|fire|warp|red alert|enemy|damage|torpedo|phaser|evasive|hull|breach|sensors|cloak|engage|hail|alert|power|impulse|photon|deflector|tractor beam|weapons|helm|captain|commander)\b/gi;
        return this.escapeHtml(text).replace(keywords, '<span class="keyword-highlight">$&</span>');
    }

    renderTranscript(segments, fullText) {
        const container = document.getElementById('transcript-content');

        if (!segments || segments.length === 0) {
            container.innerHTML = '<p class="empty">No transcription available</p>';
        } else {
            const isCaptain = (speakerId) => {
                const roleInfo = this.speakerRoles[speakerId];
                return roleInfo && (roleInfo.role === 'Captain/Command' || roleInfo.role === 'Captain');
            };

            container.innerHTML = `<div class="chat-transcript">${segments.map((seg, idx) => {
                const displayName = this.getSpeakerDisplayName(seg.speaker_id, true);
                const conf = seg.confidence || 0;
                const confPct = (conf * 100).toFixed(0);
                const confClass = conf >= 0.7 ? 'conf-high' : conf >= 0.4 ? 'conf-medium' : 'conf-low';
                const bubbleAlign = isCaptain(seg.speaker_id) ? 'bubble-right' : 'bubble-left';
                return `
                <div class="chat-bubble ${bubbleAlign}" data-index="${idx}">
                    <span class="bubble-speaker">${displayName}</span>
                    <div class="bubble-text">${this.highlightKeywords(seg.text)}</div>
                    <div class="bubble-meta">
                        <span class="bubble-time">${this.formatTime(seg.start_time)} - ${this.formatTime(seg.end_time)}</span>
                        <div class="bubble-meta-badges">
                            ${seg.sentiment?.emotion ? `<span class="emotion-badge emotion-${seg.sentiment.emotion}">${seg.sentiment.emotion}</span>` : ''}
                            <span class="bubble-confidence ${confClass}" title="Signal quality">${confPct}%</span>
                        </div>
                    </div>
                </div>
            `}).join('')}</div>`;
        }

        document.getElementById('full-text').textContent = fullText || '';
    }

    renderSpeakers(speakers, roleAssignments, speakerStress) {
        const container = document.getElementById('speakers-content');

        // Define standard bridge roles with icons and descriptions
        const BRIDGE_ROLES = [
            { id: 'Captain/Command', name: 'Captain', icon: '<svg class="icon"><use href="#icon-captain"></use></svg>', desc: 'Commands the bridge crew' },
            { id: 'Helm/Navigation', name: 'Helm', icon: '<svg class="icon"><use href="#icon-helm"></use></svg>', desc: 'Pilots the ship' },
            { id: 'Tactical/Weapons', name: 'Tactical', icon: '<svg class="icon"><use href="#icon-tactical"></use></svg>', desc: 'Weapons and defense' },
            { id: 'Science/Sensors', name: 'Science', icon: '<svg class="icon"><use href="#icon-science"></use></svg>', desc: 'Sensors and analysis' },
            { id: 'Engineering/Systems', name: 'Engineering', icon: '<svg class="icon"><use href="#icon-engineering"></use></svg>', desc: 'Ship systems and power' },
            { id: 'Operations/Monitoring', name: 'Operations', icon: '<svg class="icon"><use href="#icon-crosshair"></use></svg>', desc: 'Monitoring and logistics' },
            { id: 'Communications', name: 'Comms', icon: '<svg class="icon"><use href="#icon-comms"></use></svg>', desc: 'Hailing and signals' },
        ];

        // Build speaker data lookup
        const speakerDataMap = {};
        if (speakers) {
            for (const s of speakers) {
                speakerDataMap[s.speaker_id] = s;
            }
        }

        // Build role-to-speakers mapping from role_assignments
        const roleToSpeakers = {};
        const unassignedSpeakers = [];

        if (roleAssignments && roleAssignments.length > 0) {
            for (const ra of roleAssignments) {
                const role = ra.role || 'Crew Member';
                if (!roleToSpeakers[role]) {
                    roleToSpeakers[role] = [];
                }
                roleToSpeakers[role].push({
                    speaker_id: ra.speaker_id,
                    confidence: ra.confidence,
                    voice_confidence: ra.voice_confidence,
                    telemetry_confidence: ra.telemetry_confidence,
                    evidence_count: ra.evidence_count,
                    methodology_note: ra.methodology_note,
                    key_indicators: ra.key_indicators || [],
                    speakerData: speakerDataMap[ra.speaker_id] || {}
                });
            }
        } else if (speakers) {
            // No role assignments - all speakers are unassigned
            for (const s of speakers) {
                unassignedSpeakers.push({
                    speaker_id: s.speaker_id,
                    speakerData: s
                });
            }
        }

        if (!speakers || speakers.length === 0) {
            container.innerHTML = '<p class="empty">No speaker information available</p>';
            return;
        }

        // Helper function to get confidence class
        const getConfClass = (conf) => {
            if (conf >= 0.8) return 'high';
            if (conf >= 0.5) return 'medium';
            return 'low';
        };

        // Calculate total speaking time for participation percentage
        const totalSpeakingTime = speakers.reduce((sum, s) => sum + (s.total_speaking_time || 0), 0);

        // Render role cards
        let html = '<div class="roles-grid">';

        for (const roleDef of BRIDGE_ROLES) {
            const assignedSpeakers = roleToSpeakers[roleDef.id] || [];
            const hasAssignment = assignedSpeakers.length > 0;
            const topSpeaker = assignedSpeakers[0];

            html += `
                <div class="role-card ${hasAssignment ? 'assigned' : 'unassigned'}">
                    <div class="role-card-header">
                        <span class="role-icon">${roleDef.icon}</span>
                        <div class="role-info">
                            <span class="role-name">${roleDef.name}</span>
                            <span class="role-desc">${roleDef.desc}</span>
                        </div>
                    </div>
                    <div class="role-card-body">
                        ${hasAssignment ? assignedSpeakers.map(sp => {
                            const conf = sp.confidence || 0;
                            const voiceConf = sp.voice_confidence;
                            const telemetryConf = sp.telemetry_confidence;
                            const evidenceCount = sp.evidence_count || 0;
                            const data = sp.speakerData || {};
                            const speakingTime = data.total_speaking_time || 0;
                            const utterances = data.utterance_count || 0;
                            const participation = totalSpeakingTime > 0 ? (speakingTime / totalSpeakingTime * 100) : 0;

                            return `
                                <div class="role-speaker">
                                    <div class="role-speaker-header">
                                        <span class="speaker-name">${this.getSpeakerDisplayName(sp.speaker_id)}</span>
                                        <span class="speaker-confidence ${getConfClass(conf)}">${(conf * 100).toFixed(0)}%</span>
                                    </div>

                                    <div class="crew-metrics">
                                        <div class="crew-metric">
                                            <div class="crew-metric-value">${this.formatDuration(speakingTime)}</div>
                                            <div class="crew-metric-label">Speaking Time</div>
                                        </div>
                                        <div class="crew-metric">
                                            <div class="crew-metric-value">${utterances}</div>
                                            <div class="crew-metric-label">Utterances</div>
                                        </div>
                                    </div>

                                    <div class="participation-bar">
                                        <div class="participation-bar-label">
                                            <span>Participation</span>
                                            <span>${participation.toFixed(1)}%</span>
                                        </div>
                                        <div class="participation-bar-track">
                                            <div class="participation-bar-fill" style="width: ${Math.min(participation, 100)}%"></div>
                                        </div>
                                    </div>

                                    ${(() => {
                                        const stressInfo = speakerStress?.[sp.speaker_id];
                                        if (!stressInfo) return '';
                                        // Backend uses avg_stress (0-1 scale), convert to percentage
                                        const avgStress = stressInfo.avg_stress != null ? Math.round(stressInfo.avg_stress * 100) : null;
                                        const domEmotion = stressInfo.dominant_emotion || '';
                                        if (avgStress == null) return '';
                                        const stressClass = avgStress < 40 ? 'stress-low' : avgStress < 70 ? 'stress-moderate' : 'stress-high';
                                        return `
                                            <div class="speaker-stress-info">
                                                <div class="speaker-stress-row">
                                                    <span class="stress-indicator-label">Avg Stress</span>
                                                    <span class="stress-indicator-value ${stressClass}">${avgStress}%</span>
                                                    ${domEmotion ? `<span class="emotion-badge emotion-${domEmotion}">${domEmotion}</span>` : ''}
                                                </div>
                                            </div>
                                        `;
                                    })()}

                                    ${(voiceConf !== undefined || (telemetryConf !== undefined && telemetryConf > 0)) ? `
                                        <div class="confidence-sources">
                                            ${voiceConf !== undefined ? `
                                                <span class="conf-source voice" title="Based on speech patterns and keywords">
                                                    <span class="source-icon"><svg class="icon icon-sm"><use href="#icon-waveform"></use></svg></span> Voice ${(voiceConf * 100).toFixed(0)}%
                                                </span>
                                            ` : ''}
                                            ${telemetryConf !== undefined && telemetryConf > 0 ? `
                                                <span class="conf-source telemetry" title="${evidenceCount} console actions matched">
                                                    <span class="source-icon"><svg class="icon icon-sm"><use href="#icon-crosshair"></use></svg></span> Telemetry +${(telemetryConf * 100).toFixed(0)}%
                                                </span>
                                            ` : ''}
                                        </div>
                                    ` : ''}

                                    ${sp.key_indicators && sp.key_indicators.length > 0 ? `
                                        <div class="key-indicators">
                                            ${sp.key_indicators.slice(0, 5).map(kw => `<span class="keyword-tag">"${kw}"</span>`).join('')}
                                        </div>
                                    ` : ''}
                                </div>
                            `;
                        }).join('') : `
                            <div class="role-empty station-standby">
                                <div class="standby-circle"></div>
                                <span class="empty-text">Station Offline</span>
                            </div>
                        `}
                    </div>
                </div>
            `;
        }

        // Handle "Crew Member" / Unknown role assignments
        const crewMembers = roleToSpeakers['Crew Member'] || [];
        if (crewMembers.length > 0 || unassignedSpeakers.length > 0) {
            const allUnassigned = [...crewMembers, ...unassignedSpeakers];
            html += `
                <div class="role-card unassigned crew-members">
                    <div class="role-card-header">
                        <span class="role-icon"><svg class="icon"><use href="#icon-crosshair"></use></svg></span>
                        <div class="role-info">
                            <span class="role-name">Other Crew</span>
                            <span class="role-desc">Role not yet determined</span>
                        </div>
                    </div>
                    <div class="role-card-body">
                        ${allUnassigned.map(sp => {
                            const data = sp.speakerData || {};
                            return `
                                <div class="role-speaker minor">
                                    <div class="role-speaker-header">
                                        <span class="speaker-name">${this.getSpeakerDisplayName(sp.speaker_id)}</span>
                                    </div>
                                    <div class="speaker-mini-stats">
                                        <span>${this.formatDuration(data.total_speaking_time || 0)}</span>
                                        <span>${data.utterance_count || 0} utterances</span>
                                    </div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
            `;
        }

        html += '</div>';
        container.innerHTML = html;
    }

    renderQuality(quality, confidenceDistribution) {
        const container = document.getElementById('quality-content');

        if (!quality) {
            container.innerHTML = '<p class="empty">No quality analysis available</p>';
            return;
        }

        // Calculate effective rate for color coding
        const effectiveRate = quality.effective_percentage || 0;
        const getScoreColor = (pct) => {
            if (pct >= 70) return '#22c55e';
            if (pct >= 40) return '#f59e0b';
            return '#ef4444';
        };
        const getScoreClass = (pct) => {
            if (pct >= 70) return 'high';
            if (pct >= 40) return 'medium';
            return 'low';
        };

        // Separate patterns by category
        const effectivePatterns = (quality.patterns || []).filter(p => p.category === 'effective');
        const improvementPatterns = (quality.patterns || []).filter(p => p.category === 'needs_improvement');

        container.innerHTML = `
            <div class="quality-header">
                <h2>Communication Quality Assessment</h2>
                <div class="quality-score-ring">
                    <svg viewBox="0 0 100 100" class="score-ring">
                        <circle cx="50" cy="50" r="45" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="8"/>
                        <circle cx="50" cy="50" r="45" fill="none" stroke="${getScoreColor(effectiveRate)}" stroke-width="8"
                            stroke-dasharray="${(effectiveRate / 100) * 283} 283"
                            stroke-linecap="round" transform="rotate(-90 50 50)"/>
                    </svg>
                    <div class="score-ring-text">
                        <span class="score-big">${effectiveRate.toFixed(0)}%</span>
                    </div>
                </div>
                <div class="overall-label">Effective Communication Rate</div>
            </div>

            <div class="quality-metrics-grid">
                <div class="quality-metric-card effective">
                    <div class="metric-icon">✓</div>
                    <div class="metric-value">${quality.effective_count}</div>
                    <div class="metric-label">Effective</div>
                    <div class="metric-sublabel">Clear, actionable communications</div>
                </div>
                <div class="quality-metric-card improvement">
                    <div class="metric-icon">↗</div>
                    <div class="metric-value">${quality.improvement_count}</div>
                    <div class="metric-label">Needs Improvement</div>
                    <div class="metric-sublabel">Opportunities for growth</div>
                </div>
                <div class="quality-metric-card total">
                    <div class="metric-icon"><svg class="icon"><use href="#icon-crosshair"></use></svg></div>
                    <div class="metric-value">${quality.total_utterances_assessed || (quality.effective_count + quality.improvement_count)}</div>
                    <div class="metric-label">Total Assessed</div>
                    <div class="metric-sublabel">Utterances analyzed</div>
                </div>
            </div>

            ${quality.calculation_summary ? `
                <details class="calculation-details">
                    <summary>
                        <span class="details-icon">▶</span>
                        Show Calculation Details
                    </summary>
                    <div class="calculation-content">
                        <p>${quality.calculation_summary}</p>
                    </div>
                </details>
            ` : ''}

            ${confidenceDistribution ? `
                <div class="confidence-panel">
                    <div class="confidence-header">
                        <h3>
                            <span class="section-icon"><svg class="icon icon-sm"><use href="#icon-crosshair"></use></svg></span>
                            Signal Quality
                        </h3>
                        <div class="confidence-summary">
                            <span class="conf-stat">
                                <span class="conf-label">Average</span>
                                <span class="conf-value ${getScoreClass(confidenceDistribution.average_confidence * 100)}">${(confidenceDistribution.average_confidence * 100).toFixed(0)}%</span>
                            </span>
                            <span class="conf-stat">
                                <span class="conf-label">Median</span>
                                <span class="conf-value">${(confidenceDistribution.median_confidence * 100).toFixed(0)}%</span>
                            </span>
                        </div>
                    </div>
                    <div class="confidence-bars">
                        ${confidenceDistribution.buckets.map(b => {
                            const maxCount = Math.max(...confidenceDistribution.buckets.map(x => x.count));
                            const barWidth = maxCount > 0 ? (b.count / maxCount) * 100 : 0;
                            const isHighest = b.count === maxCount && b.count > 0;
                            return `
                            <div class="conf-bar-row ${isHighest ? 'highest' : ''}">
                                <span class="conf-bar-label">${b.label}</span>
                                <div class="conf-bar-container">
                                    <div class="conf-bar-fill" style="width: ${barWidth}%"></div>
                                </div>
                                <span class="conf-bar-count">${b.count}</span>
                            </div>
                        `}).join('')}
                    </div>
                    ${confidenceDistribution.quality_assessment ? `
                        <div class="confidence-assessment ${getScoreClass(confidenceDistribution.average_confidence * 100)}">
                            ${confidenceDistribution.quality_assessment}
                        </div>
                    ` : ''}
                    <div class="confidence-explanation">
                        Signal quality reflects audio clarity in a live bridge environment. Scores below 50% are typical for multi-speaker, high-noise scenarios.
                    </div>
                </div>
            ` : ''}

            ${effectivePatterns.length > 0 ? `
                <div class="patterns-panel effective-patterns">
                    <h3>
                        <span class="section-icon">✓</span>
                        Effective Patterns
                    </h3>
                    <div class="patterns-grid">
                        ${effectivePatterns.map(p => `
                            <div class="pattern-card-new effective">
                                <div class="pattern-card-header">
                                    <span class="pattern-title">${this.formatPatternName(p.pattern_name)}</span>
                                    <span class="pattern-badge effective">${p.count}</span>
                                </div>
                                ${p.description ? `<p class="pattern-desc">${p.description}</p>` : ''}
                                ${p.examples?.length > 0 ? `
                                    <details class="pattern-evidence">
                                        <summary>
                                            <span class="details-icon">▶</span>
                                            View Examples (${p.examples.length})
                                        </summary>
                                        <div class="examples-content">
                                            ${p.examples.map(ex => {
                                                // Handle both object and string formats
                                                const text = typeof ex === 'string' ? ex : (ex.text || '');
                                                const speaker = typeof ex === 'string' ? 'Speaker' : (ex.speaker || 'Speaker');
                                                return `
                                                <div class="example-quote">
                                                    <span class="quote-speaker">${this.getSpeakerDisplayName(speaker)}</span>
                                                    <span class="quote-text">"${this.escapeHtml(text)}"</span>
                                                </div>
                                            `}).join('')}
                                        </div>
                                    </details>
                                ` : ''}
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}

            ${improvementPatterns.length > 0 ? `
                <div class="patterns-panel improvement-patterns">
                    <h3>
                        <span class="section-icon">↗</span>
                        Areas for Improvement
                    </h3>
                    <div class="patterns-grid">
                        ${improvementPatterns.map(p => `
                            <div class="pattern-card-new improvement">
                                <div class="pattern-card-header">
                                    <span class="pattern-title">${this.formatPatternName(p.pattern_name)}</span>
                                    <span class="pattern-badge improvement">${p.count}</span>
                                </div>
                                ${p.description ? `<p class="pattern-desc">${p.description}</p>` : ''}
                                ${p.examples?.length > 0 ? `
                                    <details class="pattern-evidence">
                                        <summary>
                                            <span class="details-icon">▶</span>
                                            View Examples (${p.examples.length})
                                        </summary>
                                        <div class="examples-content">
                                            ${p.examples.map(ex => {
                                                // Handle both object and string formats
                                                const text = typeof ex === 'string' ? ex : (ex.text || '');
                                                const speaker = typeof ex === 'string' ? 'Speaker' : (ex.speaker || 'Speaker');
                                                return `
                                                <div class="example-quote">
                                                    <span class="quote-speaker">${this.getSpeakerDisplayName(speaker)}</span>
                                                    <span class="quote-text">"${this.escapeHtml(text)}"</span>
                                                </div>
                                            `}).join('')}
                                        </div>
                                    </details>
                                ` : ''}
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
        `;
    }

    renderScorecards(scorecards) {
        const container = document.getElementById('scorecards-content');

        if (!scorecards || scorecards.length === 0) {
            container.innerHTML = '<p class="empty">No scorecard data available</p>';
            return;
        }

        const isProvisional = (this.lastAvgConfidence || 0) < 0.5;

        // Badge map for high-scoring metrics
        const badgeMap = {
            'Communication Clarity': 'Ace Communicator',
            'Protocol Adherence': 'Protocol Master',
            'Team Coordination': 'Team Synergy',
            'Situational Awareness': 'Tactical Eye',
            'Leadership Initiative': 'Command Presence',
            'Decision Quality': 'Sharp Thinker',
            'Crisis Management': 'Cool Under Fire'
        };

        container.innerHTML = scorecards.map(card => {
            // Determine earned badges
            const earnedBadges = card.metrics
                .filter(m => m.score >= 4 && badgeMap[m.name])
                .map(m => badgeMap[m.name]);

            return `
            <div class="scorecard">
                <div class="scorecard-header">
                    <h3>
                        ${this.getSpeakerDisplayName(card.speaker_id)}
                        ${card.role && !this.speakerRoles[card.speaker_id] ? `<span class="speaker-role">${card.role}</span>` : ''}
                        ${isProvisional ? '<span class="scorecard-provisional-label" data-tooltip="Low audio quality means these scores are estimates. Actual performance may differ.">Provisional</span>' : ''}
                    </h3>
                    <span class="overall-score">${card.overall_score.toFixed(1)}/5</span>
                </div>

                ${earnedBadges.length > 0 ? `
                    <div class="crew-badges">
                        ${earnedBadges.map(b => `<span class="crew-badge">${b}</span>`).join('')}
                    </div>
                ` : ''}

                <div class="score-grid">
                    ${card.metrics.map(m => {
                        const displayScore = isProvisional
                            ? `${Math.max(1, m.score - 1)}-${Math.min(5, m.score + 1)}/5`
                            : null;
                        return `
                        <div class="score-item">
                            <div class="score-item-header">
                                <span class="score-name">${m.name}</span>
                                <div class="score-value">
                                    ${displayScore ? `<span class="provisional-badge" style="font-size:0.6875rem" data-tooltip="Score range due to audio uncertainty">${displayScore}</span>` : ''}
                                    <div class="score-dots">
                                        ${[1,2,3,4,5].map(i => `
                                            <span class="score-dot ${i <= m.score ? 'filled score-' + m.score : ''}"></span>
                                        `).join('')}
                                    </div>
                                </div>
                            </div>
                            <div class="score-evidence">${this.escapeHtml(m.evidence)}</div>
                            ${(m.supporting_quotes && m.supporting_quotes.length > 0) || m.threshold_info ? `
                                <div class="score-quotes">
                                    <details>
                                        <summary>Show Evidence${m.supporting_quotes?.length ? ` (${m.supporting_quotes.length} quotes)` : ''}</summary>
                                        ${m.threshold_info ? `<p class="threshold-info"><strong>Scoring:</strong> ${m.threshold_info}</p>` : ''}
                                        ${m.calculation_details ? `<p class="calc-details">${m.calculation_details}</p>` : ''}
                                        ${m.supporting_quotes?.length > 0 ? `
                                            <ul class="quote-list">
                                                ${m.supporting_quotes.map(q => `<li class="quote">${this.escapeHtml(q)}</li>`).join('')}
                                            </ul>
                                        ` : ''}
                                    </details>
                                </div>
                            ` : ''}
                        </div>
                    `}).join('')}
                </div>

                ${(card.strengths?.length > 0 || card.areas_for_improvement?.length > 0) ? `
                    <div class="strengths-weaknesses">
                        ${card.strengths?.length > 0 ? `
                            <div class="strengths">
                                <h4>Strengths</h4>
                                <ul>${card.strengths.map(s => `<li>${this.escapeHtml(s)}</li>`).join('')}</ul>
                            </div>
                        ` : ''}
                        ${card.areas_for_improvement?.length > 0 ? `
                            <div class="weaknesses">
                                <h4>Areas for Improvement</h4>
                                <ul>${card.areas_for_improvement.map(a => `<li>${this.escapeHtml(a)}</li>`).join('')}</ul>
                            </div>
                        ` : ''}
                    </div>
                ` : ''}
            </div>
        `}).join('');
    }

    renderLearning(learning) {
        const container = document.getElementById('learning-content');

        if (!learning) {
            container.innerHTML = '<p class="empty">No learning evaluation available</p>';
            return;
        }

        container.innerHTML = `
            ${learning.kirkpatrick_levels ? `
                <div class="kirkpatrick-section">
                    <h3>Kirkpatrick 4-Level Model</h3>
                    <div class="kirkpatrick-grid">
                        ${learning.kirkpatrick_levels.map((lvl, idx) => `
                            <div class="kirk-card level-${lvl.level}">
                                <div class="kirk-header">
                                    <span class="kirk-level-num">Level ${lvl.level}</span>
                                </div>
                                <div class="kirk-name">${lvl.name}</div>
                                <div class="kirk-score">${(lvl.score * 100).toFixed(0)}%</div>
                                <div class="kirk-bar">
                                    <div class="kirk-bar-fill" style="width: ${lvl.score * 100}%"></div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}

            <div class="frameworks-section">
                <h3>Learning Frameworks</h3>
                <div class="frameworks-grid">
                    ${learning.blooms_level ? `
                        <div class="framework-card blooms">
                            <div class="framework-icon"><svg class="icon"><use href="#icon-science"></use></svg></div>
                            <div class="framework-label">Bloom's Taxonomy</div>
                            <div class="framework-value">${learning.blooms_level}</div>
                            <div class="framework-sublabel">Cognitive Level</div>
                        </div>
                    ` : ''}
                    ${learning.nasa_tlx_score !== undefined ? `
                        <div class="framework-card nasa">
                            <div class="framework-icon"><svg class="icon"><use href="#icon-delta"></use></svg></div>
                            <div class="framework-label">NASA TLX</div>
                            <div class="framework-value">${(learning.nasa_tlx_score * 100).toFixed(0)}%</div>
                            <div class="framework-sublabel">Workload Index</div>
                        </div>
                    ` : ''}
                    ${learning.engagement_score !== undefined ? `
                        <div class="framework-card engagement">
                            <div class="framework-icon"><svg class="icon"><use href="#icon-crosshair"></use></svg></div>
                            <div class="framework-label">Engagement</div>
                            <div class="framework-value">${(learning.engagement_score * 100).toFixed(0)}%</div>
                            <div class="framework-sublabel">Overall Score</div>
                        </div>
                    ` : ''}
                </div>
            </div>

            ${learning.top_communications && learning.top_communications.length > 0 ? `
                <div class="communications-section">
                    <h3>Key Communications</h3>
                    <p class="section-subtitle">Top utterances by transcription confidence</p>
                    <div class="communications-list">
                        ${learning.top_communications.map(comm => {
                            const confLevel = comm.confidence >= 0.7 ? 'high' : comm.confidence >= 0.4 ? 'medium' : 'low';
                            return `
                            <div class="comm-item">
                                <span class="comm-speaker">${this.getSpeakerDisplayName(comm.speaker)}</span>
                                <span class="comm-text">"${this.escapeHtml(comm.text)}"</span>
                                <span class="comm-confidence ${confLevel}">${(comm.confidence * 100).toFixed(0)}%</span>
                            </div>
                        `}).join('')}
                    </div>
                </div>
            ` : ''}

            ${learning.speaker_statistics && Object.keys(learning.speaker_statistics).length > 0 ? `
                <div class="speaker-stats-section">
                    <h3>Speaker Statistics</h3>
                    <div class="speaker-stats-list">
                        ${(Array.isArray(learning.speaker_statistics) ? learning.speaker_statistics : Object.values(learning.speaker_statistics)).map(stat => `
                            <div class="speaker-stat-row">
                                <span class="speaker-id">${this.getSpeakerDisplayName(stat.speaker)}</span>
                                <div class="stat-bar-wrap">
                                    <div class="stat-bar" style="width: ${stat.percentage}%"></div>
                                </div>
                                <span class="stat-info">
                                    <strong>${stat.utterances}</strong> utterances
                                    <span class="stat-pct">(${stat.percentage}%)</span>
                                </span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}

            ${learning.recommendations && learning.recommendations.length > 0 ? `
                <div class="learning-recs-section">
                    <h3>Recommendations</h3>
                    <div class="learning-recs-list">
                        ${learning.recommendations.map(r => `
                            <div class="learning-rec-item">
                                <span class="rec-bullet">→</span>
                                <span class="rec-text">${r}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
        `;
    }

    renderHabits(habits) {
        const container = document.getElementById('habits-content');

        if (!habits) {
            container.innerHTML = '<p class="empty">No 7 Habits assessment available</p>';
            return;
        }

        const getScoreClass = (score) => {
            if (score >= 4) return 'high';
            if (score >= 2.5) return 'medium';
            return 'low';
        };

        const getScoreColor = (score) => {
            if (score >= 4) return '#22c55e';
            if (score >= 2.5) return '#f59e0b';
            return '#ef4444';
        };

        const getScoreLabel = (score) => {
            if (score >= 4) return 'Excellent';
            if (score >= 3) return 'Good';
            if (score >= 2) return 'Developing';
            return 'Needs Focus';
        };

        // Icons and brief descriptions for each habit
        const habitInfo = {
            1: { icon: '<svg class="icon"><use href="#icon-crosshair"></use></svg>', brief: 'Taking responsibility and initiative' },
            2: { icon: '<svg class="icon"><use href="#icon-helm"></use></svg>', brief: 'Having clear goals and vision' },
            3: { icon: '<svg class="icon"><use href="#icon-tactical"></use></svg>', brief: 'Prioritizing important tasks first' },
            4: { icon: '<svg class="icon"><use href="#icon-comms"></use></svg>', brief: 'Finding solutions that benefit everyone' },
            5: { icon: '<svg class="icon"><use href="#icon-waveform"></use></svg>', brief: 'Understanding others before being understood' },
            6: { icon: '<svg class="icon"><use href="#icon-engineering"></use></svg>', brief: 'Creative collaboration for better results' },
            7: { icon: '<svg class="icon"><use href="#icon-refresh"></use></svg>', brief: 'Continuous improvement and learning' }
        };

        container.innerHTML = `
            <div class="habits-intro">
                <div class="habits-intro-text">
                    <h2>7 Habits of Highly Effective People</h2>
                    <p>Assessment based on Stephen Covey's leadership framework, measuring team communication patterns.</p>
                </div>
                <div class="habits-overall-score">
                    <div class="overall-score-ring">
                        <svg viewBox="0 0 100 100" class="score-ring">
                            <circle cx="50" cy="50" r="45" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="8"/>
                            <circle cx="50" cy="50" r="45" fill="none" stroke="${getScoreColor(habits.overall_score)}" stroke-width="8"
                                stroke-dasharray="${(habits.overall_score / 5) * 283} 283"
                                stroke-linecap="round" transform="rotate(-90 50 50)"/>
                        </svg>
                        <div class="score-ring-text">
                            <span class="score-big">${habits.overall_score.toFixed(1)}</span>
                            <span class="score-max">/5</span>
                        </div>
                    </div>
                    <div class="overall-label">${getScoreLabel(habits.overall_score)}</div>
                </div>
            </div>

            <div class="power-level-bar">
                <div class="power-level-label">System Power Level</div>
                <div class="power-bar-track">
                    <div class="power-bar-fill" style="width: ${(habits.overall_score / 5) * 100}%"></div>
                </div>
                <div class="power-value">${((habits.overall_score / 5) * 100).toFixed(0)}%</div>
            </div>

            <div class="habits-grid">
                ${habits.habits.map(h => {
                    const scoreClass = getScoreClass(h.score);
                    const info = habitInfo[h.habit_number] || { icon: '<svg class="icon"><use href="#icon-delta"></use></svg>', brief: '' };
                    const scorePercent = (h.score / 5) * 100;
                    return `
                    <div class="habit-card ${scoreClass}">
                        <div class="habit-card-header">
                            <div class="habit-icon-wrap">
                                <span class="habit-icon">${info.icon}</span>
                            </div>
                            <div class="habit-info">
                                <span class="habit-num">Habit ${h.habit_number}</span>
                                <span class="habit-name">${h.youth_friendly_name || h.habit_name}</span>
                            </div>
                        </div>

                        <div class="habit-brief">${info.brief}</div>

                        <div class="habit-score-section">
                            <div class="habit-score-bar">
                                <div class="habit-score-fill ${scoreClass}" style="width: ${scorePercent}%"></div>
                            </div>
                            <div class="habit-score-info">
                                <span class="habit-score-value ${scoreClass}">${h.score}/5</span>
                                <span class="habit-score-label">${getScoreLabel(h.score)}</span>
                            </div>
                        </div>

                        <div class="habit-stats">
                            <span class="habit-stat">
                                <span class="stat-value">${h.observation_count}</span>
                                <span class="stat-label">observations</span>
                            </span>
                        </div>

                        <div class="habit-interpretation">${h.interpretation}</div>

                        ${h.development_tip ? `
                            <div class="habit-tip-box">
                                <span class="tip-icon"><svg class="icon icon-sm"><use href="#icon-delta"></use></svg></span>
                                <div class="tip-content">
                                    <span class="tip-label">Growth Tip</span>
                                    <span class="tip-text">${h.development_tip}</span>
                                </div>
                            </div>
                        ` : ''}

                        ${(h.examples?.length > 0 || h.gap_to_next_score) ? `
                            <details class="habit-details">
                                <summary>
                                    <span class="details-icon">▶</span>
                                    View Evidence & Progress Path
                                </summary>
                                <div class="habit-evidence-content">
                                    ${h.gap_to_next_score ? `
                                        <div class="gap-info">
                                            <span class="evidence-label">How to Improve</span>
                                            <p>${h.gap_to_next_score}</p>
                                        </div>
                                    ` : ''}
                                    ${h.examples?.length > 0 ? `
                                        <div class="examples-list">
                                            <span class="evidence-label">Examples from Session</span>
                                            ${h.examples.slice(0, 3).map(ex => `
                                                <div class="example-item">
                                                    <span class="ex-speaker">${this.getSpeakerDisplayName(ex.speaker) || 'Speaker'}</span>
                                                    <span class="ex-text">"${this.escapeHtml(ex.text || ex)}"</span>
                                                </div>
                                            `).join('')}
                                            ${h.examples.length > 3 ? `<div class="more-examples">+${h.examples.length - 3} more examples</div>` : ''}
                                        </div>
                                    ` : ''}
                                </div>
                            </details>
                        ` : ''}
                    </div>
                `}).join('')}
            </div>

            <div class="habits-summary-grid">
                ${habits.strengths && habits.strengths.length > 0 ? `
                    <div class="summary-section strengths-section">
                        <h3>
                            <span class="section-icon">✓</span>
                            Team Strengths
                        </h3>
                        <div class="summary-list">
                            ${habits.strengths.map(s => `
                                <div class="summary-item strength">
                                    <div class="summary-item-header">
                                        <span class="summary-name">${s.name}</span>
                                        <span class="summary-score high">${s.score}/5</span>
                                    </div>
                                    ${s.interpretation ? `<p class="summary-desc">${s.interpretation}</p>` : ''}
                                </div>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}

                ${habits.growth_areas && habits.growth_areas.length > 0 ? `
                    <div class="summary-section growth-section">
                        <h3>
                            <span class="section-icon">↗</span>
                            Growth Opportunities
                        </h3>
                        <div class="summary-list">
                            ${habits.growth_areas.map(g => `
                                <div class="summary-item growth">
                                    <div class="summary-item-header">
                                        <span class="summary-name">${g.name}</span>
                                        <span class="summary-score low">${g.score}/5</span>
                                    </div>
                                    ${g.development_tip ? `
                                        <p class="summary-tip">
                                            <span class="tip-bullet"><svg class="icon icon-sm"><use href="#icon-delta"></use></svg></span> ${g.development_tip}
                                        </p>
                                    ` : ''}
                                </div>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;
    }

    tactifyDrillName(name) {
        if (!name) return name;
        if (name.toLowerCase().endsWith(' drill')) {
            return 'Protocol Override: ' + name.slice(0, -6).trim();
        }
        if (name.toLowerCase().endsWith(' exercise')) {
            return 'Tactical Exercise: ' + name.slice(0, -9).trim();
        }
        return name;
    }

    renderTraining(training) {
        const container = document.getElementById('training-content');

        if (!training) {
            container.innerHTML = '<p class="empty">No training recommendations available</p>';
            return;
        }

        container.innerHTML = `
            <div class="training-header-section">
                <h2>Training Recommendations</h2>
                <span class="training-count">${training.total_recommendations} recommendations</span>
            </div>

            ${training.immediate_actions && training.immediate_actions.length > 0 ? `
                <div class="training-section">
                    <h3>Immediate Actions</h3>
                    <div class="actions-list">
                        ${training.immediate_actions.map(action => `
                            <div class="action-card priority-${action.priority.toLowerCase()}">
                                <div class="action-priority-bar"></div>
                                <div class="action-content">
                                    <div class="action-header">
                                        <span class="priority-tag ${action.priority.toLowerCase()}">${action.priority.toUpperCase() === 'CRITICAL' ? 'MISSION CRITICAL' : action.priority}</span>
                                        <span class="action-category">${action.category}</span>
                                    </div>
                                    <h4 class="action-title">${action.title}</h4>
                                    <p class="action-desc">${action.description}</p>
                                    <div class="action-meta">
                                        ${action.scout_connection ? `
                                            <div class="meta-row">
                                                <span class="meta-label">Scout:</span>
                                                <span class="meta-value">${action.scout_connection}</span>
                                            </div>
                                        ` : ''}
                                        ${action.habit_connection ? `
                                            <div class="meta-row">
                                                <span class="meta-label">7 Habits:</span>
                                                <span class="meta-value">${action.habit_connection}</span>
                                            </div>
                                        ` : ''}
                                        ${action.success_criteria ? `
                                            <div class="meta-row success">
                                                <span class="meta-label">Success:</span>
                                                <span class="meta-value">${action.success_criteria}</span>
                                            </div>
                                        ` : ''}
                                    </div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}

            ${training.drills && training.drills.length > 0 ? `
                <div class="training-section">
                    <h3>Training Drills</h3>
                    <div class="drills-list">
                        ${training.drills.map((drill, drillIdx) => `
                            <div class="drill-card" id="drill-card-${drillIdx}">
                                <div class="drill-card-header">
                                    <span class="drill-title">${this.tactifyDrillName(drill.name)}</span>
                                    <span class="drill-time">
                                        <span class="time-icon"><svg class="icon icon-sm"><use href="#icon-crosshair"></use></svg></span>
                                        ${drill.duration}
                                    </span>
                                </div>
                                <div class="drill-body">
                                    <p class="drill-purpose">${drill.purpose}</p>
                                    <button class="btn-drill-initiate" onclick="app.initiateDrill(this, ${drillIdx})">[ INITIATE DRILL ]</button>
                                </div>
                                <div class="drill-expanded hidden" id="drill-expanded-${drillIdx}">
                                    ${drill.steps && drill.steps.length > 0 ? `
                                        <div class="drill-steps">
                                            <span class="steps-label">Steps</span>
                                            ${drill.steps.map((s, sIdx) => `
                                                <label class="drill-step-check" id="drill-step-${drillIdx}-${sIdx}">
                                                    <input type="checkbox" onchange="this.parentElement.classList.toggle('checked', this.checked)">
                                                    <span>${s}</span>
                                                </label>
                                            `).join('')}
                                        </div>
                                    ` : ''}
                                    ${drill.debrief_questions && drill.debrief_questions.length > 0 ? `
                                        <div class="drill-debrief">
                                            <span class="debrief-label">Debrief Questions</span>
                                            <ul class="debrief-list">
                                                ${drill.debrief_questions.map(q => `<li>${q}</li>`).join('')}
                                            </ul>
                                        </div>
                                    ` : ''}
                                    <button class="btn-drill-complete" onclick="app.completeDrill(this, ${drillIdx})">[ COMPLETE DRILL ]</button>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}

            ${training.discussion_topics && training.discussion_topics.length > 0 ? `
                <div class="training-section">
                    <h3>Discussion Topics</h3>
                    <div class="topics-grid">
                        ${training.discussion_topics.map(topic => `
                            <div class="topic-card">
                                <h4 class="topic-title">${topic.topic}</h4>
                                <p class="topic-question">${topic.question}</p>
                                ${topic.scout_connection ? `
                                    <div class="topic-connection">
                                        <span class="conn-label">Scout Connection:</span>
                                        <span class="conn-value">${topic.scout_connection}</span>
                                    </div>
                                ` : ''}
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}

            ${training.framework_alignment && Object.keys(training.framework_alignment).length > 0 ? `
                <div class="training-section">
                    <h3>Framework Alignment</h3>
                    <div class="alignment-grid">
                        ${Object.entries(training.framework_alignment).map(([key, values]) => `
                            <div class="alignment-card">
                                <h4 class="alignment-title">${key}</h4>
                                <ul class="alignment-list">
                                    ${values.slice(0, 3).map(v => `<li>${v}</li>`).join('')}
                                </ul>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
        `;
    }

    parseMarkdown(text) {
        // Convert markdown to HTML with proper formatting
        const lines = text.split('\n');
        let html = '';
        let inTable = false;
        let inList = false;
        let listType = null;

        for (let i = 0; i < lines.length; i++) {
            let line = lines[i];

            // Skip empty lines but close lists
            if (!line.trim()) {
                if (inList) {
                    html += listType === 'ul' ? '</ul>' : '</ol>';
                    inList = false;
                    listType = null;
                }
                if (inTable) {
                    html += '</tbody></table>';
                    inTable = false;
                }
                continue;
            }

            // Headers
            if (line.startsWith('### ')) {
                if (inList) { html += listType === 'ul' ? '</ul>' : '</ol>'; inList = false; }
                html += `<h3 class="narrative-h3">${this.formatInlineMarkdown(line.slice(4))}</h3>`;
                continue;
            }
            if (line.startsWith('## ')) {
                if (inList) { html += listType === 'ul' ? '</ul>' : '</ol>'; inList = false; }
                html += `<h2 class="narrative-h2">${this.formatInlineMarkdown(line.slice(3))}</h2>`;
                continue;
            }
            if (line.startsWith('# ')) {
                if (inList) { html += listType === 'ul' ? '</ul>' : '</ol>'; inList = false; }
                html += `<h1 class="narrative-h1">${this.formatInlineMarkdown(line.slice(2))}</h1>`;
                continue;
            }

            // Table detection (lines starting with |)
            if (line.trim().startsWith('|')) {
                // Check if this is a separator line (|---|---|)
                if (line.match(/^\|[\s\-:|]+\|$/)) {
                    continue; // Skip separator lines
                }

                const cells = line.split('|').filter(c => c.trim()).map(c => this.formatInlineMarkdown(c.trim()));

                if (!inTable) {
                    // Start new table with header
                    html += '<table class="narrative-table"><thead><tr>';
                    cells.forEach(cell => html += `<th>${cell}</th>`);
                    html += '</tr></thead><tbody>';
                    inTable = true;
                } else {
                    // Table row
                    html += '<tr>';
                    cells.forEach(cell => html += `<td>${cell}</td>`);
                    html += '</tr>';
                }
                continue;
            } else if (inTable) {
                html += '</tbody></table>';
                inTable = false;
            }

            // Checkbox lists (- [ ] or - [x])
            if (line.match(/^-\s*\[\s*[xX]?\s*\]/)) {
                if (!inList || listType !== 'ul') {
                    if (inList) html += listType === 'ul' ? '</ul>' : '</ol>';
                    html += '<ul class="narrative-checklist">';
                    inList = true;
                    listType = 'ul';
                }
                const checked = line.match(/\[\s*[xX]\s*\]/) ? 'checked' : '';
                const content = line.replace(/^-\s*\[\s*[xX]?\s*\]\s*/, '');
                html += `<li class="checklist-item ${checked}"><span class="checkbox">${checked ? '☑' : '☐'}</span> ${this.formatInlineMarkdown(content)}</li>`;
                continue;
            }

            // Unordered lists (- item)
            if (line.match(/^[-*]\s+/)) {
                if (!inList || listType !== 'ul') {
                    if (inList) html += listType === 'ul' ? '</ul>' : '</ol>';
                    html += '<ul class="narrative-list">';
                    inList = true;
                    listType = 'ul';
                }
                const content = line.replace(/^[-*]\s+/, '');
                html += `<li>${this.formatInlineMarkdown(content)}</li>`;
                continue;
            }

            // Ordered lists (1. item)
            if (line.match(/^\d+\.\s+/)) {
                if (!inList || listType !== 'ol') {
                    if (inList) html += listType === 'ul' ? '</ul>' : '</ol>';
                    html += '<ol class="narrative-list">';
                    inList = true;
                    listType = 'ol';
                }
                const content = line.replace(/^\d+\.\s+/, '');
                html += `<li>${this.formatInlineMarkdown(content)}</li>`;
                continue;
            }

            // Close list if we hit a non-list item
            if (inList) {
                html += listType === 'ul' ? '</ul>' : '</ol>';
                inList = false;
                listType = null;
            }

            // Regular paragraph
            html += `<p>${this.formatInlineMarkdown(line)}</p>`;
        }

        // Close any open elements
        if (inList) html += listType === 'ul' ? '</ul>' : '</ol>';
        if (inTable) html += '</tbody></table>';

        return html;
    }

    formatInlineMarkdown(text) {
        // Bold
        text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        // Italic
        text = text.replace(/\*([^*]+)\*/g, '<em>$1</em>');
        // Inline code
        text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
        // Quotes
        text = text.replace(/"([^"]+)"/g, '<q>$1</q>');
        return text;
    }

    renderNarrative(narrativeSummary, llmSkippedReason = null) {
        const loadingEl = document.getElementById('narrative-loading');
        const contentEl = document.getElementById('narrative-section');
        const unavailableEl = document.getElementById('narrative-unavailable');
        const narrativeContent = document.getElementById('narrative-content');
        const narrativeModel = document.getElementById('narrative-model');
        const narrativeBadge = document.getElementById('narrative-badge');

        // Hide loading state
        if (loadingEl) loadingEl.classList.add('hidden');

        if (narrativeSummary?.narrative) {
            // Show content and badge
            if (unavailableEl) unavailableEl.classList.add('hidden');
            if (contentEl) contentEl.classList.remove('hidden');
            if (narrativeBadge) narrativeBadge.classList.remove('hidden');

            // Convert markdown to HTML with proper formatting
            const html = this.parseMarkdown(narrativeSummary.narrative);

            if (narrativeContent) narrativeContent.innerHTML = html;

            // Show model and generation time
            const genTime = narrativeSummary.generation_time;
            const timeStr = genTime ? ` in ${genTime}s` : '';
            if (narrativeModel) {
                narrativeModel.textContent = `Generated by ${narrativeSummary.model || 'AI'}${timeStr}`;
            }
        } else {
            // Show unavailable state with reason if provided
            if (contentEl) contentEl.classList.add('hidden');
            if (narrativeBadge) narrativeBadge.classList.add('hidden');
            if (unavailableEl) {
                unavailableEl.classList.remove('hidden');
                // Add skip reason if available
                if (llmSkippedReason) {
                    unavailableEl.innerHTML = `
                        <div class="llm-skipped-notice">
                            <span class="notice-icon"><svg class="icon"><use href="#icon-shield"></use></svg></span>
                            <strong>AI Debrief Skipped</strong>
                            <p>${llmSkippedReason}</p>
                            <p class="notice-tip">Click "Regenerate" above to generate it now, or check other tabs for analysis metrics.</p>
                        </div>
                    `;
                }
            }
        }
    }

    renderStory(storyNarrative, llmSkippedReason = null) {
        const loadingEl = document.getElementById('story-loading');
        const contentEl = document.getElementById('story-section');
        const unavailableEl = document.getElementById('story-unavailable');
        const storyContent = document.getElementById('story-content');
        const storyModel = document.getElementById('story-model');
        const storyBadge = document.getElementById('story-badge');

        // Hide loading state
        if (loadingEl) loadingEl.classList.add('hidden');

        if (storyNarrative?.story) {
            // Show content and badge
            if (unavailableEl) unavailableEl.classList.add('hidden');
            if (contentEl) contentEl.classList.remove('hidden');
            if (storyBadge) storyBadge.classList.remove('hidden');

            // Convert markdown-style to HTML with story-specific formatting
            const story = storyNarrative.story;
            let html = story.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
            // Handle italics for quotes
            html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
            // Convert paragraphs
            html = html.split(/\n\n+/).map(p => {
                const trimmed = p.trim();
                // Check if it's a quote (starts with ")
                if (trimmed.startsWith('"') || trimmed.includes('said') || trimmed.includes('ordered')) {
                    return `<p class="story-dialogue">${trimmed}</p>`;
                }
                return `<p>${trimmed}</p>`;
            }).join('');
            html = html.replace(/\n/g, '<br>');

            if (storyContent) storyContent.innerHTML = html;

            // Show model and generation time
            const genTime = storyNarrative.generation_time;
            const timeStr = genTime ? ` in ${genTime}s` : '';
            if (storyModel) {
                storyModel.textContent = `Generated by ${storyNarrative.model || 'AI'}${timeStr}`;
            }
        } else {
            // Show unavailable state with reason if provided
            if (contentEl) contentEl.classList.add('hidden');
            if (storyBadge) storyBadge.classList.add('hidden');
            if (unavailableEl) {
                unavailableEl.classList.remove('hidden');
                // Add skip reason if available
                if (llmSkippedReason) {
                    unavailableEl.innerHTML = `
                        <div class="llm-skipped-notice">
                            <span class="notice-icon"><svg class="icon"><use href="#icon-shield"></use></svg></span>
                            <strong>Mission Story Skipped</strong>
                            <p>${llmSkippedReason}</p>
                            <p class="notice-tip">Click "Regenerate" above to generate it now, or check other tabs for analysis metrics.</p>
                        </div>
                    `;
                }
            }
        }
    }

    formatDuration(seconds) {
        if (!seconds) return '0:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    formatTime(seconds) {
        if (!seconds) return '0:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    formatPatternName(name) {
        return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    generateMarkdown(results) {
        const now = new Date().toISOString().split('T')[0];
        let md = `# Mission Analysis Report\n\n`;
        md += `**Generated:** ${now}\n\n`;
        md += `---\n\n`;

        // ========== EXECUTIVE SUMMARY ==========
        md += `## Executive Summary\n\n`;
        md += `| Metric | Value |\n`;
        md += `|--------|-------|\n`;
        md += `| Duration | ${this.formatDuration(results.duration_seconds)} |\n`;
        md += `| Segments | ${results.transcription?.length || 0} |\n`;
        md += `| Speakers | ${results.speakers?.length || 0} |\n`;

        const effectivePct = results.communication_quality?.effective_percentage || 0;
        md += `| Effective Communication | ${effectivePct.toFixed(0)}% |\n`;

        const habitsScore = results.seven_habits?.overall_score || 0;
        md += `| 7 Habits Score | ${habitsScore.toFixed(1)}/5 |\n`;

        const avgConfidence = results.confidence_distribution?.average_confidence || 0;
        md += `| Transcription Confidence | ${(avgConfidence * 100).toFixed(0)}% |\n`;

        md += `| Processing Time | ${results.processing_time_seconds?.toFixed(1) || 0}s |\n`;
        md += `\n`;

        // ========== LOW CONFIDENCE WARNING ==========
        if (avgConfidence < 0.40) {
            md += `> ⚠️ **Low Transcription Confidence Warning**\n>\n`;
            md += `> Transcription confidence is ${(avgConfidence * 100).toFixed(0)}%, which is below the 40% reliability threshold.\n`;
            md += `> Assessment accuracy may be significantly affected. Consider:\n`;
            md += `> - Re-recording with better audio quality\n`;
            md += `> - Using a closer microphone placement\n`;
            md += `> - Reducing background noise\n>\n`;
            md += `> Detailed assessments below should be interpreted with caution.\n\n`;
        } else if (avgConfidence < 0.60) {
            md += `> ℹ️ **Moderate Transcription Confidence**\n>\n`;
            md += `> Transcription confidence is ${(avgConfidence * 100).toFixed(0)}%. Some assessments may be affected by transcription errors.\n\n`;
        }

        // ========== AI MISSION DEBRIEF ==========
        if (results.narrative_summary?.narrative) {
            md += `## AI Mission Debrief\n\n`;
            md += `*Generated by ${results.narrative_summary.model || 'AI'}`;
            if (results.narrative_summary.generation_time) {
                md += ` in ${results.narrative_summary.generation_time}s`;
            }
            md += `*\n\n`;
            md += `${results.narrative_summary.narrative}\n\n`;
        }

        // ========== AI MISSION STORY ==========
        if (results.story_narrative?.story) {
            md += `## Mission Story\n\n`;
            md += `*Generated by ${results.story_narrative.model || 'AI'}`;
            if (results.story_narrative.generation_time) {
                md += ` in ${results.story_narrative.generation_time}s`;
            }
            md += `*\n\n`;
            md += `${results.story_narrative.story}\n\n`;
        }

        // ========== CREW & ROLES ==========
        if (results.speakers && results.speakers.length > 0) {
            md += `## Crew & Role Assignments\n\n`;

            // Build role map with confidence details
            const roleMap = {};
            if (results.role_assignments) {
                for (const ra of results.role_assignments) {
                    roleMap[ra.speaker_id] = ra;
                }
            }

            md += `| Speaker | Role | Confidence | Voice | Telemetry | Speaking Time | Utterances |\n`;
            md += `|---------|------|------------|-------|-----------|---------------|------------|\n`;

            for (const s of results.speakers) {
                const ra = roleMap[s.speaker_id];
                const role = ra?.role || s.role || '-';
                const confidence = ra?.confidence ? `${(ra.confidence * 100).toFixed(0)}%` : '-';
                const voiceConf = ra?.voice_confidence ? `${(ra.voice_confidence * 100).toFixed(0)}%` : '-';
                const telemetryConf = ra?.telemetry_confidence ? `${(ra.telemetry_confidence * 100).toFixed(0)}%` : '-';
                const displayName = this.getSpeakerDisplayName(s.speaker_id);
                md += `| ${displayName} | ${role} | ${confidence} | ${voiceConf} | ${telemetryConf} | ${this.formatDuration(s.total_speaking_time)} | ${s.utterance_count} |\n`;
            }
            md += `\n`;

            // Role methodology notes
            const methodologies = results.role_assignments?.filter(ra => ra.methodology_note);
            if (methodologies?.length > 0) {
                md += `**Role Detection Notes:**\n\n`;
                for (const ra of methodologies) {
                    md += `- **${this.getSpeakerDisplayName(ra.speaker_id)}:** ${ra.methodology_note}\n`;
                }
                md += `\n`;
            }
        }

        // ========== 7 HABITS ASSESSMENT ==========
        if (results.seven_habits) {
            const habits = results.seven_habits;
            md += `## 7 Habits of Highly Effective People\n\n`;
            md += `**Overall Score: ${habits.overall_score.toFixed(1)}/5**\n\n`;

            if (habits.habits && habits.habits.length > 0) {
                md += `| # | Habit | Score | Assessment |\n`;
                md += `|---|-------|-------|------------|\n`;

                const getLabel = (score) => {
                    if (score >= 4) return 'Excellent';
                    if (score >= 3) return 'Good';
                    if (score >= 2) return 'Developing';
                    return 'Needs Focus';
                };

                for (const h of habits.habits) {
                    const habitName = h.youth_friendly_name || h.habit_name || 'Unknown';
                    md += `| ${h.habit_number || ''} | ${habitName} | ${h.score.toFixed(1)}/5 | ${getLabel(h.score)} |\n`;
                }
                md += `\n`;

                // Detailed habit breakdown
                for (const h of habits.habits) {
                    const habitName = h.youth_friendly_name || h.habit_name || 'Unknown';
                    md += `### Habit ${h.habit_number}: ${habitName}\n\n`;
                    md += `**Score:** ${h.score.toFixed(1)}/5\n\n`;

                    if (h.interpretation) {
                        md += `${h.interpretation}\n\n`;
                    }

                    if (h.examples && h.examples.length > 0) {
                        md += `**Evidence:**\n`;
                        for (const ex of h.examples.slice(0, 3)) {
                            const speaker = ex.speaker ? `[${this.getSpeakerDisplayName(ex.speaker)}]` : '';
                            md += `- ${speaker} "${ex.text}"\n`;
                        }
                        md += `\n`;
                    }

                    if (h.gap_to_next_score) {
                        md += `**Gap:** ${h.gap_to_next_score}\n\n`;
                    }

                    if (h.development_tip) {
                        md += `**Tip:** ${h.development_tip}\n\n`;
                    }
                }
            }

            // Strengths & Growth Areas
            if (habits.strengths?.length > 0) {
                md += `### Strengths\n\n`;
                for (const s of habits.strengths) {
                    md += `- **${s.name}** (${s.score.toFixed(1)}/5): ${s.interpretation || ''}\n`;
                }
                md += `\n`;
            }

            if (habits.growth_areas?.length > 0) {
                md += `### Growth Areas\n\n`;
                for (const g of habits.growth_areas) {
                    md += `- **${g.name}** (${g.score.toFixed(1)}/5): ${g.development_tip || ''}\n`;
                }
                md += `\n`;
            }
        }

        // ========== COMMUNICATION QUALITY ==========
        if (results.communication_quality) {
            const quality = results.communication_quality;
            md += `## Communication Quality\n\n`;
            md += `| Metric | Value |\n`;
            md += `|--------|-------|\n`;
            md += `| Effective Rate | ${quality.effective_percentage?.toFixed(0) || 0}% |\n`;
            md += `| Effective Communications | ${quality.effective_count || 0} |\n`;
            md += `| Needs Improvement | ${quality.improvement_count || 0} |\n`;
            md += `| Total Assessed | ${quality.total_utterances_assessed || (quality.effective_count + quality.improvement_count)} |\n`;
            md += `\n`;

            // Patterns
            const effectivePatterns = (quality.patterns || []).filter(p => p.category === 'effective');
            const improvementPatterns = (quality.patterns || []).filter(p => p.category === 'needs_improvement');

            if (effectivePatterns.length > 0) {
                md += `### Effective Patterns\n\n`;
                for (const p of effectivePatterns) {
                    md += `- **${p.name}** (${p.count} instances): ${p.description || ''}\n`;
                    if (p.examples?.length > 0) {
                        for (const ex of p.examples.slice(0, 2)) {
                            md += `  - "${ex.text?.substring(0, 100)}${ex.text?.length > 100 ? '...' : ''}"\n`;
                        }
                    }
                }
                md += `\n`;
            }

            if (improvementPatterns.length > 0) {
                md += `### Areas for Improvement\n\n`;
                for (const p of improvementPatterns) {
                    md += `- **${p.name}** (${p.count} instances): ${p.description || ''}\n`;
                    if (p.examples?.length > 0) {
                        for (const ex of p.examples.slice(0, 2)) {
                            md += `  - "${ex.text?.substring(0, 100)}${ex.text?.length > 100 ? '...' : ''}"\n`;
                        }
                    }
                }
                md += `\n`;
            }
        }

        // ========== SPEAKER SCORECARDS ==========
        if (results.speaker_scorecards && results.speaker_scorecards.length > 0) {
            md += `## Individual Speaker Scorecards\n\n`;

            for (const card of results.speaker_scorecards) {
                const displayName = this.getSpeakerDisplayName(card.speaker_id);
                md += `### ${displayName}`;
                if (card.role) md += ` (${card.role})`;
                md += `\n\n`;

                md += `**Overall Score: ${card.overall_score.toFixed(1)}/5**\n\n`;

                if (card.metrics && card.metrics.length > 0) {
                    md += `| Metric | Score | Evidence |\n`;
                    md += `|--------|-------|----------|\n`;
                    for (const m of card.metrics) {
                        const evidence = m.evidence?.substring(0, 60) || '';
                        md += `| ${m.name} | ${m.score}/5 | ${evidence}${m.evidence?.length > 60 ? '...' : ''} |\n`;
                    }
                    md += `\n`;
                }

                if (card.strengths?.length > 0) {
                    md += `**Strengths:** ${card.strengths.join(', ')}\n\n`;
                }

                if (card.areas_for_improvement?.length > 0) {
                    md += `**Areas for Improvement:** ${card.areas_for_improvement.join(', ')}\n\n`;
                }
            }
        }

        // ========== TRAINING RECOMMENDATIONS ==========
        if (results.training_recommendations) {
            const training = results.training_recommendations;
            md += `## Training Recommendations\n\n`;
            md += `*${training.total_recommendations || 0} recommendations identified*\n\n`;

            // Immediate Actions
            if (training.immediate_actions?.length > 0) {
                md += `### Immediate Actions\n\n`;
                for (const action of training.immediate_actions) {
                    md += `#### ${action.priority}: ${action.title}\n\n`;
                    md += `**Category:** ${action.category}\n\n`;
                    md += `${action.description}\n\n`;
                    if (action.scout_connection) {
                        md += `- **Scout Connection:** ${action.scout_connection}\n`;
                    }
                    if (action.habit_connection) {
                        md += `- **7 Habits Connection:** ${action.habit_connection}\n`;
                    }
                    if (action.success_criteria) {
                        md += `- **Success Criteria:** ${action.success_criteria}\n`;
                    }
                    md += `\n`;
                }
            }

            // Training Drills
            if (training.drills?.length > 0) {
                md += `### Training Drills\n\n`;
                for (const drill of training.drills) {
                    md += `#### ${drill.name}`;
                    if (drill.duration) md += ` (${drill.duration})`;
                    md += `\n\n`;

                    if (drill.purpose) {
                        md += `**Purpose:** ${drill.purpose}\n\n`;
                    }

                    if (drill.steps?.length > 0) {
                        md += `**Steps:**\n`;
                        drill.steps.forEach((step, i) => {
                            md += `${i + 1}. ${step}\n`;
                        });
                        md += `\n`;
                    }

                    if (drill.debrief_questions?.length > 0) {
                        md += `**Debrief Questions:**\n`;
                        for (const q of drill.debrief_questions) {
                            md += `- ${q}\n`;
                        }
                        md += `\n`;
                    }
                }
            }

            // Discussion Topics
            if (training.discussion_topics?.length > 0) {
                md += `### Discussion Topics\n\n`;
                for (const topic of training.discussion_topics) {
                    md += `- **${topic.title}:** ${topic.opening_question || ''}\n`;
                    if (topic.connection) {
                        md += `  - Connection: ${topic.connection}\n`;
                    }
                }
                md += `\n`;
            }
        }

        // ========== LEARNING EVALUATION ==========
        if (results.learning_evaluation) {
            const learning = results.learning_evaluation;
            md += `## Learning Evaluation\n\n`;

            // Kirkpatrick Levels
            if (learning.kirkpatrick_levels?.length > 0) {
                md += `### Kirkpatrick 4-Level Model\n\n`;
                md += `| Level | Name | Score |\n`;
                md += `|-------|------|-------|\n`;
                for (const lvl of learning.kirkpatrick_levels) {
                    md += `| ${lvl.level} | ${lvl.name} | ${(lvl.score * 100).toFixed(0)}% |\n`;
                }
                md += `\n`;
            }

            // Frameworks
            md += `### Learning Frameworks\n\n`;
            md += `| Framework | Value |\n`;
            md += `|-----------|-------|\n`;
            if (learning.blooms_level) {
                md += `| Bloom's Taxonomy | ${learning.blooms_level} |\n`;
            }
            if (learning.nasa_tlx_score !== undefined) {
                md += `| NASA TLX Workload | ${(learning.nasa_tlx_score * 100).toFixed(0)}% |\n`;
            }
            if (learning.engagement_score !== undefined) {
                md += `| Engagement Score | ${(learning.engagement_score * 100).toFixed(0)}% |\n`;
            }
            md += `\n`;

            // Top Communications
            if (learning.top_communications?.length > 0) {
                md += `### High-Value Communications\n\n`;
                for (const comm of learning.top_communications.slice(0, 5)) {
                    const speaker = comm.speaker_id ? `[${this.getSpeakerDisplayName(comm.speaker_id)}]` : '';
                    const confidence = comm.learning_confidence ? ` (${(comm.learning_confidence * 100).toFixed(0)}% learning value)` : '';
                    md += `- ${speaker} "${comm.text}"${confidence}\n`;
                }
                md += `\n`;
            }

            // Learning Recommendations
            if (learning.recommendations?.length > 0) {
                md += `### Learning Recommendations\n\n`;
                for (const rec of learning.recommendations) {
                    md += `- ${rec}\n`;
                }
                md += `\n`;
            }
        }

        // ========== CONFIDENCE ANALYSIS ==========
        if (results.confidence_distribution) {
            const conf = results.confidence_distribution;
            md += `## Transcription Confidence\n\n`;
            md += `| Metric | Value |\n`;
            md += `|--------|-------|\n`;
            md += `| Average Confidence | ${(conf.average_confidence * 100).toFixed(0)}% |\n`;
            const totalSegments = results.transcription?.length || conf.total_segments || 0;
            md += `| Total Segments | ${totalSegments} |\n`;

            if (conf.high_confidence_count !== undefined) {
                md += `| High Confidence (>80%) | ${conf.high_confidence_count} |\n`;
            }
            if (conf.medium_confidence_count !== undefined) {
                md += `| Medium Confidence (50-80%) | ${conf.medium_confidence_count} |\n`;
            }
            if (conf.low_confidence_count !== undefined) {
                md += `| Low Confidence (<50%) | ${conf.low_confidence_count} |\n`;
            }
            md += `\n`;

            if (conf.assessment) {
                md += `**Assessment:** ${conf.assessment}\n\n`;
            }
        }

        // ========== TRANSCRIPT ==========
        md += `## Full Transcript\n\n`;
        md += `> The percentage shown after each speaker indicates **transcription accuracy confidence** - how certain the speech-to-text model is about the transcription. Higher values (70%+) indicate clear audio; lower values (<40%) may contain errors.\n\n`;
        if (results.transcription && results.transcription.length > 0) {
            for (const seg of results.transcription) {
                const speaker = this.getSpeakerDisplayName(seg.speaker_id) || 'Speaker';
                const time = this.formatTime(seg.start_time);
                const confPct = seg.confidence ? (seg.confidence * 100).toFixed(0) : null;
                const confidence = confPct ? ` [${confPct}% accuracy]` : '';
                md += `**[${time}] ${speaker}${confidence}:** ${seg.text}\n\n`;
            }
        } else {
            md += `*No transcription available*\n\n`;
        }

        // ========== FULL TEXT ==========
        md += `## Full Text (Continuous)\n\n`;
        md += `${results.full_text || '*No text available*'}\n\n`;

        // ========== FOOTER ==========
        md += `---\n\n`;
        md += `*Generated by Bridge Crew Training AI Analyzer*\n`;
        md += `*Analysis includes: Speaker diarization, Role inference, 7 Habits assessment, Communication quality, Learning evaluation, and Training recommendations*\n`;

        return md;
    }

    // =========================================================================
    // LCARS Visualizations
    // =========================================================================

    destroyCharts() {
        for (const chart of this.chartInstances) {
            try { chart.destroy(); } catch (e) {}
        }
        this.chartInstances = [];
    }

    renderBridgeMap(roleAssignments, speakers) {
        const stations = document.querySelectorAll('.bridge-station');
        if (!stations.length) return;

        // Build role-to-speaker map
        const roleMap = {};
        if (roleAssignments) {
            for (const ra of roleAssignments) {
                roleMap[ra.role] = ra;
            }
        }

        // Calculate total speaking time for all speakers
        const speakerDataMap = {};
        if (speakers) {
            for (const s of speakers) {
                speakerDataMap[s.speaker_id] = s;
            }
        }

        stations.forEach(station => {
            const role = station.dataset.role;
            const crewEl = station.querySelector('.station-crew');
            const confEl = station.querySelector('.station-conf');
            const ra = roleMap[role];

            if (ra) {
                station.classList.add('station-glow', 'assigned');
                station.classList.remove('unassigned');
                const displayName = this.getSpeakerDisplayName(ra.speaker_id);
                if (crewEl) crewEl.textContent = displayName;
                if (confEl) {
                    const pct = (ra.confidence * 100).toFixed(0);
                    confEl.textContent = `${pct}%`;
                    confEl.className = 'station-conf ' + (ra.confidence >= 0.7 ? 'high' : ra.confidence >= 0.4 ? 'medium' : 'low');
                }

                // Click to switch to Crew Stations tab
                station.style.cursor = 'pointer';
                station.onclick = () => {
                    if (window.app) {
                        window.app.switchTab('crew-stations');
                    }
                };
            } else {
                station.classList.remove('station-glow', 'assigned');
                station.classList.add('unassigned');
                if (crewEl) crewEl.innerHTML = '<span class="station-offline">Station Offline</span>';
                if (confEl) {
                    confEl.textContent = '';
                    confEl.className = 'station-conf';
                }
                station.style.cursor = 'default';
                station.onclick = null;
            }
        });
    }

    renderTimeline(segments, durationSeconds, sentimentSummary) {
        const container = document.getElementById('mission-timeline');
        if (!container || !segments || segments.length === 0 || !durationSeconds) {
            if (container) container.innerHTML = '';
            return;
        }

        // Assign colors to speakers
        const speakerColors = [
            'var(--lcars-gold)', 'var(--lcars-blue)', 'var(--lcars-purple)',
            'var(--lcars-lilac)', 'var(--lcars-orange)', 'var(--lcars-tan)'
        ];
        const speakerIds = [...new Set(segments.map(s => s.speaker_id))];
        const colorMap = {};
        speakerIds.forEach((id, i) => {
            colorMap[id] = speakerColors[i % speakerColors.length];
        });

        // Build timeline blocks with confidence-based opacity
        const blocks = segments.map((seg, idx) => {
            const left = (seg.start_time / durationSeconds) * 100;
            const width = Math.max(0.5, ((seg.end_time - seg.start_time) / durationSeconds) * 100);
            const color = colorMap[seg.speaker_id] || 'var(--lcars-gold)';
            const conf = seg.confidence || 0.5;
            const opacity = 0.3 + (conf * 0.7);
            // Backend uses 0-1 scale, convert to percentage for display
            const stressVal = seg.sentiment?.stress_level != null ? Math.round(seg.sentiment.stress_level * 100) : '';
            const emotionVal = seg.sentiment?.emotion || '';
            const stressLabel = stressVal !== '' ? (stressVal < 40 ? 'low' : stressVal < 70 ? 'moderate' : 'high') : '';
            return `<div class="timeline-block" data-index="${idx}"
                data-speaker="${this.escapeHtml(this.getSpeakerDisplayName(seg.speaker_id, true))}"
                data-text="${this.escapeHtml(seg.text.substring(0, 120))}"
                data-start="${seg.start_time}"
                data-end="${seg.end_time}"
                data-confidence="${conf}"
                data-stress="${stressVal}"
                data-emotion="${emotionVal}"
                data-stress-label="${stressLabel}"
                style="left:${left}%;width:${width}%;background:${color};opacity:${opacity.toFixed(2)}"></div>`;
        }).join('');

        // Build stress track blocks (if sentiment data exists)
        const hasSentiment = segments.some(s => s.sentiment?.stress_level != null);
        const stressBlocks = hasSentiment ? segments.map(seg => {
            if (seg.sentiment?.stress_level == null) return '';
            const left = (seg.start_time / durationSeconds) * 100;
            const width = Math.max(0.5, ((seg.end_time - seg.start_time) / durationSeconds) * 100);
            const stress = seg.sentiment.stress_level * 100; // Convert 0-1 to percentage
            const stressColor = stress < 40 ? 'var(--stress-low)' : stress < 70 ? 'var(--stress-moderate)' : 'var(--stress-high)';
            return `<div class="timeline-stress-block" style="left:${left}%;width:${width}%;background:${stressColor}"></div>`;
        }).join('') : '';

        // Time markers
        const markers = [0, 25, 50, 75, 100].map(pct => {
            const time = this.formatTime((pct / 100) * durationSeconds);
            return `<span class="timeline-marker" style="left:${pct}%">${time}</span>`;
        }).join('');

        // Speaker legend
        const legend = speakerIds.map(id => {
            const color = colorMap[id];
            const name = this.getSpeakerDisplayName(id);
            return `<span class="timeline-legend-item">
                <span class="timeline-legend-swatch" style="background:${color}"></span>
                ${name}
            </span>`;
        }).join('');

        container.innerHTML = `
            <div class="timeline-label">Mission Timeline</div>
            <div class="timeline-bar">
                <div class="timeline-bar-inner">
                    ${blocks}
                </div>
                <div class="timeline-tooltip" id="timeline-tooltip"></div>
            </div>
            ${hasSentiment ? `
                <div class="timeline-stress-label">Stress Level</div>
                <div class="timeline-stress-bar">
                    <div class="timeline-stress-bar-inner">
                        ${stressBlocks}
                    </div>
                </div>
            ` : ''}
            <div class="timeline-markers">${markers}</div>
            <div class="timeline-legend">${legend}</div>
        `;

        // Tooltip handler
        const tooltip = container.querySelector('#timeline-tooltip');
        container.querySelectorAll('.timeline-block').forEach(block => {
            block.addEventListener('mouseenter', (e) => {
                const speaker = block.dataset.speaker;
                const text = block.dataset.text;
                const startTime = parseFloat(block.dataset.start);
                const endTime = parseFloat(block.dataset.end);
                const conf = parseFloat(block.dataset.confidence);
                const confPct = (conf * 100).toFixed(0);
                const confClass = conf >= 0.7 ? 'conf-high' : conf >= 0.4 ? 'conf-medium' : 'conf-low';

                const stress = block.dataset.stress;
                const emotion = block.dataset.emotion;
                const stressLabel = block.dataset.stressLabel;

                tooltip.innerHTML = `
                    <div class="tooltip-speaker">${speaker}</div>
                    <div class="tooltip-text">${text}</div>
                    <div class="tooltip-meta">
                        <span class="tooltip-time">${this.formatTime(startTime)} - ${this.formatTime(endTime)}</span>
                        <span class="tooltip-confidence ${confClass}">${confPct}%</span>
                    </div>
                    ${stress ? `
                        <div class="tooltip-sentiment">
                            <span class="tooltip-stress stress-${stressLabel}">Stress: ${Math.round(stress)}%</span>
                            ${emotion ? `<span class="tooltip-emotion">${emotion}</span>` : ''}
                        </div>
                    ` : ''}
                `;

                // Position tooltip centered on block
                const barRect = block.parentElement.getBoundingClientRect();
                const blockRect = block.getBoundingClientRect();
                const blockCenter = blockRect.left - barRect.left + blockRect.width / 2;
                tooltip.style.left = blockCenter + 'px';
                tooltip.classList.add('visible');
            });

            block.addEventListener('mouseleave', () => {
                tooltip.classList.remove('visible');
            });

            // Click handler: highlight transcript chat bubble + click-to-play audio
            block.addEventListener('click', () => {
                const idx = parseInt(block.dataset.index);
                const startTime = parseFloat(block.dataset.start);

                // Highlight chat bubble in transcript
                const transcriptList = document.getElementById('transcript-content');
                if (transcriptList) {
                    const bubbles = transcriptList.querySelectorAll('.chat-bubble');
                    bubbles.forEach(s => s.classList.remove('highlighted'));
                    if (bubbles[idx]) {
                        bubbles[idx].classList.add('highlighted');
                        if (window.app) {
                            window.app.switchTab('mission-log');
                        }
                        bubbles[idx].scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                }

                // Click-to-play audio
                const audioPlayer = document.getElementById('audio-player');
                if (audioPlayer && audioPlayer.src) {
                    audioPlayer.currentTime = startTime;
                    audioPlayer.play();
                    const audioPreview = document.getElementById('audio-preview');
                    if (audioPreview) audioPreview.classList.remove('hidden');
                }
            });
        });
    }

    renderWaveform(waveformData, durationSeconds) {
        // Remove any existing waveform
        const existing = document.querySelector('.waveform-canvas-wrapper');
        if (existing) existing.remove();

        if (!waveformData?.amplitude || waveformData.amplitude.length === 0) return;

        const container = document.getElementById('mission-timeline');
        if (!container) return;

        // Insert waveform after the stress bar (or timeline bar) and before time markers
        const markersEl = container.querySelector('.timeline-markers');
        if (!markersEl) return;

        const wrapper = document.createElement('div');
        wrapper.className = 'waveform-canvas-wrapper';
        wrapper.innerHTML = `
            <div class="waveform-label">Audio Amplitude</div>
            <canvas class="waveform-canvas"></canvas>
        `;
        container.insertBefore(wrapper, markersEl);

        const canvas = wrapper.querySelector('.waveform-canvas');
        this.drawWaveformCanvas(canvas, waveformData.amplitude);

        // Keep canvas width aligned with timeline bar via ResizeObserver
        const timelineBar = container.querySelector('.timeline-bar');
        if (timelineBar && typeof ResizeObserver !== 'undefined') {
            const ro = new ResizeObserver(() => {
                const barWidth = timelineBar.clientWidth;
                if (canvas.width !== barWidth) {
                    canvas.width = barWidth;
                    this.drawWaveformCanvas(canvas, waveformData.amplitude);
                }
            });
            ro.observe(timelineBar);
        }
    }

    drawWaveformCanvas(canvas, amplitudeData) {
        if (!canvas || !amplitudeData || amplitudeData.length === 0) return;

        const ctx = canvas.getContext('2d');
        const w = canvas.width || canvas.clientWidth;
        const h = canvas.height || canvas.clientHeight;
        canvas.width = w;
        canvas.height = h;

        ctx.clearRect(0, 0, w, h);

        // Draw filled area chart
        const len = amplitudeData.length;
        const step = w / (len - 1 || 1);

        ctx.beginPath();
        ctx.moveTo(0, h);
        for (let i = 0; i < len; i++) {
            const x = i * step;
            const val = Math.min(1, Math.max(0, amplitudeData[i]));
            const y = h - val * h;
            if (i === 0) {
                ctx.lineTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.lineTo(w, h);
        ctx.closePath();

        // Fill with LCARS blue gradient
        const gradient = ctx.createLinearGradient(0, 0, 0, h);
        gradient.addColorStop(0, 'rgba(153, 204, 255, 0.6)');
        gradient.addColorStop(1, 'rgba(153, 204, 255, 0.05)');
        ctx.fillStyle = gradient;
        ctx.fill();

        // Stroke the top edge
        ctx.beginPath();
        ctx.moveTo(0, h);
        for (let i = 0; i < len; i++) {
            const x = i * step;
            const val = Math.min(1, Math.max(0, amplitudeData[i]));
            const y = h - val * h;
            ctx.lineTo(x, y);
        }
        ctx.strokeStyle = 'rgba(153, 204, 255, 0.8)';
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    renderRadarCharts(scorecards) {
        if (!scorecards || scorecards.length === 0) return;
        if (typeof Chart === 'undefined') return;

        scorecards.forEach((card, idx) => {
            // Find or create canvas in the scorecard
            const scorecardEls = document.querySelectorAll('#scorecards-content .scorecard');
            if (!scorecardEls[idx]) return;

            // Check if canvas already exists
            let canvas = scorecardEls[idx].querySelector('.radar-canvas');
            if (!canvas) {
                const wrapper = document.createElement('div');
                wrapper.className = 'radar-chart-wrapper';
                canvas = document.createElement('canvas');
                canvas.className = 'radar-canvas';
                canvas.width = 280;
                canvas.height = 280;
                wrapper.appendChild(canvas);
                // Insert after scorecard-header
                const header = scorecardEls[idx].querySelector('.scorecard-header');
                if (header && header.nextSibling) {
                    scorecardEls[idx].insertBefore(wrapper, header.nextSibling);
                } else {
                    scorecardEls[idx].appendChild(wrapper);
                }
            }

            const labels = card.metrics.map(m => m.name.length > 15 ? m.name.substring(0, 14) + '...' : m.name);
            const data = card.metrics.map(m => m.score);

            const chart = new Chart(canvas, {
                type: 'radar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: this.getSpeakerDisplayName(card.speaker_id),
                        data: data,
                        backgroundColor: 'rgba(153, 204, 255, 0.2)',
                        borderColor: 'rgba(153, 204, 255, 0.8)',
                        pointBackgroundColor: '#FFCC99',
                        pointBorderColor: '#FFCC99',
                        pointRadius: 4,
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    scales: {
                        r: {
                            min: 0,
                            max: 5,
                            ticks: {
                                stepSize: 1,
                                color: 'rgba(240, 232, 216, 0.5)',
                                backdropColor: 'transparent',
                                font: { size: 10 }
                            },
                            grid: {
                                color: 'rgba(153, 204, 255, 0.15)'
                            },
                            angleLines: {
                                color: 'rgba(153, 204, 255, 0.15)'
                            },
                            pointLabels: {
                                color: 'rgba(240, 232, 216, 0.8)',
                                font: { size: 11, family: 'Inter' }
                            }
                        }
                    },
                    plugins: {
                        legend: { display: false }
                    }
                }
            });

            this.chartInstances.push(chart);
        });
    }

    renderStressTimeline(sentimentSummary) {
        const section = document.getElementById('stress-timeline-chart');
        const canvas = document.getElementById('stress-timeline-canvas');
        if (!section || !canvas) return;

        const timeline = sentimentSummary?.stress_timeline;
        if (!timeline || timeline.length === 0) {
            section.classList.add('hidden');
            return;
        }

        section.classList.remove('hidden');

        if (typeof Chart === 'undefined') return;

        const labels = timeline.map(p => {
            const totalSec = p.time || 0;
            const min = Math.floor(totalSec / 60);
            const sec = Math.floor(totalSec % 60);
            return `${String(min).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
        });
        const data = timeline.map(p => (p.stress || 0) * 100); // Convert 0-1 to percentage

        const ctx = canvas.getContext('2d');

        // Gradient fill: green at bottom, orange in middle, red at top
        const gradient = ctx.createLinearGradient(0, 0, 0, canvas.parentElement.clientHeight || 200);
        gradient.addColorStop(0, 'rgba(204, 102, 102, 0.4)');
        gradient.addColorStop(0.5, 'rgba(255, 153, 51, 0.25)');
        gradient.addColorStop(1, 'rgba(102, 204, 153, 0.1)');

        const chart = new Chart(canvas, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Stress Level',
                    data: data,
                    fill: true,
                    backgroundColor: gradient,
                    borderColor: 'rgba(255, 204, 153, 0.8)',
                    borderWidth: 2,
                    pointBackgroundColor: 'rgba(255, 204, 153, 1)',
                    pointBorderColor: 'rgba(255, 204, 153, 1)',
                    pointRadius: 2,
                    pointHoverRadius: 5,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        min: 0,
                        max: 100,
                        ticks: {
                            stepSize: 25,
                            color: 'rgba(240, 232, 216, 0.5)',
                            font: { family: 'JetBrains Mono', size: 10 },
                            callback: (val) => val + '%'
                        },
                        grid: {
                            color: 'rgba(153, 204, 255, 0.1)'
                        },
                        title: {
                            display: true,
                            text: 'STRESS',
                            color: 'rgba(240, 232, 216, 0.5)',
                            font: { family: 'Antonio', size: 11, weight: 600 }
                        }
                    },
                    x: {
                        ticks: {
                            color: 'rgba(240, 232, 216, 0.5)',
                            font: { family: 'JetBrains Mono', size: 10 },
                            maxRotation: 0,
                            autoSkip: true,
                            maxTicksLimit: 10
                        },
                        grid: {
                            color: 'rgba(153, 204, 255, 0.05)'
                        },
                        title: {
                            display: true,
                            text: 'MISSION TIME',
                            color: 'rgba(240, 232, 216, 0.5)',
                            font: { family: 'Antonio', size: 11, weight: 600 }
                        }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(13, 13, 36, 0.95)',
                        titleColor: '#FFCC99',
                        bodyColor: '#F0E8D8',
                        borderColor: 'rgba(255, 204, 153, 0.3)',
                        borderWidth: 1,
                        titleFont: { family: 'Antonio', size: 12 },
                        bodyFont: { family: 'JetBrains Mono', size: 11 },
                        callbacks: {
                            label: (context) => `Stress: ${Math.round(context.parsed.y)}%`
                        }
                    }
                }
            }
        });

        this.chartInstances.push(chart);
    }

    animateValue(element, start, end, duration, suffix = '') {
        if (!element) return;
        const range = end - start;
        const startTime = performance.now();

        const step = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            // Ease-out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = start + range * eased;

            if (Number.isInteger(end)) {
                element.textContent = Math.round(current) + suffix;
            } else {
                element.textContent = current.toFixed(1) + suffix;
            }

            if (progress < 1) {
                requestAnimationFrame(step);
            }
        };

        requestAnimationFrame(step);
    }
}


// ============================================================================
// Main Application
// ============================================================================

class App {
    constructor() {
        this.recorder = new AudioRecorder();
        this.api = new ApiClient();
        this.renderer = new ResultsRenderer(document.getElementById('results-section'));
        this.currentBlob = null;
        this.currentResults = null;
        this.savedRecordingPath = null;
        this.currentFilename = null;
        this.currentMetadata = null;
        this.archiveData = [];
        this.collapsedSections = new Set();
        this.audioDisabled = false; // Server config for disabling audio uploads/downloads
        this.readOnlyMode = false; // Server config for read-only mode (no analysis/regeneration)
        this.horizonsAvailable = false; // Whether Horizons game server is connected
        this.telemetrySessionId = null; // Active telemetry recording session

        this.initElements();
        this.bindEvents();
        this.loadConfig(); // Load server config first
        this.loadArchive();
        this.checkServicesStatus(); // Check service status on load

        // Start with results section collapsed if no results
        this.collapsedSections.add('results-section');
        this.updateSectionState('results-section');
    }

    async loadConfig() {
        try {
            const response = await fetch('/api/config');
            if (response.ok) {
                const config = await response.json();
                this.audioDisabled = config.audio_disabled || false;
                this.readOnlyMode = config.read_only || false;

                this.applyAudioDisabledState();
                this.applyReadOnlyState();

                // Re-render archive list to apply config states
                if (this.archiveData.length > 0) {
                    this.renderArchiveList(this.archiveData);
                }
            }
        } catch (error) {
            console.error('Failed to load config:', error);
        }
    }

    applyAudioDisabledState() {
        if (!this.audioDisabled) return;

        // Hide the input row with record/upload buttons
        const inputRow = document.querySelector('#input-section .input-row');
        if (inputRow) {
            inputRow.classList.add('hidden');
        }

        // Hide the audio preview section
        const audioPreview = document.getElementById('audio-preview');
        if (audioPreview) {
            audioPreview.classList.add('hidden');
        }

        // Show audio disabled notice in the card content
        const cardContent = document.querySelector('#input-section .card-content');
        if (cardContent) {
            const notice = document.createElement('div');
            notice.className = 'audio-disabled-notice';
            notice.innerHTML = `
                <span class="notice-icon"><svg class="icon"><use href="#icon-shield"></use></svg></span>
                <div class="notice-text">
                    <strong>Audio Upload Disabled</strong>
                    <p>Audio recording and upload are disabled on this server. You can still view existing analyses from the archive below.</p>
                </div>
            `;
            cardContent.insertBefore(notice, cardContent.firstChild);
        }

        // Hide the download audio button in results
        if (this.downloadAudioBtn) {
            this.downloadAudioBtn.classList.add('hidden');
        }
        if (this.saveAudioBtn) {
            this.saveAudioBtn.classList.add('hidden');
        }
    }

    applyReadOnlyState() {
        if (!this.readOnlyMode) return;

        // Hide the analyze button
        if (this.analyzeBtn) {
            this.analyzeBtn.classList.add('hidden');
        }

        // Hide the analysis options (LLM checkboxes)
        const analysisOptions = document.querySelector('.analysis-options');
        if (analysisOptions) {
            analysisOptions.classList.add('hidden');
        }

        // Hide regenerate buttons
        const regenerateNarrativeBtn = document.getElementById('regenerate-narrative-btn');
        const regenerateStoryBtn = document.getElementById('regenerate-story-btn');
        if (regenerateNarrativeBtn) {
            regenerateNarrativeBtn.classList.add('hidden');
        }
        if (regenerateStoryBtn) {
            regenerateStoryBtn.classList.add('hidden');
        }

        // Update the input section notice if audio is also disabled
        // (READ_ONLY_MODE implies audio disabled, so update the message)
        const existingNotice = document.querySelector('.audio-disabled-notice');
        if (existingNotice) {
            existingNotice.innerHTML = `
                <span class="notice-icon"><svg class="icon"><use href="#icon-shield"></use></svg></span>
                <div class="notice-text">
                    <strong>Read-Only Mode</strong>
                    <p>This server is in read-only mode. You can browse and download existing analyses from the archive below, but new analysis and regeneration are disabled.</p>
                </div>
            `;
            existingNotice.classList.remove('audio-disabled-notice');
            existingNotice.classList.add('read-only-notice');
        }
    }

    initElements() {
        this.recordBtn = document.getElementById('record-btn');
        this.recordTime = document.getElementById('record-time');
        this.uploadBtn = document.getElementById('upload-btn');
        this.fileInput = document.getElementById('file-input');
        this.fileName = document.getElementById('file-name');
        this.audioPreview = document.getElementById('audio-preview');
        this.audioPlayer = document.getElementById('audio-player');
        this.analyzeBtn = document.getElementById('analyze-btn');
        this.processing = document.getElementById('processing');
        this.resultsSection = document.getElementById('results-section');
        this.statusBanner = document.getElementById('status-banner');
        this.statusText = document.getElementById('status-text');
        this.downloadBtn = document.getElementById('download-btn');
        this.downloadAudioBtn = document.getElementById('download-audio-btn');
        this.saveAudioBtn = document.getElementById('save-audio-btn');
        this.archiveList = document.getElementById('archive-list');
        this.refreshArchiveBtn = document.getElementById('refresh-archive-btn');
        this.titleInput = document.getElementById('analysis-title-input');
        this.starBtn = document.getElementById('star-btn');
    }

    bindEvents() {
        // Record button
        this.recordBtn.addEventListener('click', () => this.toggleRecording());

        // Upload button
        this.uploadBtn.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Analyze button
        this.analyzeBtn.addEventListener('click', () => this.analyzeAudio());

        // Download buttons
        this.downloadBtn.addEventListener('click', () => this.downloadReport());
        this.downloadAudioBtn.addEventListener('click', () => this.downloadAudio());
        this.saveAudioBtn.addEventListener('click', () => this.downloadAudio());

        // Tabs
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });

        // Title input
        if (this.titleInput) {
            this.titleInput.addEventListener('change', () => this.saveTitle());
            this.titleInput.addEventListener('blur', () => this.saveTitle());
        }
    }

    toggleSection(sectionId) {
        if (this.collapsedSections.has(sectionId)) {
            this.collapsedSections.delete(sectionId);
        } else {
            this.collapsedSections.add(sectionId);
        }
        this.updateSectionState(sectionId);
    }

    updateSectionState(sectionId) {
        const section = document.getElementById(sectionId);
        if (!section) return;

        const isCollapsed = this.collapsedSections.has(sectionId);
        section.classList.toggle('collapsed', isCollapsed);

        const icon = section.querySelector('.collapse-icon');
        if (icon) {
            icon.innerHTML = isCollapsed ? '&#9654;' : '&#9660;';
        }
    }

    // =========================================================================
    // Service Status Methods
    // =========================================================================

    toggleServiceStatus() {
        const panel = document.getElementById('service-status');
        const details = document.getElementById('service-status-details');
        if (panel && details) {
            panel.classList.toggle('expanded');
            details.classList.toggle('hidden');
        }
    }

    async checkServicesStatus() {
        try {
            const status = await this.api.getServicesStatus();
            this.updateServiceStatusDisplay(status);
        } catch (error) {
            console.error('Failed to check services status:', error);
            this.updateServiceStatusDisplay(null);
        }
    }

    updateServiceStatusDisplay(status) {
        const summaryEl = document.getElementById('service-status-summary');

        if (!status) {
            if (summaryEl) {
                summaryEl.textContent = 'Unable to check services';
                summaryEl.style.color = 'var(--danger)';
            }
            return;
        }

        // Update individual service statuses
        this.updateServiceItem('whisper-status', status.whisper);
        this.updateServiceItem('ollama-status', status.ollama);
        this.updateServiceItem('diarization-status', status.diarization);
        this.updateServiceItem('horizons-status', status.horizons);

        // Update summary (core services: whisper, ollama, diarization)
        const coreServices = [status.whisper, status.ollama, status.diarization];
        const readyCount = coreServices.filter(s => s.available).length;
        const horizonsConnected = status.horizons?.available;

        // Track horizons availability for telemetry recording
        this.horizonsAvailable = horizonsConnected;
        this.updateTelemetryCheckboxState();

        if (summaryEl) {
            if (readyCount === 3) {
                summaryEl.textContent = horizonsConnected ? 'All services ready' : 'Services ready (no game)';
                summaryEl.style.color = 'var(--success)';
            } else if (readyCount >= 1) {
                summaryEl.textContent = `${readyCount}/3 services ready`;
                summaryEl.style.color = 'var(--warning)';
            } else {
                summaryEl.textContent = 'No services available';
                summaryEl.style.color = 'var(--danger)';
            }
        }
    }

    updateTelemetryCheckboxState() {
        const telemetryOption = document.getElementById('telemetry-option');
        const telemetryCheckbox = document.getElementById('include-telemetry');
        const badge = telemetryOption?.querySelector('.option-badge');

        if (telemetryOption && telemetryCheckbox) {
            if (this.horizonsAvailable) {
                telemetryOption.classList.remove('unavailable');
                telemetryCheckbox.disabled = false;
                if (badge) {
                    badge.textContent = 'LIVE';
                    badge.classList.remove('unavailable');
                }
                telemetryOption.title = 'Record game telemetry alongside audio';
            } else {
                telemetryOption.classList.add('unavailable');
                telemetryCheckbox.disabled = true;
                telemetryCheckbox.checked = false;
                if (badge) {
                    badge.textContent = 'OFFLINE';
                }
                telemetryOption.title = 'Horizons game server not connected';
            }
        }
    }

    updateServiceItem(elementId, serviceStatus) {
        const item = document.getElementById(elementId);
        if (!item) return;

        const stateEl = item.querySelector('.service-state');
        const detailEl = item.querySelector('.service-detail');

        if (stateEl) {
            stateEl.textContent = serviceStatus.status;

            // Determine state for styling
            let state = 'unavailable';
            if (serviceStatus.available) {
                state = serviceStatus.status.toLowerCase().includes('ready') ? 'ready' : 'available';
            } else if (serviceStatus.status.toLowerCase().includes('error')) {
                state = 'error';
            }
            stateEl.setAttribute('data-state', state);
        }

        if (detailEl) {
            detailEl.textContent = serviceStatus.details || '';
            detailEl.title = serviceStatus.details || '';
        }
    }

    async toggleRecording() {
        if (this.recorder.isRecording) {
            // Stop recording
            this.recordBtn.innerHTML = '<svg class="icon"><use href="#icon-record"></use></svg><span>Record</span>';
            this.recordBtn.classList.remove('recording');
            clearInterval(this.timerInterval);

            // Stop telemetry recording if active
            if (this.telemetrySessionId) {
                try {
                    await this.api.stopTelemetry(this.telemetrySessionId);
                } catch (error) {
                    console.error('Failed to stop telemetry:', error);
                }
            }

            const blob = await this.recorder.stop();
            if (blob) {
                this.currentBlob = blob;
                this.showAudioPreview(blob);
                // Show save button for recorded audio
                this.saveAudioBtn.classList.remove('hidden');
            }
        } else {
            // Start recording
            try {
                // Check if telemetry recording is enabled
                const includeTelemetry = document.getElementById('include-telemetry')?.checked && this.horizonsAvailable;

                // Start telemetry recording first if enabled
                if (includeTelemetry) {
                    try {
                        const telemetryResult = await this.api.startTelemetry();
                        this.telemetrySessionId = telemetryResult.session_id;
                        console.log('Telemetry recording started:', this.telemetrySessionId);
                    } catch (error) {
                        console.error('Failed to start telemetry:', error);
                        this.showStatus('Warning: Telemetry recording failed to start', 'warning');
                        this.telemetrySessionId = null;
                    }
                } else {
                    this.telemetrySessionId = null;
                }

                await this.recorder.start();
                this.recordBtn.innerHTML = '<svg class="icon"><use href="#icon-stop"></use></svg><span>Stop</span>';
                this.recordBtn.classList.add('recording');

                // Update timer
                this.timerInterval = setInterval(() => {
                    const elapsed = this.recorder.getElapsedTime();
                    const mins = Math.floor(elapsed / 60);
                    const secs = elapsed % 60;
                    this.recordTime.textContent =
                        `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
                }, 1000);

                this.fileName.textContent = '';
                this.hideStatus();
                this.saveAudioBtn.classList.add('hidden');

                // Disable telemetry checkbox during recording
                const telemetryOption = document.getElementById('telemetry-option');
                if (telemetryOption) {
                    telemetryOption.style.pointerEvents = 'none';
                    telemetryOption.style.opacity = '0.6';
                }
            } catch (error) {
                this.showStatus('Failed to access microphone: ' + error.message, 'error');
                // Clean up telemetry if audio failed
                if (this.telemetrySessionId) {
                    try {
                        await this.api.stopTelemetry(this.telemetrySessionId);
                    } catch (e) {}
                    this.telemetrySessionId = null;
                }
            }
        }

        // Re-enable telemetry checkbox when not recording
        if (!this.recorder.isRecording) {
            const telemetryOption = document.getElementById('telemetry-option');
            if (telemetryOption) {
                telemetryOption.style.pointerEvents = '';
                telemetryOption.style.opacity = '';
            }
            this.updateTelemetryCheckboxState();
        }
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;

        this.currentBlob = file;
        this.fileName.textContent = file.name;
        this.showAudioPreview(file);
    }

    showAudioPreview(blob) {
        const url = URL.createObjectURL(blob);
        this.audioPlayer.src = url;
        this.audioPreview.classList.remove('hidden');

        // Collapse results section when preparing new audio
        this.collapsedSections.add('results-section');
        this.updateSectionState('results-section');
    }

    async analyzeAudio() {
        if (!this.currentBlob) {
            this.showStatus('No audio to analyze', 'error');
            return;
        }

        this.analyzeBtn.disabled = true;
        this.processing.classList.remove('hidden');
        this.hideStatus();
        this.resetProgress();

        try {
            // Create file from blob if needed
            let file = this.currentBlob;
            if (!(file instanceof File)) {
                const extension = this.getExtensionFromMimeType(file.type);
                file = new File([file], `recording${extension}`, { type: file.type });
            }

            // Read analysis options from checkboxes
            const includeNarrative = document.getElementById('include-narrative')?.checked ?? true;
            const includeStory = document.getElementById('include-story')?.checked ?? true;

            // Include telemetry session ID if available
            const telemetrySessionId = this.telemetrySessionId;

            // Use streaming endpoint with progress updates
            const results = await this.api.analyzeWithProgress(file, {
                includeNarrative,
                includeStory,
                telemetrySessionId
            }, (step, label, progress) => {
                this.updateProgress(step, label, progress);
            });

            // Clear telemetry session after analysis
            this.telemetrySessionId = null;

            this.currentResults = results;
            this.savedRecordingPath = results.saved_recording_path || null;
            this.currentFilename = results.saved_analysis_path?.split('/').pop() || null;

            this.renderer.render(results);

            // Show results section and expand it
            this.collapsedSections.delete('results-section');
            this.updateSectionState('results-section');

            // Show/hide download audio button based on saved recording
            if (this.savedRecordingPath) {
                this.downloadAudioBtn.classList.remove('hidden');
            } else {
                this.downloadAudioBtn.classList.add('hidden');
            }

            // Update analysis info section
            this.updateAnalysisInfo();

            this.showStatus('Analysis complete!', 'success');

            // Refresh archive to show new analysis
            this.loadArchive();
        } catch (error) {
            console.error('Analysis failed:', error);
            this.showStatus('Analysis failed: ' + error.message, 'error');
        } finally {
            this.analyzeBtn.disabled = false;
            this.processing.classList.add('hidden');
            this.stopElapsedTimer();
        }
    }

    updateAnalysisInfo() {
        const infoSection = document.getElementById('current-analysis-info');
        if (!infoSection) return;

        if (this.currentFilename) {
            infoSection.classList.remove('hidden');

            // Try to get metadata from archive
            const metadata = this.archiveData.find(a => a.filename === this.currentFilename);
            if (metadata) {
                this.currentMetadata = metadata;
                this.titleInput.value = metadata.display_title || '';
                this.starBtn.innerHTML = metadata.starred
                    ? '<svg class="icon"><use href="#icon-star-filled"></use></svg>'
                    : '<svg class="icon"><use href="#icon-star"></use></svg>';
                this.starBtn.classList.toggle('starred', metadata.starred);
            }
        } else {
            infoSection.classList.add('hidden');
        }
    }

    async saveTitle() {
        if (!this.currentFilename || !this.titleInput) return;

        const newTitle = this.titleInput.value.trim();
        if (!newTitle) return;

        try {
            await this.api.updateMetadata(this.currentFilename, {
                user_title: newTitle
            });
            this.showStatus('Title saved', 'success');
            // Refresh archive to show updated title
            this.loadArchive();
        } catch (error) {
            console.error('Failed to save title:', error);
            this.showStatus('Failed to save title', 'error');
        }
    }

    async toggleStar() {
        if (!this.currentFilename) return;

        const newStarred = !(this.currentMetadata?.starred || false);

        try {
            await this.api.updateMetadata(this.currentFilename, {
                starred: newStarred
            });

            this.starBtn.innerHTML = newStarred
                ? '<svg class="icon"><use href="#icon-star-filled"></use></svg>'
                : '<svg class="icon"><use href="#icon-star"></use></svg>';
            this.starBtn.classList.toggle('starred', newStarred);

            if (this.currentMetadata) {
                this.currentMetadata.starred = newStarred;
            }

            // Refresh archive
            this.loadArchive();
        } catch (error) {
            console.error('Failed to toggle star:', error);
        }
    }

    resetProgress() {
        // Reset progress bar
        const progressFill = document.getElementById('progress-fill');
        const progressPercent = document.getElementById('progress-percent');
        if (progressFill) progressFill.style.width = '0%';
        if (progressPercent) progressPercent.textContent = '0%';

        // Reset current step label and counter
        const currentStepLabel = document.getElementById('current-step-label');
        const stepCounter = document.getElementById('step-counter');
        if (currentStepLabel) currentStepLabel.textContent = 'Starting analysis...';
        if (stepCounter) stepCounter.textContent = 'Step 0 of 10';

        // Reset time displays
        const elapsedTime = document.getElementById('elapsed-time');
        const remainingTime = document.getElementById('remaining-time');
        if (elapsedTime) elapsedTime.textContent = '0s elapsed';
        if (remainingTime) remainingTime.textContent = 'estimating...';

        // Start elapsed time tracking
        this.analysisStartTime = Date.now();
        this.stopElapsedTimer(); // Clear any existing timer
        this.elapsedTimerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - this.analysisStartTime) / 1000);
            const elapsedEl = document.getElementById('elapsed-time');
            if (elapsedEl) {
                elapsedEl.textContent = `${elapsed}s elapsed`;
            }
        }, 1000);

        // Original step labels
        this.originalLabels = {
            'convert': 'Converting audio',
            'transcribe': 'Transcribing',
            'diarize': 'Identifying crew',
            'roles': 'Inferring roles',
            'quality': 'Communication quality',
            'scorecards': 'Generating scorecards',
            'confidence': 'Analyzing confidence',
            'learning': 'Learning metrics',
            'habits': '7 Habits analysis',
            'training': 'Recommendations'
        };

        // Step order for index tracking
        this.stepOrder = ['convert', 'transcribe', 'diarize', 'roles', 'quality', 'scorecards', 'confidence', 'learning', 'habits', 'training'];

        // Reset all steps
        document.querySelectorAll('#progress-steps .step').forEach(step => {
            step.classList.remove('active', 'completed');
            const icon = step.querySelector('.step-icon');
            const stepLabel = step.querySelector('.step-label');
            const stepId = step.dataset.step;
            if (icon) icon.textContent = '○'; // Empty circle
            // Restore original label
            if (stepLabel && this.originalLabels[stepId]) {
                stepLabel.textContent = this.originalLabels[stepId];
            }
        });
    }

    stopElapsedTimer() {
        if (this.elapsedTimerInterval) {
            clearInterval(this.elapsedTimerInterval);
            this.elapsedTimerInterval = null;
        }
    }

    updateProgress(stepId, label, progress) {
        // Update progress bar
        const progressFill = document.getElementById('progress-fill');
        const progressPercent = document.getElementById('progress-percent');
        if (progressFill) progressFill.style.width = `${progress}%`;
        if (progressPercent) progressPercent.textContent = `${Math.round(progress)}%`;

        // Show narrative loading when narrative step starts
        if (stepId === 'narrative') {
            this.showNarrativeLoading();
        } else if (stepId === 'complete') {
            this.stopNarrativeLoading();
            this.stopElapsedTimer();
        }

        // Get step index for "Step X of Y" display
        const stepIndex = this.stepOrder ? this.stepOrder.indexOf(stepId) : -1;
        const totalSteps = this.stepOrder ? this.stepOrder.length : 10;

        // Update current step label (prominent display)
        const currentStepLabelEl = document.getElementById('current-step-label');
        if (currentStepLabelEl && label) {
            currentStepLabelEl.textContent = label;
        }

        // Update step counter
        const stepCounter = document.getElementById('step-counter');
        if (stepCounter && stepIndex >= 0) {
            stepCounter.textContent = `Step ${stepIndex + 1} of ${totalSteps}`;
        }

        // Update remaining time estimate
        this.updateRemainingTime(stepIndex, progress);

        // Update step states in the details list
        const steps = document.querySelectorAll('#progress-steps .step');
        let foundCurrent = false;

        steps.forEach(step => {
            const currentStepId = step.dataset.step;
            const icon = step.querySelector('.step-icon');
            const stepLabelEl = step.querySelector('.step-label');

            if (currentStepId === stepId) {
                // This is the current active step
                step.classList.add('active');
                step.classList.remove('completed');
                if (icon) icon.textContent = '●'; // Filled circle
                // Update label with dynamic text (e.g., "Transcribing... 5.2s / 20.5s")
                if (stepLabelEl && label) stepLabelEl.textContent = label;
                foundCurrent = true;
            } else if (!foundCurrent) {
                // Steps before current are completed
                step.classList.remove('active');
                step.classList.add('completed');
                if (icon) icon.textContent = '✓'; // Checkmark
                // Restore original label for completed steps
                if (stepLabelEl && this.originalLabels && this.originalLabels[currentStepId]) {
                    stepLabelEl.textContent = this.originalLabels[currentStepId];
                }
            } else {
                // Steps after current are pending
                step.classList.remove('active', 'completed');
                if (icon) icon.textContent = '○'; // Empty circle
            }
        });
    }

    updateRemainingTime(currentStepIndex, progress) {
        const remainingTimeEl = document.getElementById('remaining-time');
        if (!remainingTimeEl || !this.analysisStartTime) return;

        const elapsed = (Date.now() - this.analysisStartTime) / 1000;

        // Need at least some progress to estimate
        if (progress < 5) {
            remainingTimeEl.textContent = 'estimating...';
            return;
        }

        // Calculate remaining time based on progress
        const estimatedTotal = (elapsed / progress) * 100;
        const remaining = Math.max(0, Math.round(estimatedTotal - elapsed));

        if (remaining === 0 || progress >= 100) {
            remainingTimeEl.textContent = 'almost done...';
        } else if (remaining < 60) {
            remainingTimeEl.textContent = `~${remaining}s remaining`;
        } else {
            const mins = Math.floor(remaining / 60);
            const secs = remaining % 60;
            remainingTimeEl.textContent = `~${mins}m ${secs}s remaining`;
        }
    }

    // Show narrative loading with timer
    showNarrativeLoading() {
        const loadingEl = document.getElementById('narrative-loading');
        const contentEl = document.getElementById('narrative-section');
        const unavailableEl = document.getElementById('narrative-unavailable');

        if (loadingEl) loadingEl.classList.remove('hidden');
        if (contentEl) contentEl.classList.add('hidden');
        if (unavailableEl) unavailableEl.classList.add('hidden');

        // Start timer
        this.narrativeStartTime = Date.now();
        this.narrativeTimerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - this.narrativeStartTime) / 1000);
            const timerEl = document.getElementById('narrative-timer');
            if (timerEl) {
                timerEl.textContent = `${elapsed}s elapsed`;
            }
        }, 1000);
    }

    // Stop narrative loading timer
    stopNarrativeLoading() {
        if (this.narrativeTimerInterval) {
            clearInterval(this.narrativeTimerInterval);
            this.narrativeTimerInterval = null;
        }
    }

    getExtensionFromMimeType(mimeType) {
        const map = {
            'audio/webm': '.webm',
            'audio/webm;codecs=opus': '.webm',
            'audio/ogg': '.ogg',
            'audio/ogg;codecs=opus': '.ogg',
            'audio/mp4': '.m4a',
            'audio/mpeg': '.mp3',
            'audio/wav': '.wav',
            'audio/x-wav': '.wav'
        };
        return map[mimeType] || '.webm';
    }

    switchTab(tabId) {
        // Update tab buttons
        document.querySelectorAll('.tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.tab === tabId);
        });

        // Update tab panels
        document.querySelectorAll('.tab-panel').forEach(panel => {
            panel.classList.toggle('active', panel.id === `tab-${tabId}`);
            panel.classList.toggle('hidden', panel.id !== `tab-${tabId}`);
        });
    }

    downloadReport() {
        if (!this.currentResults) {
            this.showStatus('No analysis results to download', 'error');
            return;
        }

        // Generate markdown
        const markdown = this.renderer.generateMarkdown(this.currentResults);

        // Create and download file
        const blob = new Blob([markdown], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;

        // Generate filename with timestamp
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
        a.download = `audio-analysis-${timestamp}.md`;

        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        this.showStatus('Report downloaded!', 'success');
    }

    downloadAudio() {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);

        // If we have a server-saved path, use the API endpoint
        if (this.savedRecordingPath) {
            const filename = this.savedRecordingPath.split('/').pop();
            const a = document.createElement('a');
            a.href = `/api/recordings/${filename}`;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            this.showStatus('Audio downloaded!', 'success');
            return;
        }

        // Otherwise download the local blob (before analysis)
        if (this.currentBlob) {
            const extension = this.getExtensionFromMimeType(this.currentBlob.type);
            const filename = `recording-${timestamp}${extension}`;
            const url = URL.createObjectURL(this.currentBlob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            this.showStatus('Audio downloaded!', 'success');
            return;
        }

        this.showStatus('No audio to download', 'error');
    }

    async loadArchive() {
        try {
            const data = await this.api.getArchiveIndex();
            this.archiveData = data.analyses || [];

            // Update archive count badge
            const countBadge = document.getElementById('archive-count');
            if (countBadge) {
                countBadge.textContent = this.archiveData.length;
            }

            this.renderArchive(this.archiveData);
        } catch (error) {
            console.error('Failed to load archive:', error);
            // Fall back to simple analyses list
            try {
                const data = await this.api.listAnalyses();
                this.archiveData = data.analyses || [];
                this.renderArchive(this.archiveData);
            } catch (e) {
                this.archiveList.innerHTML = '<p class="empty">Failed to load archive</p>';
            }
        }
    }

    filterArchive() {
        const searchQuery = document.getElementById('archive-search')?.value?.toLowerCase() || '';
        const starredOnly = document.getElementById('starred-only')?.checked || false;

        let filtered = this.archiveData;

        if (searchQuery) {
            filtered = filtered.filter(a =>
                (a.display_title || '').toLowerCase().includes(searchQuery) ||
                (a.auto_title || '').toLowerCase().includes(searchQuery) ||
                (a.notes || '').toLowerCase().includes(searchQuery)
            );
        }

        if (starredOnly) {
            filtered = filtered.filter(a => a.starred);
        }

        this.renderArchive(filtered);
    }

    renderArchive(analyses) {
        if (!analyses || analyses.length === 0) {
            this.archiveList.innerHTML = '<p class="empty">No saved analyses yet</p>';
            return;
        }

        this.archiveList.innerHTML = analyses.map(analysis => {
            const date = new Date(analysis.created_at);
            const dateStr = date.toLocaleDateString();
            const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            const duration = this.formatDuration(analysis.duration_seconds);
            const title = analysis.display_title || `${dateStr} ${timeStr}`;
            const isStarred = analysis.starred;

            return `
                <div class="archive-item ${isStarred ? 'starred' : ''}" data-filename="${analysis.filename}">
                    <div class="archive-item-info" onclick="app.loadArchivedAnalysis('${analysis.filename}')">
                        <div class="archive-item-header">
                            ${isStarred ? '<span class="star-indicator"><svg class="icon icon-sm"><use href="#icon-star-filled"></use></svg></span>' : ''}
                            <span class="archive-item-title">${this.escapeHtml(title)}</span>
                        </div>
                        <div class="archive-item-meta">
                            <span>${dateStr} ${timeStr}</span>
                            <span>${duration}</span>
                            <span>${analysis.speaker_count || 0} speakers</span>
                        </div>
                    </div>
                    <div class="archive-item-actions">
                        ${analysis.recording_filename && !this.audioDisabled ? `
                            <button class="btn-icon" onclick="event.stopPropagation(); app.downloadArchivedAudio('${analysis.recording_filename}')" title="Download Audio">
                                <svg class="icon"><use href="#icon-waveform"></use></svg>
                            </button>
                        ` : ''}
                        ${!this.readOnlyMode ? `
                            <button class="btn-icon delete" onclick="event.stopPropagation(); app.deleteAnalysis('${analysis.filename}')" title="Delete">
                                <svg class="icon"><use href="#icon-trash"></use></svg>
                            </button>
                        ` : ''}
                    </div>
                </div>
            `;
        }).join('');
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    async loadArchivedAnalysis(filename) {
        // Find and mark the clicked item as loading
        const archiveItem = document.querySelector(`.archive-item[data-filename="${filename}"]`);
        if (archiveItem) {
            archiveItem.classList.add('loading');
        }

        try {
            const data = await this.api.getAnalysis(filename);

            if (data && data.results) {
                this.currentResults = data.results;
                this.currentFilename = filename;
                this.savedRecordingPath = null;

                // Check for linked recording
                if (data.metadata && data.metadata.recording_file) {
                    this.savedRecordingPath = `data/recordings/${data.metadata.recording_file}`;
                    this.downloadAudioBtn.classList.remove('hidden');
                    // Set audio player source for click-to-play timeline
                    if (this.audioPlayer) {
                        this.audioPlayer.src = `/api/recordings/${data.metadata.recording_file}`;
                    }
                } else {
                    this.downloadAudioBtn.classList.add('hidden');
                }

                this.renderer.render(data.results);

                // Expand results section
                this.collapsedSections.delete('results-section');
                this.updateSectionState('results-section');

                // Update analysis info
                this.updateAnalysisInfo();

                this.hideStatus();

                // Scroll to results
                this.resultsSection.scrollIntoView({ behavior: 'smooth' });
            }
        } catch (error) {
            console.error('Failed to load analysis:', error);
            this.showStatus('Failed to load analysis: ' + error.message, 'error');
        } finally {
            // Remove loading state
            if (archiveItem) {
                archiveItem.classList.remove('loading');
            }
        }
    }

    downloadArchivedAudio(filename) {
        const a = document.createElement('a');
        a.href = `/api/recordings/${filename}`;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        this.showStatus('Audio downloaded!', 'success');
    }

    async deleteAnalysis(filename) {
        if (!confirm('Delete this analysis? This cannot be undone.')) {
            return;
        }

        try {
            await this.api.deleteAnalysis(filename);
            this.showStatus('Analysis deleted', 'success');
            this.loadArchive();
        } catch (error) {
            console.error('Failed to delete analysis:', error);
            this.showStatus('Failed to delete: ' + error.message, 'error');
        }
    }

    async regenerateNarrative() {
        if (!this.currentFilename) {
            this.showStatus('No analysis loaded to regenerate', 'error');
            return;
        }

        const btn = document.getElementById('regenerate-narrative-btn');
        const contentEl = document.getElementById('narrative-section');
        const unavailableEl = document.getElementById('narrative-unavailable');
        const narrativeContent = document.getElementById('narrative-content');

        // Show loading state
        btn.classList.add('regenerating');
        btn.disabled = true;
        const originalText = btn.querySelector('span:last-child').textContent;
        btn.querySelector('span:last-child').textContent = 'Regenerating...';

        // Hide unavailable, show content section with progress UI
        if (unavailableEl) unavailableEl.classList.add('hidden');
        if (contentEl) contentEl.classList.remove('hidden');

        const startTime = Date.now();

        const updateProgress = (chars, message) => {
            const elapsed = Math.round((Date.now() - startTime) / 1000);
            if (narrativeContent) {
                narrativeContent.innerHTML = `
                    <div class="regenerating-progress">
                        <div class="progress-spinner"></div>
                        <div class="progress-stats">
                            <div class="progress-message">${message}</div>
                            <div class="progress-details">
                                <span class="stat"><strong>${chars.toLocaleString()}</strong> characters</span>
                                <span class="stat"><strong>${elapsed}s</strong> elapsed</span>
                            </div>
                        </div>
                    </div>
                `;
            }
        };

        updateProgress(0, 'Connecting to LLM...');

        try {
            const response = await fetch(`/api/analyses/${this.currentFilename}/regenerate-narrative-stream`, {
                method: 'POST'
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));

                            if (data.type === 'progress') {
                                updateProgress(data.chars || 0, data.message || 'Generating...');
                            } else if (data.type === 'complete') {
                                this.renderer.renderNarrative(data.narrative_summary);
                                this.showStatus(`Team analysis regenerated (${data.chars?.toLocaleString() || 0} chars)`, 'success');
                                if (this.currentResults) {
                                    this.currentResults.narrative_summary = data.narrative_summary;
                                }
                            } else if (data.type === 'error') {
                                throw new Error(data.message);
                            }
                        } catch (e) {
                            if (e.message !== 'Unexpected end of JSON input') {
                                throw e;
                            }
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Error regenerating narrative:', error);
            this.showStatus(`Failed to regenerate: ${error.message}`, 'error');
            if (this.currentResults?.narrative_summary) {
                this.renderer.renderNarrative(this.currentResults.narrative_summary);
            } else {
                if (contentEl) contentEl.classList.add('hidden');
                if (unavailableEl) {
                    unavailableEl.classList.remove('hidden');
                    unavailableEl.innerHTML = `
                        <div class="llm-skipped-notice">
                            <span class="notice-icon"><svg class="icon"><use href="#icon-shield"></use></svg></span>
                            <strong>Generation Failed</strong>
                            <p>${error.message}</p>
                            <p class="notice-tip">Click "Regenerate" to try again.</p>
                        </div>
                    `;
                }
            }
        } finally {
            btn.classList.remove('regenerating');
            btn.disabled = false;
            btn.querySelector('span:last-child').textContent = originalText;
        }
    }

    async regenerateStory() {
        if (!this.currentFilename) {
            this.showStatus('No analysis loaded to regenerate', 'error');
            return;
        }

        const btn = document.getElementById('regenerate-story-btn');
        const contentEl = document.getElementById('story-section');
        const unavailableEl = document.getElementById('story-unavailable');
        const storyContent = document.getElementById('story-content');

        // Show loading state
        btn.classList.add('regenerating');
        btn.disabled = true;
        const originalText = btn.querySelector('span:last-child').textContent;
        btn.querySelector('span:last-child').textContent = 'Regenerating...';

        // Hide unavailable, show content section with progress UI
        if (unavailableEl) unavailableEl.classList.add('hidden');
        if (contentEl) contentEl.classList.remove('hidden');

        const startTime = Date.now();

        const updateProgress = (chars, message) => {
            const elapsed = Math.round((Date.now() - startTime) / 1000);
            if (storyContent) {
                storyContent.innerHTML = `
                    <div class="regenerating-progress">
                        <div class="progress-spinner"></div>
                        <div class="progress-stats">
                            <div class="progress-message">${message}</div>
                            <div class="progress-details">
                                <span class="stat"><strong>${chars.toLocaleString()}</strong> characters</span>
                                <span class="stat"><strong>${elapsed}s</strong> elapsed</span>
                            </div>
                        </div>
                    </div>
                `;
            }
        };

        updateProgress(0, 'Connecting to LLM...');

        try {
            const response = await fetch(`/api/analyses/${this.currentFilename}/regenerate-story-stream`, {
                method: 'POST'
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));

                            if (data.type === 'progress') {
                                updateProgress(data.chars || 0, data.message || 'Writing...');
                            } else if (data.type === 'complete') {
                                this.renderer.renderStory(data.story_narrative);
                                this.showStatus(`Mission story regenerated (${data.chars?.toLocaleString() || 0} chars)`, 'success');
                                if (this.currentResults) {
                                    this.currentResults.story_narrative = data.story_narrative;
                                }
                            } else if (data.type === 'error') {
                                throw new Error(data.message);
                            }
                        } catch (e) {
                            if (e.message !== 'Unexpected end of JSON input') {
                                throw e;
                            }
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Error regenerating story:', error);
            this.showStatus(`Failed to regenerate: ${error.message}`, 'error');
            if (this.currentResults?.story_narrative) {
                this.renderer.renderStory(this.currentResults.story_narrative);
            } else {
                if (contentEl) contentEl.classList.add('hidden');
                if (unavailableEl) {
                    unavailableEl.classList.remove('hidden');
                    unavailableEl.innerHTML = `
                        <div class="llm-skipped-notice">
                            <span class="notice-icon"><svg class="icon"><use href="#icon-shield"></use></svg></span>
                            <strong>Generation Failed</strong>
                            <p>${error.message}</p>
                            <p class="notice-tip">Click "Regenerate" to try again.</p>
                        </div>
                    `;
                }
            }
        } finally {
            btn.classList.remove('regenerating');
            btn.disabled = false;
            btn.querySelector('span:last-child').textContent = originalText;
        }
    }

    formatDuration(seconds) {
        if (!seconds) return '0:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    initiateDrill(button, drillIdx) {
        const card = document.getElementById(`drill-card-${drillIdx}`);
        const expanded = document.getElementById(`drill-expanded-${drillIdx}`);
        if (!card || !expanded) return;

        // Hide the initiate button
        button.classList.add('hidden');
        // Show expanded section
        expanded.classList.remove('hidden');
        // Add active glow
        card.classList.add('drill-active');
    }

    completeDrill(button, drillIdx) {
        const card = document.getElementById(`drill-card-${drillIdx}`);
        const expanded = document.getElementById(`drill-expanded-${drillIdx}`);
        if (!card || !expanded) return;

        // Success animation
        card.classList.remove('drill-active');
        card.classList.add('drill-completing');

        setTimeout(() => {
            card.classList.remove('drill-completing');
            card.classList.add('drill-completed');
            // Collapse the expanded section
            expanded.classList.add('hidden');
        }, 800);
    }

    showStatus(message, type = 'info') {
        this.statusText.textContent = message;
        this.statusBanner.className = `status-banner ${type}`;
        this.statusBanner.classList.remove('hidden');
    }

    hideStatus() {
        this.statusBanner.classList.add('hidden');
    }
}


// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});