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

    analyzeWithProgress(file, onProgress) {
        return new Promise((resolve, reject) => {
            const formData = new FormData();
            formData.append('file', file);

            fetch(`${this.baseUrl}/api/analyze-stream`, {
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
}


// ============================================================================
// Results Renderer
// ============================================================================

class ResultsRenderer {
    constructor(container) {
        this.container = container;
    }

    render(results) {
        // Update stats
        document.getElementById('duration').textContent =
            this.formatDuration(results.duration_seconds);
        document.getElementById('segment-count').textContent =
            results.transcription.length;
        document.getElementById('proc-time').textContent =
            `${results.processing_time_seconds.toFixed(1)}s`;

        // Render summary tab
        this.renderSummary(results);

        // Render transcript
        this.renderTranscript(results.transcription, results.full_text);

        // Render speakers
        this.renderSpeakers(results.speakers, results.role_assignments);

        // Render scorecards
        this.renderScorecards(results.speaker_scorecards);

        // Render quality
        this.renderQuality(results.communication_quality, results.confidence_distribution);

        // Render learning
        this.renderLearning(results.learning_evaluation);

        // Render 7 Habits
        this.renderHabits(results.seven_habits);

        // Render Training recommendations
        this.renderTraining(results.training_recommendations);
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

        // Key metrics bars
        const habitsScore = results.seven_habits?.overall_score || 0;
        const habitsPercent = (habitsScore / 5) * 100;
        document.getElementById('metric-habits-bar').style.width = `${habitsPercent}%`;
        document.getElementById('metric-habits-value').textContent = `${habitsScore.toFixed(1)}/5`;

        document.getElementById('metric-quality-bar').style.width = `${effectivePct}%`;
        document.getElementById('metric-quality-value').textContent = `${effectivePct.toFixed(0)}%`;

        const avgConfidence = results.confidence_distribution?.average_confidence || 0;
        const confidencePct = avgConfidence * 100;
        document.getElementById('metric-confidence-bar').style.width = `${confidencePct}%`;
        document.getElementById('metric-confidence-value').textContent = `${confidencePct.toFixed(0)}%`;

        // Top recommendations preview
        const topRecs = document.getElementById('top-recommendations');
        const recsPreview = document.getElementById('recommendations-preview');

        if (results.training_recommendations?.immediate_actions?.length > 0) {
            const actions = results.training_recommendations.immediate_actions.slice(0, 3);
            recsPreview.innerHTML = actions.map(action => `
                <div class="rec-preview-item priority-${action.priority.toLowerCase()}">
                    <span class="rec-priority-badge">${action.priority}</span>
                    <span class="rec-preview-title">${action.title}</span>
                </div>
            `).join('');
            topRecs.classList.remove('hidden');
        } else {
            topRecs.classList.add('hidden');
        }
    }

    renderTranscript(segments, fullText) {
        const container = document.getElementById('transcript-content');

        if (!segments || segments.length === 0) {
            container.innerHTML = '<p class="empty">No transcription available</p>';
        } else {
            container.innerHTML = segments.map(seg => `
                <div class="segment">
                    <div class="segment-header">
                        <span class="segment-speaker">${seg.speaker_id || 'Speaker'}</span>
                        <span class="segment-time">${this.formatTime(seg.start_time)} - ${this.formatTime(seg.end_time)}</span>
                    </div>
                    <div class="segment-text">${this.escapeHtml(seg.text)}</div>
                </div>
            `).join('');
        }

        document.getElementById('full-text').textContent = fullText || '';
    }

    renderSpeakers(speakers, roleAssignments) {
        const container = document.getElementById('speakers-content');

        if (!speakers || speakers.length === 0) {
            container.innerHTML = '<p class="empty">No speaker information available</p>';
            return;
        }

        // Build role lookup from role_assignments
        const roleMap = {};
        if (roleAssignments) {
            for (const ra of roleAssignments) {
                roleMap[ra.speaker_id] = {
                    role: ra.role,
                    confidence: ra.confidence
                };
            }
        }

        container.innerHTML = speakers.map(speaker => {
            const roleInfo = roleMap[speaker.speaker_id];
            const role = roleInfo?.role || speaker.role;
            const roleConfidence = roleInfo?.confidence;

            return `
                <div class="speaker-card">
                    <h3>
                        ${speaker.speaker_id}
                        ${role ? `<span class="speaker-role">${role}${roleConfidence ? ` (${(roleConfidence * 100).toFixed(0)}%)` : ''}</span>` : ''}
                    </h3>
                    <div class="speaker-stats">
                        <div class="speaker-stat">
                            <strong>${this.formatDuration(speaker.total_speaking_time)}</strong>
                            Speaking time
                        </div>
                        <div class="speaker-stat">
                            <strong>${speaker.utterance_count}</strong>
                            Utterances
                        </div>
                        <div class="speaker-stat">
                            <strong>${speaker.avg_utterance_duration.toFixed(1)}s</strong>
                            Avg duration
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    }

    renderQuality(quality, confidenceDistribution) {
        const container = document.getElementById('quality-content');

        if (!quality) {
            container.innerHTML = '<p class="empty">No quality analysis available</p>';
            return;
        }

        container.innerHTML = `
            <div class="quality-summary">
                <div class="quality-metric effective">
                    <div class="value">${quality.effective_count}</div>
                    <div class="label">Effective</div>
                </div>
                <div class="quality-metric improvement">
                    <div class="value">${quality.improvement_count}</div>
                    <div class="label">Needs Improvement</div>
                </div>
                <div class="quality-metric">
                    <div class="value">${quality.effective_percentage.toFixed(0)}%</div>
                    <div class="label">Effective Rate</div>
                </div>
            </div>

            ${quality.calculation_summary ? `
                <div class="evidence-panel">
                    <details>
                        <summary>Show Calculation Evidence</summary>
                        <div class="evidence-content">
                            <p>${quality.calculation_summary}</p>
                            <p>Total utterances assessed: ${quality.total_utterances_assessed || 'N/A'}</p>
                        </div>
                    </details>
                </div>
            ` : ''}

            ${confidenceDistribution ? `
                <div class="confidence-section">
                    <h3>Confidence Distribution</h3>
                    <div class="confidence-stats">
                        <span>Average: <strong>${(confidenceDistribution.average_confidence * 100).toFixed(0)}%</strong></span>
                        <span>Median: <strong>${(confidenceDistribution.median_confidence * 100).toFixed(0)}%</strong></span>
                    </div>
                    <div class="confidence-buckets">
                        ${confidenceDistribution.buckets.map(b => `
                            <div class="confidence-bucket">
                                <div class="bucket-bar" style="height: ${Math.min(100, b.percentage * 2)}%"></div>
                                <div class="bucket-label">${b.label}</div>
                                <div class="bucket-count">${b.count}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}

            ${quality.patterns && quality.patterns.length > 0 ? `
                <div class="patterns-section">
                    <h3>Detected Patterns</h3>
                    ${quality.patterns.map(p => `
                        <div class="pattern-card ${p.category}">
                            <div class="pattern-header">
                                <span class="pattern-name">${this.formatPatternName(p.pattern_name)}</span>
                                <span class="pattern-count ${p.category}">${p.count} instances</span>
                            </div>
                            ${p.description ? `<p class="pattern-description">${p.description}</p>` : ''}
                            ${p.examples?.length > 0 ? `
                                <div class="pattern-examples">
                                    <details>
                                        <summary>View Examples (${p.examples.length})</summary>
                                        <ul class="quote-list">
                                            ${p.examples.map(ex => `
                                                <li class="quote-item">
                                                    <span class="quote-speaker">${ex.speaker || 'Speaker'}:</span>
                                                    <span class="quote-text">"${this.escapeHtml(ex.text || ex)}"</span>
                                                    ${ex.assessment ? `<span class="quote-assessment">${ex.assessment}</span>` : ''}
                                                </li>
                                            `).join('')}
                                        </ul>
                                    </details>
                                </div>
                            ` : ''}
                        </div>
                    `).join('')}
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

        container.innerHTML = scorecards.map(card => `
            <div class="scorecard">
                <div class="scorecard-header">
                    <h3>
                        ${card.speaker_id}
                        ${card.role ? `<span class="speaker-role">${card.role}</span>` : ''}
                    </h3>
                    <span class="overall-score">${card.overall_score.toFixed(1)}/5</span>
                </div>

                <div class="score-grid">
                    ${card.metrics.map(m => `
                        <div class="score-item">
                            <div class="score-item-header">
                                <span class="score-name">${m.name}</span>
                                <div class="score-value">
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
                    `).join('')}
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
        `).join('');
    }

    renderLearning(learning) {
        const container = document.getElementById('learning-content');

        if (!learning) {
            container.innerHTML = '<p class="empty">No learning evaluation available</p>';
            return;
        }

        container.innerHTML = `
            ${learning.kirkpatrick_levels ? `
                <div class="learning-section">
                    <h3>Kirkpatrick 4-Level Model</h3>
                    <div class="kirkpatrick-levels">
                        ${learning.kirkpatrick_levels.map(lvl => `
                            <div class="kirk-level">
                                <div class="level-num">Level ${lvl.level}</div>
                                <div class="level-name">${lvl.name}</div>
                                <div class="level-score">${(lvl.score * 100).toFixed(0)}%</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}

            <div class="learning-section">
                <h3>Learning Frameworks</h3>
                <div class="framework-scores">
                    ${learning.blooms_level ? `
                        <div class="framework-score">
                            <div class="framework-name">Bloom's Taxonomy</div>
                            <div class="framework-value">${learning.blooms_level}</div>
                            <div class="framework-label">Cognitive Level</div>
                        </div>
                    ` : ''}
                    ${learning.nasa_tlx_score !== undefined ? `
                        <div class="framework-score">
                            <div class="framework-name">NASA TLX</div>
                            <div class="framework-value">${(learning.nasa_tlx_score * 100).toFixed(0)}%</div>
                            <div class="framework-label">Workload Index</div>
                        </div>
                    ` : ''}
                    ${learning.engagement_score !== undefined ? `
                        <div class="framework-score">
                            <div class="framework-name">Engagement</div>
                            <div class="framework-value">${(learning.engagement_score * 100).toFixed(0)}%</div>
                            <div class="framework-label">Overall Score</div>
                        </div>
                    ` : ''}
                </div>
            </div>

            ${learning.recommendations && learning.recommendations.length > 0 ? `
                <div class="learning-section">
                    <h3>Recommendations</h3>
                    <ul class="recommendations-list">
                        ${learning.recommendations.map(r => `<li>${r}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}

            ${learning.top_communications && learning.top_communications.length > 0 ? `
                <div class="learning-section">
                    <h3>Key Communications</h3>
                    <p class="section-description">Top utterances by transcription confidence:</p>
                    <ul class="quote-list">
                        ${learning.top_communications.map(comm => `
                            <li class="quote-item">
                                <span class="quote-speaker">${comm.speaker}:</span>
                                <span class="quote-text">"${this.escapeHtml(comm.text)}"</span>
                                <span class="quote-confidence">(${(comm.confidence * 100).toFixed(0)}% confidence)</span>
                            </li>
                        `).join('')}
                    </ul>
                </div>
            ` : ''}

            ${learning.speaker_statistics && Object.keys(learning.speaker_statistics).length > 0 ? `
                <div class="learning-section">
                    <h3>Speaker Statistics</h3>
                    <div class="speaker-stats-grid">
                        ${(Array.isArray(learning.speaker_statistics) ? learning.speaker_statistics : Object.values(learning.speaker_statistics)).map(stat => `
                            <div class="speaker-stat-card">
                                <strong>${stat.speaker}</strong>
                                <span>${stat.utterances} utterances (${stat.percentage}%)</span>
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

        container.innerHTML = `
            <div class="habits-overview">
                <h3>Leadership Effectiveness Assessment</h3>
                <div class="overall-score-display">
                    <div class="score-circle" style="--score: ${(habits.overall_score / 5) * 100}%">
                        <span class="score-value">${habits.overall_score.toFixed(1)}</span>
                        <span class="score-max">/5</span>
                    </div>
                    <div class="score-label">Overall Effectiveness</div>
                </div>
            </div>

            <div class="habits-grid">
                ${habits.habits.map(h => `
                    <div class="habit-card">
                        <div class="habit-header">
                            <span class="habit-number">Habit ${h.habit_number}</span>
                            <span class="habit-score score-${h.score}">${h.score}/5</span>
                        </div>
                        <div class="habit-name">${h.youth_friendly_name || h.habit_name}</div>
                        <div class="habit-observations">${h.observation_count} observations</div>
                        <div class="habit-interpretation">${h.interpretation}</div>
                        ${h.development_tip ? `
                            <div class="habit-tip">
                                <strong>Tip:</strong> ${h.development_tip}
                            </div>
                        ` : ''}
                        ${h.examples?.length > 0 || h.pattern_breakdown ? `
                            <div class="habit-evidence">
                                <details>
                                    <summary>Show Evidence</summary>
                                    ${h.pattern_breakdown && Object.keys(h.pattern_breakdown).length > 0 ? `
                                        <div class="pattern-breakdown">
                                            <strong>Pattern breakdown:</strong>
                                            <ul>
                                                ${Object.entries(h.pattern_breakdown).map(([k, v]) => `<li>${k}: ${v} matches</li>`).join('')}
                                            </ul>
                                        </div>
                                    ` : ''}
                                    ${h.gap_to_next_score ? `<p class="gap-info"><strong>To improve:</strong> ${h.gap_to_next_score}</p>` : ''}
                                    ${h.examples?.length > 0 ? `
                                        <div class="habit-examples">
                                            <strong>Evidence:</strong>
                                            <ul class="quote-list">
                                                ${h.examples.map(ex => `
                                                    <li class="quote-item">
                                                        <span class="quote-speaker">${ex.speaker || 'Speaker'}:</span>
                                                        <span class="quote-text">"${this.escapeHtml(ex.text || ex)}"</span>
                                                    </li>
                                                `).join('')}
                                            </ul>
                                        </div>
                                    ` : ''}
                                </details>
                            </div>
                        ` : ''}
                    </div>
                `).join('')}
            </div>

            ${habits.strengths && habits.strengths.length > 0 ? `
                <div class="habits-section strengths">
                    <h3>Team Strengths</h3>
                    <ul class="habits-list">
                        ${habits.strengths.map(s => `
                            <li>
                                <strong>${s.name}</strong> (${s.score}/5)
                                ${s.interpretation ? `<p>${s.interpretation}</p>` : ''}
                            </li>
                        `).join('')}
                    </ul>
                </div>
            ` : ''}

            ${habits.growth_areas && habits.growth_areas.length > 0 ? `
                <div class="habits-section growth-areas">
                    <h3>Growth Opportunities</h3>
                    <ul class="habits-list">
                        ${habits.growth_areas.map(g => `
                            <li>
                                <strong>${g.name}</strong> (${g.score}/5)
                                ${g.development_tip ? `<p class="dev-tip"><em>Tip:</em> ${g.development_tip}</p>` : ''}
                            </li>
                        `).join('')}
                    </ul>
                </div>
            ` : ''}
        `;
    }

    renderTraining(training) {
        const container = document.getElementById('training-content');

        if (!training) {
            container.innerHTML = '<p class="empty">No training recommendations available</p>';
            return;
        }

        container.innerHTML = `
            <div class="training-header">
                <h3>Training Recommendations</h3>
                <span class="total-count">${training.total_recommendations} recommendations</span>
            </div>

            ${training.immediate_actions && training.immediate_actions.length > 0 ? `
                <div class="training-section">
                    <h3>Immediate Actions</h3>
                    <div class="recommendations-grid">
                        ${training.immediate_actions.map(action => `
                            <div class="recommendation-card priority-${action.priority.toLowerCase()}">
                                <div class="rec-header">
                                    <span class="rec-priority">${action.priority}</span>
                                    <span class="rec-category">${action.category}</span>
                                </div>
                                <h4>${action.title}</h4>
                                <p>${action.description}</p>
                                ${action.trigger_metrics?.length > 0 || action.current_value ? `
                                    <details class="rec-evidence">
                                        <summary>Show Evidence</summary>
                                        ${action.trigger_metrics?.length > 0 ? `
                                            <p><strong>Triggered by:</strong> ${action.trigger_metrics.join(', ')}</p>
                                        ` : ''}
                                        ${action.current_value ? `<p><strong>Current:</strong> ${action.current_value}</p>` : ''}
                                        ${action.target_value ? `<p><strong>Target:</strong> ${action.target_value}</p>` : ''}
                                        ${action.gap_explanation ? `<p><strong>Gap:</strong> ${action.gap_explanation}</p>` : ''}
                                    </details>
                                ` : ''}
                                ${action.scout_connection ? `
                                    <div class="rec-connection scout">
                                        <strong>Scout:</strong> ${action.scout_connection}
                                    </div>
                                ` : ''}
                                ${action.habit_connection ? `
                                    <div class="rec-connection habit">
                                        <strong>7 Habits:</strong> ${action.habit_connection}
                                    </div>
                                ` : ''}
                                ${action.success_criteria ? `
                                    <div class="rec-success">
                                        <strong>Success:</strong> ${action.success_criteria}
                                    </div>
                                ` : ''}
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}

            ${training.drills && training.drills.length > 0 ? `
                <div class="training-section">
                    <h3>Training Drills</h3>
                    <div class="drills-list">
                        ${training.drills.map(drill => `
                            <div class="drill-card">
                                <div class="drill-header">
                                    <h4>${drill.name}</h4>
                                    <span class="drill-duration">${drill.duration}</span>
                                </div>
                                <p class="drill-purpose">${drill.purpose}</p>
                                ${drill.steps && drill.steps.length > 0 ? `
                                    <div class="drill-steps">
                                        <strong>Steps:</strong>
                                        <ol>
                                            ${drill.steps.map(s => `<li>${s}</li>`).join('')}
                                        </ol>
                                    </div>
                                ` : ''}
                                ${drill.debrief_questions && drill.debrief_questions.length > 0 ? `
                                    <div class="drill-debrief">
                                        <strong>Debrief Questions:</strong>
                                        <ul>
                                            ${drill.debrief_questions.map(q => `<li>${q}</li>`).join('')}
                                        </ul>
                                    </div>
                                ` : ''}
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}

            ${training.discussion_topics && training.discussion_topics.length > 0 ? `
                <div class="training-section">
                    <h3>Discussion Topics</h3>
                    <div class="topics-list">
                        ${training.discussion_topics.map(topic => `
                            <div class="topic-card">
                                <h4>${topic.topic}</h4>
                                <p class="topic-question">${topic.question}</p>
                                ${topic.scout_connection ? `
                                    <div class="topic-scout">
                                        <strong>Scout Connection:</strong> ${topic.scout_connection}
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
                    <div class="framework-alignment">
                        ${Object.entries(training.framework_alignment).map(([key, values]) => `
                            <div class="alignment-group">
                                <h4>${key}</h4>
                                <ul>
                                    ${values.slice(0, 3).map(v => `<li>${v}</li>`).join('')}
                                </ul>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
        `;
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
        let md = `# Audio Analysis Report\n\n`;
        md += `**Generated:** ${now}\n\n`;
        md += `---\n\n`;

        // Summary
        md += `## Summary\n\n`;
        md += `| Metric | Value |\n`;
        md += `|--------|-------|\n`;
        md += `| Duration | ${this.formatDuration(results.duration_seconds)} |\n`;
        md += `| Segments | ${results.transcription.length} |\n`;
        md += `| Speakers | ${results.speakers.length} |\n`;
        md += `| Processing Time | ${results.processing_time_seconds.toFixed(1)}s |\n`;
        md += `\n`;

        // Speakers with Role Assignments
        if (results.speakers && results.speakers.length > 0) {
            md += `## Speakers\n\n`;
            md += `| Speaker | Role | Speaking Time | Utterances | Avg Duration |\n`;
            md += `|---------|------|---------------|------------|-------------|\n`;

            // Build role map from role_assignments
            const roleMap = {};
            if (results.role_assignments) {
                for (const ra of results.role_assignments) {
                    roleMap[ra.speaker_id] = ra.role;
                }
            }

            for (const s of results.speakers) {
                const role = roleMap[s.speaker_id] || s.role || '-';
                md += `| ${s.speaker_id} | ${role} | ${this.formatDuration(s.total_speaking_time)} | ${s.utterance_count} | ${s.avg_utterance_duration.toFixed(1)}s |\n`;
            }
            md += `\n`;
        }

        // Continue with other sections...
        // (keeping the existing markdown generation for other sections)

        // Transcript
        md += `## Transcript\n\n`;
        if (results.transcription && results.transcription.length > 0) {
            for (const seg of results.transcription) {
                const speaker = seg.speaker_id || 'Speaker';
                const time = this.formatTime(seg.start_time);
                md += `**[${time}] ${speaker}:** ${seg.text}\n\n`;
            }
        } else {
            md += `*No transcription available*\n\n`;
        }

        // Full Text
        md += `## Full Text\n\n`;
        md += `${results.full_text || '*No text available*'}\n\n`;

        md += `---\n\n`;
        md += `*Generated by AI Audio Analyzer*\n`;

        return md;
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

        this.initElements();
        this.bindEvents();
        this.loadArchive();
        this.checkServicesStatus(); // Check service status on load

        // Start with results section collapsed if no results
        this.collapsedSections.add('results-section');
        this.updateSectionState('results-section');
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

        // Update summary
        const readyCount = [status.whisper, status.ollama, status.diarization]
            .filter(s => s.available).length;

        if (summaryEl) {
            if (readyCount === 3) {
                summaryEl.textContent = 'All services ready';
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
            this.recordBtn.innerHTML = '<span class="icon">&#9679;</span> Record';
            this.recordBtn.classList.remove('recording');
            clearInterval(this.timerInterval);

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
                await this.recorder.start();
                this.recordBtn.innerHTML = '<span class="icon">&#9632;</span> Stop';
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
            } catch (error) {
                this.showStatus('Failed to access microphone: ' + error.message, 'error');
            }
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

            // Use streaming endpoint with progress updates
            const results = await this.api.analyzeWithProgress(file, (step, label, progress) => {
                this.updateProgress(step, label, progress);
            });

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
                this.starBtn.innerHTML = metadata.starred ? '&#9733;' : '&#9734;';
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

            this.starBtn.innerHTML = newStarred ? '&#9733;' : '&#9734;';
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

        // Original step labels
        const originalLabels = {
            'convert': 'Converting audio',
            'transcribe': 'Transcribing with Whisper',
            'diarize': 'Identifying speakers',
            'roles': 'Inferring roles',
            'quality': 'Analyzing communication quality',
            'scorecards': 'Generating scorecards',
            'confidence': 'Analyzing confidence',
            'learning': 'Evaluating learning metrics',
            'habits': 'Analyzing 7 Habits',
            'training': 'Generating training recommendations'
        };

        // Reset all steps
        document.querySelectorAll('#progress-steps .step').forEach(step => {
            step.classList.remove('active', 'completed');
            const icon = step.querySelector('.step-icon');
            const stepLabel = step.querySelector('.step-label');
            const stepId = step.dataset.step;
            if (icon) icon.innerHTML = '&#9675;'; // Empty circle
            // Restore original label
            if (stepLabel && originalLabels[stepId]) {
                stepLabel.textContent = originalLabels[stepId];
            }
        });
    }

    updateProgress(stepId, label, progress) {
        // Update progress bar
        const progressFill = document.getElementById('progress-fill');
        const progressPercent = document.getElementById('progress-percent');
        if (progressFill) progressFill.style.width = `${progress}%`;
        if (progressPercent) progressPercent.textContent = `${progress}%`;

        // Update step states
        const steps = document.querySelectorAll('#progress-steps .step');
        let foundCurrent = false;

        steps.forEach(step => {
            const currentStepId = step.dataset.step;
            const icon = step.querySelector('.step-icon');
            const stepLabel = step.querySelector('.step-label');

            if (currentStepId === stepId) {
                // This is the current active step
                step.classList.add('active');
                step.classList.remove('completed');
                if (icon) icon.innerHTML = '&#9679;'; // Filled circle
                // Update label with dynamic text (e.g., "Transcribing... 5.2s / 20.5s")
                if (stepLabel && label) stepLabel.textContent = label;
                foundCurrent = true;
            } else if (!foundCurrent) {
                // Steps before current are completed
                step.classList.remove('active');
                step.classList.add('completed');
                if (icon) icon.innerHTML = '&#10003;'; // Checkmark
            } else {
                // Steps after current are pending
                step.classList.remove('active', 'completed');
                if (icon) icon.innerHTML = '&#9675;'; // Empty circle
            }
        });
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
                            ${isStarred ? '<span class="star-indicator">&#9733;</span>' : ''}
                            <span class="archive-item-title">${this.escapeHtml(title)}</span>
                        </div>
                        <div class="archive-item-meta">
                            <span>${dateStr} ${timeStr}</span>
                            <span>${duration}</span>
                            <span>${analysis.speaker_count || 0} speakers</span>
                        </div>
                    </div>
                    <div class="archive-item-actions">
                        ${analysis.recording_filename ? `
                            <button class="btn-icon" onclick="event.stopPropagation(); app.downloadArchivedAudio('${analysis.recording_filename}')" title="Download Audio">
                                &#127911;
                            </button>
                        ` : ''}
                        <button class="btn-icon delete" onclick="event.stopPropagation(); app.deleteAnalysis('${analysis.filename}')" title="Delete">
                            &#128465;
                        </button>
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
        try {
            this.showStatus('Loading analysis...', 'info');
            const data = await this.api.getAnalysis(filename);

            if (data && data.results) {
                this.currentResults = data.results;
                this.currentFilename = filename;
                this.savedRecordingPath = null;

                // Check for linked recording
                if (data.metadata && data.metadata.recording_file) {
                    this.savedRecordingPath = `data/recordings/${data.metadata.recording_file}`;
                    this.downloadAudioBtn.classList.remove('hidden');
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

    formatDuration(seconds) {
        if (!seconds) return '0:00';
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
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
