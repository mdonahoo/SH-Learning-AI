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
                    <ul class="pattern-list">
                        ${quality.patterns.map(p => `
                            <li class="pattern-item">
                                <span class="pattern-name">${this.formatPatternName(p.pattern_name)}</span>
                                <span class="pattern-count ${p.category}">${p.count}</span>
                            </li>
                        `).join('')}
                    </ul>
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
                            <span class="score-name">${m.name}</span>
                            <div class="score-value">
                                <div class="score-dots">
                                    ${[1,2,3,4,5].map(i => `
                                        <span class="score-dot ${i <= m.score ? 'filled score-' + m.score : ''}"></span>
                                    `).join('')}
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>

                ${(card.strengths?.length > 0 || card.areas_for_improvement?.length > 0) ? `
                    <div class="strengths-weaknesses">
                        ${card.strengths?.length > 0 ? `
                            <div class="strengths">
                                <h4>Strengths</h4>
                                <ul>${card.strengths.map(s => `<li>${s}</li>`).join('')}</ul>
                            </div>
                        ` : ''}
                        ${card.areas_for_improvement?.length > 0 ? `
                            <div class="weaknesses">
                                <h4>Areas for Improvement</h4>
                                <ul>${card.areas_for_improvement.map(a => `<li>${a}</li>`).join('')}</ul>
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

        // Speaker Scorecards
        if (results.speaker_scorecards && results.speaker_scorecards.length > 0) {
            md += `## Speaker Scorecards\n\n`;

            for (const card of results.speaker_scorecards) {
                md += `### ${card.speaker_id}${card.role ? ` (${card.role})` : ''}\n\n`;
                md += `**Overall Score:** ${card.overall_score.toFixed(1)}/5\n\n`;

                if (card.metrics && card.metrics.length > 0) {
                    md += `| Metric | Score |\n`;
                    md += `|--------|-------|\n`;
                    for (const m of card.metrics) {
                        md += `| ${m.name} | ${'★'.repeat(m.score)}${'☆'.repeat(5 - m.score)} (${m.score}/5) |\n`;
                    }
                    md += `\n`;
                }

                if (card.strengths && card.strengths.length > 0) {
                    md += `**Strengths:**\n`;
                    for (const s of card.strengths) {
                        md += `- ${s}\n`;
                    }
                    md += `\n`;
                }

                if (card.areas_for_improvement && card.areas_for_improvement.length > 0) {
                    md += `**Areas for Improvement:**\n`;
                    for (const a of card.areas_for_improvement) {
                        md += `- ${a}\n`;
                    }
                    md += `\n`;
                }
            }
        }

        // Communication Quality
        if (results.communication_quality) {
            const q = results.communication_quality;
            md += `## Communication Quality\n\n`;
            md += `- **Effective Communications:** ${q.effective_count}\n`;
            md += `- **Needs Improvement:** ${q.improvement_count}\n`;
            md += `- **Effective Rate:** ${q.effective_percentage.toFixed(0)}%\n\n`;

            if (q.patterns && q.patterns.length > 0) {
                md += `### Detected Patterns\n\n`;
                md += `| Pattern | Category | Count |\n`;
                md += `|---------|----------|-------|\n`;
                for (const p of q.patterns) {
                    md += `| ${this.formatPatternName(p.pattern_name)} | ${p.category} | ${p.count} |\n`;
                }
                md += `\n`;
            }
        }

        // Confidence Distribution
        if (results.confidence_distribution) {
            const c = results.confidence_distribution;
            md += `## Confidence Analysis\n\n`;
            md += `- **Average Confidence:** ${(c.average_confidence * 100).toFixed(0)}%\n`;
            md += `- **Median Confidence:** ${(c.median_confidence * 100).toFixed(0)}%\n\n`;

            if (c.buckets && c.buckets.length > 0) {
                md += `| Tier | Count | Percentage |\n`;
                md += `|------|-------|------------|\n`;
                for (const b of c.buckets) {
                    md += `| ${b.label} | ${b.count} | ${b.percentage.toFixed(1)}% |\n`;
                }
                md += `\n`;
            }
        }

        // Learning Evaluation
        if (results.learning_evaluation) {
            const learn = results.learning_evaluation;
            md += `## Learning Evaluation\n\n`;

            if (learn.kirkpatrick_levels && learn.kirkpatrick_levels.length > 0) {
                md += `### Kirkpatrick 4-Level Model\n\n`;
                md += `| Level | Name | Score |\n`;
                md += `|-------|------|-------|\n`;
                for (const lvl of learn.kirkpatrick_levels) {
                    md += `| ${lvl.level} | ${lvl.name} | ${(lvl.score * 100).toFixed(0)}% |\n`;
                }
                md += `\n`;
            }

            md += `### Learning Frameworks\n\n`;
            if (learn.blooms_level) {
                md += `- **Bloom's Taxonomy Level:** ${learn.blooms_level}\n`;
            }
            if (learn.nasa_tlx_score !== undefined) {
                md += `- **NASA TLX Workload:** ${(learn.nasa_tlx_score * 100).toFixed(0)}%\n`;
            }
            if (learn.engagement_score !== undefined) {
                md += `- **Engagement Score:** ${(learn.engagement_score * 100).toFixed(0)}%\n`;
            }
            md += `\n`;

            if (learn.recommendations && learn.recommendations.length > 0) {
                md += `### Recommendations\n\n`;
                for (const r of learn.recommendations) {
                    md += `- ${r}\n`;
                }
                md += `\n`;
            }
        }

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
        md += `*Generated by Starship Horizons Audio Analyzer*\n`;

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

        this.initElements();
        this.bindEvents();
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
    }

    bindEvents() {
        // Record button
        this.recordBtn.addEventListener('click', () => this.toggleRecording());

        // Upload button
        this.uploadBtn.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Analyze button
        this.analyzeBtn.addEventListener('click', () => this.analyzeAudio());

        // Download button
        this.downloadBtn.addEventListener('click', () => this.downloadReport());

        // Tabs
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });
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
        this.resultsSection.classList.add('hidden');
    }

    async analyzeAudio() {
        if (!this.currentBlob) {
            this.showStatus('No audio to analyze', 'error');
            return;
        }

        this.analyzeBtn.disabled = true;
        this.processing.classList.remove('hidden');
        this.resultsSection.classList.add('hidden');
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
            this.renderer.render(results);
            this.resultsSection.classList.remove('hidden');
            this.showStatus('Analysis complete!', 'success');
        } catch (error) {
            console.error('Analysis failed:', error);
            this.showStatus('Analysis failed: ' + error.message, 'error');
        } finally {
            this.analyzeBtn.disabled = false;
            this.processing.classList.add('hidden');
        }
    }

    resetProgress() {
        // Reset progress bar
        const progressFill = document.getElementById('progress-fill');
        const progressPercent = document.getElementById('progress-percent');
        if (progressFill) progressFill.style.width = '0%';
        if (progressPercent) progressPercent.textContent = '0%';

        // Reset all steps
        document.querySelectorAll('#progress-steps .step').forEach(step => {
            step.classList.remove('active', 'completed');
            const icon = step.querySelector('.step-icon');
            if (icon) icon.innerHTML = '&#9675;'; // Empty circle
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

            if (currentStepId === stepId) {
                // This is the current active step
                step.classList.add('active');
                step.classList.remove('completed');
                if (icon) icon.innerHTML = '&#9679;'; // Filled circle
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
