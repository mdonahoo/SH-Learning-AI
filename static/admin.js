/**
 * Admin Panel - Workspace Management Console
 *
 * Provides cross-workspace visibility: list workspaces, browse files,
 * view/delete analyses, download recordings, see disk usage, and
 * delete entire workspaces.
 */

// ============================================================================
// API Client
// ============================================================================

class AdminApiClient {
    /**
     * Fetch wrapper for /api/admin/* endpoints.
     * No X-Workspace-ID header required.
     */

    async getGlobalStats() {
        const resp = await fetch('/api/admin/stats');
        if (!resp.ok) throw new Error(`Stats failed: ${resp.status}`);
        return resp.json();
    }

    async listWorkspaces() {
        const resp = await fetch('/api/admin/workspaces');
        if (!resp.ok) throw new Error(`List failed: ${resp.status}`);
        return resp.json();
    }

    async getWorkspace(workspaceId) {
        const resp = await fetch(`/api/admin/workspaces/${workspaceId}`);
        if (!resp.ok) throw new Error(`Workspace not found: ${resp.status}`);
        return resp.json();
    }

    async deleteWorkspace(workspaceId) {
        const resp = await fetch(`/api/admin/workspaces/${workspaceId}`, {
            method: 'DELETE',
        });
        if (!resp.ok) throw new Error(`Delete failed: ${resp.status}`);
        return resp.json();
    }

    async getWorkspaceArchiveIndex(workspaceId) {
        const resp = await fetch(`/api/admin/workspaces/${workspaceId}/archive-index`);
        if (!resp.ok) throw new Error(`Archive index failed: ${resp.status}`);
        return resp.json();
    }

    async getAnalysis(workspaceId, filename) {
        const resp = await fetch(`/api/admin/workspaces/${workspaceId}/analyses/${filename}`);
        if (!resp.ok) throw new Error(`Analysis not found: ${resp.status}`);
        return resp.json();
    }

    async deleteAnalysis(workspaceId, filename) {
        const resp = await fetch(`/api/admin/workspaces/${workspaceId}/analyses/${filename}`, {
            method: 'DELETE',
        });
        if (!resp.ok) throw new Error(`Delete analysis failed: ${resp.status}`);
        return resp.json();
    }

    async listShared() {
        const resp = await fetch('/api/admin/shared');
        if (!resp.ok) throw new Error(`Shared list failed: ${resp.status}`);
        return resp.json();
    }
}


// ============================================================================
// Utilities
// ============================================================================

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    const val = bytes / Math.pow(1024, i);
    return `${val.toFixed(i > 0 ? 1 : 0)} ${units[i]}`;
}

function formatDate(isoString) {
    if (!isoString) return '--';
    const d = new Date(isoString);
    return d.toLocaleDateString('en-US', {
        month: 'short', day: 'numeric', year: 'numeric',
        hour: '2-digit', minute: '2-digit',
    });
}

function formatTimestamp(ts) {
    if (!ts) return '--';
    const d = new Date(ts * 1000);
    return d.toLocaleDateString('en-US', {
        month: 'short', day: 'numeric', year: 'numeric',
        hour: '2-digit', minute: '2-digit',
    });
}

function truncateUUID(uuid) {
    if (!uuid || uuid.length < 8) return uuid || '';
    return uuid.substring(0, 8) + '...';
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}


// ============================================================================
// Admin App
// ============================================================================

class AdminApp {
    constructor() {
        this.api = new AdminApiClient();
        this.currentView = 'workspaces';
        this.currentWorkspaceId = null;
        this.currentAnalysisFile = null;
    }

    async init() {
        await Promise.all([
            this.loadGlobalStats(),
            this.loadWorkspaces(),
            this.loadSharedData(),
        ]);
    }

    // --- Navigation ---

    setBreadcrumb(items) {
        const nav = document.getElementById('breadcrumb');
        nav.innerHTML = '';
        items.forEach((item, idx) => {
            if (idx > 0) {
                const sep = document.createElement('span');
                sep.className = 'breadcrumb-separator';
                sep.textContent = '>';
                nav.appendChild(sep);
            }
            const span = document.createElement('span');
            span.className = 'breadcrumb-item';
            span.textContent = item.label;
            if (idx === items.length - 1) {
                span.classList.add('active');
            } else {
                span.onclick = item.action;
            }
            nav.appendChild(span);
        });
    }

    showView(view) {
        this.currentView = view;
        const sections = {
            workspaces: ['stats-section', 'workspaces-section', 'shared-section'],
            detail: ['stats-section', 'workspace-detail-section'],
            analysis: ['analysis-viewer-section'],
        };

        // Hide all
        ['stats-section', 'workspaces-section', 'shared-section',
         'workspace-detail-section', 'analysis-viewer-section'
        ].forEach(id => {
            document.getElementById(id).classList.add('hidden');
        });

        // Show target
        (sections[view] || []).forEach(id => {
            document.getElementById(id).classList.remove('hidden');
        });
    }

    navigateToWorkspaces() {
        this.currentWorkspaceId = null;
        this.currentAnalysisFile = null;
        this.showView('workspaces');
        this.setBreadcrumb([{ label: 'Workspaces' }]);
    }

    navigateToDetail(workspaceId) {
        this.currentWorkspaceId = workspaceId;
        this.currentAnalysisFile = null;
        this.showView('detail');
        this.setBreadcrumb([
            { label: 'Workspaces', action: () => this.navigateToWorkspaces() },
            { label: truncateUUID(workspaceId) },
        ]);
        this.loadWorkspaceDetail(workspaceId);
    }

    navigateToAnalysis(workspaceId, filename) {
        this.currentWorkspaceId = workspaceId;
        this.currentAnalysisFile = filename;
        this.showView('analysis');
        this.setBreadcrumb([
            { label: 'Workspaces', action: () => this.navigateToWorkspaces() },
            { label: truncateUUID(workspaceId), action: () => this.navigateToDetail(workspaceId) },
            { label: filename },
        ]);
        this.loadAnalysisView(workspaceId, filename);
    }

    // --- Global Stats ---

    async loadGlobalStats() {
        try {
            const stats = await this.api.getGlobalStats();
            this.renderGlobalStats(stats);
        } catch (err) {
            console.error('Failed to load global stats:', err);
        }
    }

    renderGlobalStats(stats) {
        document.getElementById('stat-workspaces').textContent = stats.workspace_count;
        document.getElementById('stat-analyses').textContent = stats.total_analyses;
        document.getElementById('stat-recordings').textContent = stats.total_recordings;
        document.getElementById('stat-disk').textContent = formatBytes(stats.total_disk_usage_bytes);
    }

    // --- Workspaces Table ---

    async loadWorkspaces() {
        try {
            const data = await this.api.listWorkspaces();
            this.renderWorkspaceTable(data.workspaces);
        } catch (err) {
            console.error('Failed to load workspaces:', err);
            document.getElementById('workspaces-table').innerHTML =
                '<p class="empty-state">Failed to load workspaces</p>';
        }
    }

    renderWorkspaceTable(workspaces) {
        const container = document.getElementById('workspaces-table');

        if (!workspaces || workspaces.length === 0) {
            container.innerHTML = '<p class="empty-state">No workspaces found</p>';
            return;
        }

        let html = `
            <div class="workspace-row workspace-row-header">
                <span>Workspace ID</span>
                <span>Created</span>
                <span style="text-align:center">Analyses</span>
                <span style="text-align:center">Recordings</span>
                <span style="text-align:right">Disk</span>
                <span>Last Activity</span>
                <span></span>
            </div>
        `;

        for (const ws of workspaces) {
            html += `
                <div class="workspace-row">
                    <span class="uuid-display">${escapeHtml(truncateUUID(ws.workspace_id))}</span>
                    <span class="ws-cell-date">${formatDate(ws.created_at)}</span>
                    <span class="ws-cell-count">${ws.analysis_count}</span>
                    <span class="ws-cell-count">${ws.recording_count}</span>
                    <span class="ws-cell-size">${formatBytes(ws.disk_usage_bytes)}</span>
                    <span class="ws-cell-date">${formatDate(ws.last_activity)}</span>
                    <span class="ws-cell-actions">
                        <button class="btn-browse" onclick="adminApp.navigateToDetail('${escapeHtml(ws.workspace_id)}')">Browse</button>
                        <button class="delete-btn-danger" onclick="adminApp.deleteWorkspace('${escapeHtml(ws.workspace_id)}')">Delete</button>
                    </span>
                </div>
            `;
        }

        container.innerHTML = html;
    }

    // --- Workspace Detail ---

    async loadWorkspaceDetail(workspaceId) {
        try {
            const stats = await this.api.getWorkspace(workspaceId);
            this.renderWorkspaceDetail(stats);
        } catch (err) {
            console.error('Failed to load workspace detail:', err);
        }
    }

    renderWorkspaceDetail(stats) {
        document.getElementById('detail-workspace-id').textContent = stats.workspace_id;

        // Stat cards
        const statsContainer = document.getElementById('workspace-stats');
        const subdirs = stats.subdirectories;
        statsContainer.innerHTML = `
            <div class="stat-card">
                <div class="stat-value">${subdirs.analyses.file_count}</div>
                <div class="stat-label">Analyses</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${subdirs.recordings.file_count}</div>
                <div class="stat-label">Recordings</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${subdirs.telemetry.file_count}</div>
                <div class="stat-label">Telemetry</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${formatBytes(stats.total_disk_usage_bytes)}</div>
                <div class="stat-label">Total Size</div>
            </div>
        `;

        // File lists
        this.renderFileList('detail-analyses', subdirs.analyses.files, stats.workspace_id, 'analyses');
        this.renderFileList('detail-recordings', subdirs.recordings.files, stats.workspace_id, 'recordings');
        this.renderFileList('detail-telemetry', subdirs.telemetry.files, stats.workspace_id, 'telemetry');
    }

    renderFileList(containerId, files, workspaceId, type) {
        const container = document.getElementById(containerId);

        if (!files || files.length === 0) {
            container.innerHTML = '<p class="empty-state">No files</p>';
            return;
        }

        let html = '';
        for (const f of files) {
            let actions = '';
            if (type === 'analyses') {
                actions = `
                    <button class="btn-browse" onclick="adminApp.navigateToAnalysis('${escapeHtml(workspaceId)}', '${escapeHtml(f.filename)}')">View</button>
                    <button class="btn-browse" onclick="adminApp.downloadAnalysis('${escapeHtml(workspaceId)}', '${escapeHtml(f.filename)}')">DL</button>
                    <button class="delete-btn-danger" onclick="adminApp.deleteAnalysis('${escapeHtml(workspaceId)}', '${escapeHtml(f.filename)}')">Del</button>
                `;
            } else if (type === 'recordings') {
                actions = `
                    <button class="btn-browse" onclick="adminApp.downloadRecording('${escapeHtml(workspaceId)}', '${escapeHtml(f.filename)}')">Download</button>
                `;
            }

            html += `
                <div class="file-row">
                    <span class="file-name" title="${escapeHtml(f.filename)}">${escapeHtml(f.filename)}</span>
                    <span class="file-size">${formatBytes(f.size_bytes)}</span>
                    <span class="file-date">${formatDate(f.modified)}</span>
                    <span class="file-actions">${actions}</span>
                </div>
            `;
        }

        container.innerHTML = html;
    }

    // --- Analysis Viewer ---

    async loadAnalysisView(workspaceId, filename) {
        try {
            const data = await this.api.getAnalysis(workspaceId, filename);
            this.renderAnalysisView(data, workspaceId, filename);
        } catch (err) {
            console.error('Failed to load analysis:', err);
            document.getElementById('analysis-viewer-content').innerHTML =
                '<p class="empty-state">Failed to load analysis</p>';
        }
    }

    renderAnalysisView(data, workspaceId, filename) {
        document.getElementById('analysis-viewer-title').textContent = filename;

        // Download button
        const dlBtn = document.getElementById('analysis-download-btn');
        dlBtn.onclick = () => this.downloadAnalysis(workspaceId, filename);

        // Extract key metrics from analysis
        const results = data.results || data;
        const meta = data.metadata || {};

        const title = meta.user_title || results.auto_title || filename;
        const duration = results.duration_seconds
            ? `${Math.floor(results.duration_seconds / 60)}m ${Math.round(results.duration_seconds % 60)}s`
            : '--';
        const speakers = results.speakers ? results.speakers.length : '--';
        const procTime = results.processing_time_seconds
            ? `${results.processing_time_seconds.toFixed(1)}s`
            : '--';
        const segments = results.transcription ? results.transcription.length : '--';

        const container = document.getElementById('analysis-viewer-content');
        container.innerHTML = `
            <h3 style="color: var(--lcars-gold); margin-bottom: 16px;">${escapeHtml(title)}</h3>
            <div class="analysis-meta">
                <div class="meta-item">
                    <div class="meta-label">Duration</div>
                    <div class="meta-value">${duration}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Speakers</div>
                    <div class="meta-value">${speakers}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Segments</div>
                    <div class="meta-value">${segments}</div>
                </div>
                <div class="meta-item">
                    <div class="meta-label">Processing Time</div>
                    <div class="meta-value">${procTime}</div>
                </div>
            </div>
        `;

        // Raw JSON
        document.getElementById('analysis-raw-json').textContent =
            JSON.stringify(data, null, 2);
    }

    // --- Actions ---

    async deleteWorkspace(workspaceId) {
        if (!confirm(`Delete workspace ${truncateUUID(workspaceId)} and ALL its data? This cannot be undone.`)) {
            return;
        }
        try {
            await this.api.deleteWorkspace(workspaceId);
            await Promise.all([
                this.loadGlobalStats(),
                this.loadWorkspaces(),
            ]);
        } catch (err) {
            alert(`Failed to delete workspace: ${err.message}`);
        }
    }

    async deleteAnalysis(workspaceId, filename) {
        if (!confirm(`Delete analysis ${filename}? This cannot be undone.`)) {
            return;
        }
        try {
            await this.api.deleteAnalysis(workspaceId, filename);
            // Refresh current view
            if (this.currentView === 'detail') {
                await this.loadWorkspaceDetail(workspaceId);
            }
            await this.loadGlobalStats();
        } catch (err) {
            alert(`Failed to delete analysis: ${err.message}`);
        }
    }

    downloadAnalysis(workspaceId, filename) {
        window.open(`/api/admin/workspaces/${workspaceId}/analyses/${filename}/download`, '_blank');
    }

    downloadRecording(workspaceId, filename) {
        window.open(`/api/admin/workspaces/${workspaceId}/recordings/${filename}`, '_blank');
    }

    // --- Shared Data ---

    async loadSharedData() {
        try {
            const data = await this.api.listShared();
            this.renderSharedData(data);
        } catch (err) {
            console.error('Failed to load shared data:', err);
            document.getElementById('shared-data-content').innerHTML =
                '<p class="empty-state">Failed to load shared data</p>';
        }
    }

    renderSharedData(data) {
        const container = document.getElementById('shared-data-content');
        let html = '';

        for (const [group, files] of Object.entries(data)) {
            html += `<div class="shared-group">`;
            html += `<div class="shared-group-header">${escapeHtml(group)} (${files.length})</div>`;

            if (files.length === 0) {
                html += '<p class="empty-state">None</p>';
            } else {
                for (const f of files) {
                    html += `
                        <div class="file-row">
                            <span class="file-name" title="${escapeHtml(f.filename)}">${escapeHtml(f.filename)}</span>
                            <span class="file-size">${formatBytes(f.size_bytes)}</span>
                            <span class="file-date">${formatTimestamp(f.modified)}</span>
                            <span class="file-actions"></span>
                        </div>
                    `;
                }
            }
            html += '</div>';
        }

        container.innerHTML = html;
    }
}


// ============================================================================
// Initialize
// ============================================================================

const adminApp = new AdminApp();
document.addEventListener('DOMContentLoaded', () => {
    adminApp.init();
});
