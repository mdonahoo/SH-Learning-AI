"""
Workspace manager for multi-user session isolation.

Manages per-workspace directories and caches ArchiveManager instances
so that multiple simultaneous users can upload, record, and analyze
missions independently without authentication.
"""

import logging
import os
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.web.archive_manager import ArchiveManager

load_dotenv()

logger = logging.getLogger(__name__)

# UUID v4 pattern for workspace ID validation
_UUID_PATTERN = re.compile(
    r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
)


class WorkspaceManager:
    """
    Manages per-workspace directories and ArchiveManager instances.

    Each workspace is identified by a UUID and gets its own directory
    tree under ``base_dir`` for analyses, recordings, and telemetry.

    Pre-existing data in the global ``data/`` directory is exposed as
    read-only shared content visible to all workspaces.

    Attributes:
        base_dir: Root directory for all workspaces.
        shared_data_dir: Root directory for shared (pre-existing) data.
    """

    def __init__(
        self,
        base_dir: Optional[str] = None,
        shared_data_dir: Optional[str] = None
    ):
        """
        Initialize workspace manager.

        Args:
            base_dir: Root directory for workspaces. Defaults to
                      WORKSPACE_BASE_DIR env var or ``data/workspaces``.
            shared_data_dir: Root directory for shared pre-existing data.
                             Defaults to SHARED_DATA_DIR env var or ``data``.
        """
        self.base_dir = Path(
            base_dir or os.getenv('WORKSPACE_BASE_DIR', 'data/workspaces')
        )
        self.shared_data_dir = Path(
            shared_data_dir or os.getenv('SHARED_DATA_DIR', 'data')
        )
        self._archive_managers: Dict[str, ArchiveManager] = {}
        self._shared_archive_manager: Optional[ArchiveManager] = None
        self._lock = threading.Lock()
        logger.info(
            f"WorkspaceManager initialized: base_dir={self.base_dir}, "
            f"shared_data_dir={self.shared_data_dir}"
        )

    def validate_workspace_id(self, workspace_id: str) -> bool:
        """
        Validate that a workspace ID is a well-formed UUID v4.

        Prevents path traversal and other injection attacks.

        Args:
            workspace_id: The workspace identifier to validate.

        Returns:
            True if the ID matches UUID v4 format, False otherwise.
        """
        return bool(_UUID_PATTERN.match(workspace_id))

    def ensure_workspace(self, workspace_id: str) -> Dict[str, Path]:
        """
        Create workspace directories if they don't exist.

        Args:
            workspace_id: A validated UUID workspace identifier.

        Returns:
            Dictionary mapping directory names to their Path objects.
        """
        ws_dir = self.base_dir / workspace_id
        dirs = {
            'base': ws_dir,
            'analyses': ws_dir / 'analyses',
            'recordings': ws_dir / 'recordings',
            'telemetry': ws_dir / 'telemetry',
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return dirs

    def get_archive_manager(self, workspace_id: str) -> ArchiveManager:
        """
        Get or create a workspace-scoped ArchiveManager (thread-safe, cached).

        Args:
            workspace_id: A validated UUID workspace identifier.

        Returns:
            ArchiveManager instance scoped to the workspace.
        """
        with self._lock:
            if workspace_id not in self._archive_managers:
                dirs = self.ensure_workspace(workspace_id)
                mgr = ArchiveManager(
                    index_path=dirs['base'] / 'archive_index.json',
                    analyses_dir=dirs['analyses']
                )
                mgr.sync_with_filesystem()
                self._archive_managers[workspace_id] = mgr
            return self._archive_managers[workspace_id]

    def cleanup_workspace_cache(self, workspace_id: str) -> None:
        """
        Remove a workspace's ArchiveManager from the in-memory cache.

        Args:
            workspace_id: The workspace identifier to evict.
        """
        with self._lock:
            self._archive_managers.pop(workspace_id, None)

    @property
    def shared_analyses_dir(self) -> Path:
        """Path to the shared (global) analyses directory."""
        return self.shared_data_dir / 'analyses'

    @property
    def shared_recordings_dir(self) -> Path:
        """Path to the shared (global) recordings directory."""
        return self.shared_data_dir / 'recordings'

    @property
    def shared_telemetry_dir(self) -> Path:
        """Path to the shared (global) telemetry directory."""
        return self.shared_data_dir / 'telemetry'

    def get_shared_archive_manager(self) -> ArchiveManager:
        """
        Get or create the shared (global) ArchiveManager (thread-safe, cached).

        This manages the pre-existing analyses in the global data directory,
        making them visible as read-only content to all workspaces.

        Returns:
            ArchiveManager instance for shared analyses.
        """
        with self._lock:
            if self._shared_archive_manager is None:
                index_path = self.shared_data_dir / 'archive_index.json'
                analyses_dir = self.shared_analyses_dir
                if analyses_dir.exists():
                    self._shared_archive_manager = ArchiveManager(
                        index_path=index_path,
                        analyses_dir=analyses_dir
                    )
                    self._shared_archive_manager.sync_with_filesystem()
                    logger.info(
                        f"Shared archive manager initialized: "
                        f"{len(self._shared_archive_manager.list_analyses())} analyses"
                    )
                else:
                    # No shared data directory — return an empty manager
                    self._shared_archive_manager = ArchiveManager(
                        index_path=index_path,
                        analyses_dir=analyses_dir
                    )
            return self._shared_archive_manager

    # ==================================================================
    # Admin / cross-workspace methods
    # ==================================================================

    def list_workspaces(self) -> List[Dict[str, Any]]:
        """
        List all workspaces with summary statistics.

        Walks ``self.base_dir``, filters directories matching UUID pattern,
        and collects per-workspace stats.

        Returns:
            List of workspace info dicts sorted by last_activity descending.
        """
        workspaces: List[Dict[str, Any]] = []
        if not self.base_dir.exists():
            return workspaces

        for entry in self.base_dir.iterdir():
            if not entry.is_dir():
                continue
            if not _UUID_PATTERN.match(entry.name):
                continue

            ws_id = entry.name
            analyses_dir = entry / 'analyses'
            recordings_dir = entry / 'recordings'
            telemetry_dir = entry / 'telemetry'

            analysis_count = len(list(analyses_dir.glob('*.json'))) if analyses_dir.exists() else 0
            recording_count = len(list(recordings_dir.glob('*'))) if recordings_dir.exists() else 0
            telemetry_count = len(list(telemetry_dir.glob('*.json'))) if telemetry_dir.exists() else 0

            # Disk usage and last activity
            disk_usage = 0
            last_mtime = 0.0
            for f in entry.rglob('*'):
                if f.is_file():
                    try:
                        stat = f.stat()
                        disk_usage += stat.st_size
                        if stat.st_mtime > last_mtime:
                            last_mtime = stat.st_mtime
                    except OSError:
                        continue

            try:
                created_ts = entry.stat().st_ctime
            except OSError:
                created_ts = 0.0

            workspaces.append({
                'workspace_id': ws_id,
                'created_at': datetime.fromtimestamp(created_ts, tz=timezone.utc).isoformat(),
                'last_activity': datetime.fromtimestamp(last_mtime, tz=timezone.utc).isoformat() if last_mtime else None,
                'analysis_count': analysis_count,
                'recording_count': recording_count,
                'telemetry_count': telemetry_count,
                'disk_usage_bytes': disk_usage,
            })

        workspaces.sort(key=lambda w: w.get('last_activity') or '', reverse=True)
        return workspaces

    def get_workspace_stats(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed statistics for a single workspace.

        Returns per-subdirectory file lists with sizes.

        Args:
            workspace_id: A validated UUID workspace identifier.

        Returns:
            Detailed stats dict, or None if the workspace does not exist.
        """
        ws_dir = self.base_dir / workspace_id
        if not ws_dir.exists():
            return None

        subdirs = ['analyses', 'recordings', 'telemetry']
        result: Dict[str, Any] = {
            'workspace_id': workspace_id,
            'subdirectories': {},
            'total_disk_usage_bytes': 0,
        }

        for subdir in subdirs:
            sub_path = ws_dir / subdir
            files: List[Dict[str, Any]] = []
            subdir_size = 0
            if sub_path.exists():
                for f in sorted(sub_path.iterdir()):
                    if f.is_file():
                        try:
                            size = f.stat().st_size
                        except OSError:
                            size = 0
                        files.append({
                            'filename': f.name,
                            'size_bytes': size,
                            'modified': datetime.fromtimestamp(
                                f.stat().st_mtime, tz=timezone.utc
                            ).isoformat(),
                        })
                        subdir_size += size

            result['subdirectories'][subdir] = {
                'file_count': len(files),
                'disk_usage_bytes': subdir_size,
                'files': files,
            }
            result['total_disk_usage_bytes'] += subdir_size

        return result

    def delete_workspace(self, workspace_id: str) -> bool:
        """
        Delete an entire workspace directory and evict its cache.

        Args:
            workspace_id: A validated UUID workspace identifier.

        Returns:
            True if the workspace was deleted, False if it did not exist.
        """
        ws_dir = self.base_dir / workspace_id
        if not ws_dir.exists():
            return False

        try:
            shutil.rmtree(ws_dir)
            self.cleanup_workspace_cache(workspace_id)
            logger.info(f"Deleted workspace: {workspace_id}")
            return True
        except OSError as e:
            logger.error(f"Failed to delete workspace {workspace_id}: {e}")
            return False

    def get_global_stats(self) -> Dict[str, Any]:
        """
        Aggregate statistics across all workspaces and shared data.

        Returns:
            Dict with total counts and disk usage for workspaces and shared data.
        """
        workspaces = self.list_workspaces()
        total_analyses = sum(w['analysis_count'] for w in workspaces)
        total_recordings = sum(w['recording_count'] for w in workspaces)
        total_telemetry = sum(w['telemetry_count'] for w in workspaces)
        total_disk = sum(w['disk_usage_bytes'] for w in workspaces)

        # Shared data stats
        shared_stats: Dict[str, Any] = {
            'analyses': {'count': 0, 'disk_usage_bytes': 0},
            'recordings': {'count': 0, 'disk_usage_bytes': 0},
            'telemetry': {'count': 0, 'disk_usage_bytes': 0},
        }

        for key, dir_path in [
            ('analyses', self.shared_analyses_dir),
            ('recordings', self.shared_recordings_dir),
            ('telemetry', self.shared_telemetry_dir),
        ]:
            if dir_path.exists():
                for f in dir_path.iterdir():
                    if f.is_file():
                        try:
                            shared_stats[key]['count'] += 1
                            shared_stats[key]['disk_usage_bytes'] += f.stat().st_size
                        except OSError:
                            continue

        shared_disk = sum(s['disk_usage_bytes'] for s in shared_stats.values())

        return {
            'workspace_count': len(workspaces),
            'total_analyses': total_analyses,
            'total_recordings': total_recordings,
            'total_telemetry': total_telemetry,
            'total_disk_usage_bytes': total_disk + shared_disk,
            'workspace_disk_usage_bytes': total_disk,
            'shared_data': shared_stats,
            'shared_disk_usage_bytes': shared_disk,
        }
