"""Tests for workspace manager module."""

import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.web.workspace_manager import WorkspaceManager


class TestValidateWorkspaceId:
    """Test suite for workspace ID validation."""

    def test_valid_uuid(self):
        """Test that valid UUID v4 strings are accepted."""
        mgr = WorkspaceManager(base_dir="/tmp/test_workspaces")
        assert mgr.validate_workspace_id("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        assert mgr.validate_workspace_id("00000000-0000-0000-0000-000000000000")
        assert mgr.validate_workspace_id("ffffffff-ffff-ffff-ffff-ffffffffffff")

    def test_invalid_uuid_rejected(self):
        """Test that non-UUID strings are rejected."""
        mgr = WorkspaceManager(base_dir="/tmp/test_workspaces")
        assert not mgr.validate_workspace_id("")
        assert not mgr.validate_workspace_id("not-a-uuid")
        assert not mgr.validate_workspace_id("12345")
        assert not mgr.validate_workspace_id("a1b2c3d4-e5f6-7890-abcd")

    def test_path_traversal_rejected(self):
        """Test that path traversal attempts are rejected."""
        mgr = WorkspaceManager(base_dir="/tmp/test_workspaces")
        assert not mgr.validate_workspace_id("../../../etc/passwd")
        assert not mgr.validate_workspace_id("..%2F..%2Fetc%2Fpasswd")
        assert not mgr.validate_workspace_id("a1b2c3d4/../../../etc")

    def test_uppercase_hex_rejected(self):
        """Test that uppercase hex characters are rejected."""
        mgr = WorkspaceManager(base_dir="/tmp/test_workspaces")
        assert not mgr.validate_workspace_id("A1B2C3D4-E5F6-7890-ABCD-EF1234567890")


class TestEnsureWorkspace:
    """Test suite for workspace directory creation."""

    def test_creates_directories(self, tmp_path):
        """Test that all workspace subdirectories are created."""
        mgr = WorkspaceManager(base_dir=str(tmp_path))
        workspace_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

        dirs = mgr.ensure_workspace(workspace_id)

        assert dirs['base'].exists()
        assert dirs['analyses'].exists()
        assert dirs['recordings'].exists()
        assert dirs['telemetry'].exists()

    def test_idempotent(self, tmp_path):
        """Test that calling ensure_workspace twice doesn't error."""
        mgr = WorkspaceManager(base_dir=str(tmp_path))
        workspace_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

        dirs1 = mgr.ensure_workspace(workspace_id)
        dirs2 = mgr.ensure_workspace(workspace_id)

        assert dirs1 == dirs2
        assert dirs1['base'].exists()

    def test_returns_correct_paths(self, tmp_path):
        """Test that returned paths match expected structure."""
        mgr = WorkspaceManager(base_dir=str(tmp_path))
        workspace_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

        dirs = mgr.ensure_workspace(workspace_id)

        assert dirs['base'] == tmp_path / workspace_id
        assert dirs['analyses'] == tmp_path / workspace_id / 'analyses'
        assert dirs['recordings'] == tmp_path / workspace_id / 'recordings'
        assert dirs['telemetry'] == tmp_path / workspace_id / 'telemetry'


class TestGetArchiveManager:
    """Test suite for archive manager caching."""

    def test_returns_archive_manager(self, tmp_path):
        """Test that an ArchiveManager instance is returned."""
        mgr = WorkspaceManager(base_dir=str(tmp_path))
        workspace_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

        archive_mgr = mgr.get_archive_manager(workspace_id)

        from src.web.archive_manager import ArchiveManager
        assert isinstance(archive_mgr, ArchiveManager)

    def test_cached_same_instance(self, tmp_path):
        """Test that same instance is returned for same workspace."""
        mgr = WorkspaceManager(base_dir=str(tmp_path))
        workspace_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

        mgr1 = mgr.get_archive_manager(workspace_id)
        mgr2 = mgr.get_archive_manager(workspace_id)

        assert mgr1 is mgr2

    def test_isolation_different_workspaces(self, tmp_path):
        """Test that different workspaces get different instances."""
        mgr = WorkspaceManager(base_dir=str(tmp_path))
        id_a = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        id_b = "b1b2c3d4-e5f6-7890-abcd-ef1234567890"

        mgr_a = mgr.get_archive_manager(id_a)
        mgr_b = mgr.get_archive_manager(id_b)

        assert mgr_a is not mgr_b


class TestCleanupWorkspaceCache:
    """Test suite for cache eviction."""

    def test_removes_cached_instance(self, tmp_path):
        """Test that cleanup removes the cached ArchiveManager."""
        mgr = WorkspaceManager(base_dir=str(tmp_path))
        workspace_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

        mgr1 = mgr.get_archive_manager(workspace_id)
        mgr.cleanup_workspace_cache(workspace_id)
        mgr2 = mgr.get_archive_manager(workspace_id)

        # After cleanup, a new instance should be created
        assert mgr1 is not mgr2

    def test_cleanup_nonexistent_workspace(self, tmp_path):
        """Test that cleaning up a non-existent workspace doesn't error."""
        mgr = WorkspaceManager(base_dir=str(tmp_path))
        # Should not raise
        mgr.cleanup_workspace_cache("00000000-0000-0000-0000-000000000000")


class TestSharedDataAccess:
    """Test suite for shared (global) data visibility."""

    def test_shared_directory_properties(self, tmp_path):
        """Test that shared directory paths are derived from shared_data_dir."""
        shared_dir = tmp_path / "shared_data"
        shared_dir.mkdir()
        mgr = WorkspaceManager(
            base_dir=str(tmp_path / "workspaces"),
            shared_data_dir=str(shared_dir)
        )

        assert mgr.shared_analyses_dir == shared_dir / 'analyses'
        assert mgr.shared_recordings_dir == shared_dir / 'recordings'
        assert mgr.shared_telemetry_dir == shared_dir / 'telemetry'

    def test_shared_archive_manager_returned(self, tmp_path):
        """Test that a shared ArchiveManager is created and cached."""
        shared_dir = tmp_path / "shared_data"
        (shared_dir / "analyses").mkdir(parents=True)

        mgr = WorkspaceManager(
            base_dir=str(tmp_path / "workspaces"),
            shared_data_dir=str(shared_dir)
        )

        from src.web.archive_manager import ArchiveManager
        shared_mgr = mgr.get_shared_archive_manager()
        assert isinstance(shared_mgr, ArchiveManager)

    def test_shared_archive_manager_cached(self, tmp_path):
        """Test that shared ArchiveManager is the same instance on repeat call."""
        shared_dir = tmp_path / "shared_data"
        (shared_dir / "analyses").mkdir(parents=True)

        mgr = WorkspaceManager(
            base_dir=str(tmp_path / "workspaces"),
            shared_data_dir=str(shared_dir)
        )

        mgr1 = mgr.get_shared_archive_manager()
        mgr2 = mgr.get_shared_archive_manager()
        assert mgr1 is mgr2

    def test_shared_archive_manager_sees_existing_files(self, tmp_path):
        """Test that shared ArchiveManager detects pre-existing analysis files."""
        shared_dir = tmp_path / "shared_data"
        analyses_dir = shared_dir / "analyses"
        analyses_dir.mkdir(parents=True)

        # Create a fake analysis file
        analysis = {
            "metadata": {
                "created_at": "2026-01-28T06:00:00",
                "duration_seconds": 120,
                "speaker_count": 3,
                "segment_count": 10,
                "auto_title": "Test Mission"
            },
            "results": {"full_text": "hello world"}
        }
        with open(analyses_dir / "analysis_20260128_060000.json", "w") as f:
            json.dump(analysis, f)

        mgr = WorkspaceManager(
            base_dir=str(tmp_path / "workspaces"),
            shared_data_dir=str(shared_dir)
        )

        shared_mgr = mgr.get_shared_archive_manager()
        all_analyses = shared_mgr.list_analyses()
        filenames = [a.filename for a in all_analyses]
        assert "analysis_20260128_060000.json" in filenames

    def test_shared_separate_from_workspace(self, tmp_path):
        """Test that shared and workspace ArchiveManagers are independent."""
        shared_dir = tmp_path / "shared_data"
        (shared_dir / "analyses").mkdir(parents=True)

        mgr = WorkspaceManager(
            base_dir=str(tmp_path / "workspaces"),
            shared_data_dir=str(shared_dir)
        )

        workspace_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        ws_mgr = mgr.get_archive_manager(workspace_id)
        shared_mgr = mgr.get_shared_archive_manager()

        assert ws_mgr is not shared_mgr


class TestListWorkspaces:
    """Test suite for list_workspaces admin method."""

    def test_empty_base_dir(self, tmp_path):
        """Test that empty base_dir returns empty list."""
        mgr = WorkspaceManager(base_dir=str(tmp_path / "workspaces"))
        result = mgr.list_workspaces()
        assert result == []

    def test_lists_uuid_directories(self, tmp_path):
        """Test that only UUID-named directories are returned."""
        base = tmp_path / "workspaces"
        ws_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        (base / ws_id / "analyses").mkdir(parents=True)
        (base / "not-a-uuid").mkdir(parents=True)
        (base / "random_file.txt").touch()

        mgr = WorkspaceManager(base_dir=str(base))
        result = mgr.list_workspaces()

        assert len(result) == 1
        assert result[0]['workspace_id'] == ws_id

    def test_counts_files(self, tmp_path):
        """Test that analysis, recording, and telemetry counts are correct."""
        base = tmp_path / "workspaces"
        ws_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        analyses_dir = base / ws_id / "analyses"
        recordings_dir = base / ws_id / "recordings"
        telemetry_dir = base / ws_id / "telemetry"
        analyses_dir.mkdir(parents=True)
        recordings_dir.mkdir(parents=True)
        telemetry_dir.mkdir(parents=True)

        (analyses_dir / "a1.json").write_text("{}")
        (analyses_dir / "a2.json").write_text("{}")
        (recordings_dir / "rec.wav").write_bytes(b"fake")
        (telemetry_dir / "tel.json").write_text("{}")

        mgr = WorkspaceManager(base_dir=str(base))
        result = mgr.list_workspaces()

        assert result[0]['analysis_count'] == 2
        assert result[0]['recording_count'] == 1
        assert result[0]['telemetry_count'] == 1
        assert result[0]['disk_usage_bytes'] > 0


class TestGetWorkspaceStats:
    """Test suite for get_workspace_stats admin method."""

    def test_nonexistent_workspace(self, tmp_path):
        """Test that None is returned for missing workspace."""
        mgr = WorkspaceManager(base_dir=str(tmp_path))
        result = mgr.get_workspace_stats("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        assert result is None

    def test_returns_file_details(self, tmp_path):
        """Test that detailed file info is returned per subdirectory."""
        base = tmp_path / "workspaces"
        ws_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        analyses_dir = base / ws_id / "analyses"
        analyses_dir.mkdir(parents=True)
        (base / ws_id / "recordings").mkdir()
        (base / ws_id / "telemetry").mkdir()
        (analyses_dir / "test.json").write_text('{"data": 1}')

        mgr = WorkspaceManager(base_dir=str(base))
        result = mgr.get_workspace_stats(ws_id)

        assert result is not None
        assert result['workspace_id'] == ws_id
        assert result['subdirectories']['analyses']['file_count'] == 1
        assert result['subdirectories']['analyses']['files'][0]['filename'] == 'test.json'
        assert result['total_disk_usage_bytes'] > 0


class TestDeleteWorkspace:
    """Test suite for delete_workspace admin method."""

    def test_deletes_existing_workspace(self, tmp_path):
        """Test that workspace directory is removed."""
        base = tmp_path / "workspaces"
        ws_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        ws_dir = base / ws_id / "analyses"
        ws_dir.mkdir(parents=True)
        (ws_dir / "data.json").write_text("{}")

        mgr = WorkspaceManager(base_dir=str(base))
        # Pre-cache an archive manager
        mgr.get_archive_manager(ws_id)

        result = mgr.delete_workspace(ws_id)

        assert result is True
        assert not (base / ws_id).exists()
        # Cache should be cleaned
        assert ws_id not in mgr._archive_managers

    def test_delete_nonexistent_returns_false(self, tmp_path):
        """Test that deleting a missing workspace returns False."""
        mgr = WorkspaceManager(base_dir=str(tmp_path))
        result = mgr.delete_workspace("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
        assert result is False


class TestGetGlobalStats:
    """Test suite for get_global_stats admin method."""

    def test_empty_returns_zeros(self, tmp_path):
        """Test that empty system returns zero counts."""
        mgr = WorkspaceManager(
            base_dir=str(tmp_path / "workspaces"),
            shared_data_dir=str(tmp_path / "shared")
        )
        result = mgr.get_global_stats()

        assert result['workspace_count'] == 0
        assert result['total_analyses'] == 0
        assert result['total_recordings'] == 0
        assert result['total_disk_usage_bytes'] == 0

    def test_aggregates_across_workspaces(self, tmp_path):
        """Test that stats are summed across multiple workspaces."""
        base = tmp_path / "workspaces"
        shared = tmp_path / "shared"
        shared_analyses = shared / "analyses"
        shared_analyses.mkdir(parents=True)
        (shared_analyses / "shared.json").write_text('{"x": 1}')

        ws_id1 = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        ws_id2 = "b1b2c3d4-e5f6-7890-abcd-ef1234567890"

        for ws_id in [ws_id1, ws_id2]:
            a_dir = base / ws_id / "analyses"
            a_dir.mkdir(parents=True)
            (base / ws_id / "recordings").mkdir()
            (base / ws_id / "telemetry").mkdir()
            (a_dir / "data.json").write_text("{}")

        mgr = WorkspaceManager(base_dir=str(base), shared_data_dir=str(shared))
        result = mgr.get_global_stats()

        assert result['workspace_count'] == 2
        assert result['total_analyses'] == 2
        assert result['shared_data']['analyses']['count'] == 1
        assert result['total_disk_usage_bytes'] > 0
