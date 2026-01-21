"""
Archive index manager for analysis metadata.

Maintains a JSON-based index of all saved analyses with user-editable
titles, tags, notes, and other metadata.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = Path(os.getenv('DATA_DIR', 'data'))
DEFAULT_INDEX_PATH = DEFAULT_DATA_DIR / 'archive_index.json'


class ArchiveMetadata:
    """Metadata for a single archived analysis."""

    def __init__(
        self,
        filename: str,
        recording_filename: Optional[str] = None,
        auto_title: Optional[str] = None,
        user_title: Optional[str] = None,
        created_at: Optional[str] = None,
        duration_seconds: float = 0,
        speaker_count: int = 0,
        segment_count: int = 0,
        tags: Optional[List[str]] = None,
        notes: str = "",
        starred: bool = False
    ):
        """
        Initialize archive metadata.

        Args:
            filename: Analysis JSON filename
            recording_filename: Associated audio recording filename
            auto_title: LLM-generated title
            user_title: User-edited title (overrides auto_title)
            created_at: ISO timestamp of creation
            duration_seconds: Audio duration
            speaker_count: Number of speakers detected
            segment_count: Number of transcript segments
            tags: User-defined tags
            notes: User notes
            starred: Whether analysis is starred/favorited
        """
        self.filename = filename
        self.recording_filename = recording_filename
        self.auto_title = auto_title
        self.user_title = user_title
        self.created_at = created_at or datetime.now().isoformat()
        self.duration_seconds = duration_seconds
        self.speaker_count = speaker_count
        self.segment_count = segment_count
        self.tags = tags or []
        self.notes = notes
        self.starred = starred

    @property
    def display_title(self) -> str:
        """Get the title to display (user_title or auto_title or filename)."""
        if self.user_title:
            return self.user_title
        if self.auto_title:
            return self.auto_title
        return self.filename.replace('analysis_', '').replace('.json', '')

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'filename': self.filename,
            'recording_filename': self.recording_filename,
            'auto_title': self.auto_title,
            'user_title': self.user_title,
            'created_at': self.created_at,
            'duration_seconds': self.duration_seconds,
            'speaker_count': self.speaker_count,
            'segment_count': self.segment_count,
            'tags': self.tags,
            'notes': self.notes,
            'starred': self.starred,
            'display_title': self.display_title
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArchiveMetadata':
        """Create from dictionary."""
        return cls(
            filename=data.get('filename', ''),
            recording_filename=data.get('recording_filename'),
            auto_title=data.get('auto_title'),
            user_title=data.get('user_title'),
            created_at=data.get('created_at'),
            duration_seconds=data.get('duration_seconds', 0),
            speaker_count=data.get('speaker_count', 0),
            segment_count=data.get('segment_count', 0),
            tags=data.get('tags', []),
            notes=data.get('notes', ''),
            starred=data.get('starred', False)
        )


class ArchiveManager:
    """
    Manages the archive index for analysis metadata.

    Provides CRUD operations for analysis metadata and maintains
    synchronization with the filesystem.
    """

    INDEX_VERSION = 1

    def __init__(
        self,
        index_path: Optional[Path] = None,
        analyses_dir: Optional[Path] = None
    ):
        """
        Initialize archive manager.

        Args:
            index_path: Path to the archive index JSON file
            analyses_dir: Path to the analyses directory
        """
        # Convert strings to Path objects if needed
        if index_path is not None:
            self.index_path = Path(index_path) if isinstance(index_path, str) else index_path
        else:
            self.index_path = DEFAULT_INDEX_PATH

        if analyses_dir is not None:
            self.analyses_dir = Path(analyses_dir) if isinstance(analyses_dir, str) else analyses_dir
        else:
            self.analyses_dir = DEFAULT_DATA_DIR / 'analyses'

        self._index: Dict[str, ArchiveMetadata] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load the archive index from disk."""
        if not self.index_path.exists():
            logger.info("Archive index not found, will create on first save")
            return

        try:
            with open(self.index_path, 'r') as f:
                data = json.load(f)

            version = data.get('version', 1)
            if version > self.INDEX_VERSION:
                logger.warning(
                    f"Index version {version} is newer than supported {self.INDEX_VERSION}"
                )

            analyses = data.get('analyses', {})
            for analysis_id, metadata in analyses.items():
                self._index[analysis_id] = ArchiveMetadata.from_dict(metadata)

            logger.info(f"Loaded archive index with {len(self._index)} entries")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse archive index: {e}")
        except Exception as e:
            logger.error(f"Failed to load archive index: {e}")

    def _save_index(self) -> bool:
        """
        Save the archive index to disk.

        Returns:
            True if save successful, False otherwise
        """
        try:
            # Ensure parent directory exists
            self.index_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'version': self.INDEX_VERSION,
                'updated_at': datetime.now().isoformat(),
                'analyses': {
                    analysis_id: metadata.to_dict()
                    for analysis_id, metadata in self._index.items()
                }
            }

            # Write atomically with temp file
            temp_path = self.index_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)

            temp_path.replace(self.index_path)
            logger.debug(f"Saved archive index with {len(self._index)} entries")
            return True

        except Exception as e:
            logger.error(f"Failed to save archive index: {e}")
            return False

    def get_analysis_id(self, filename: str) -> str:
        """
        Get analysis ID from filename.

        Args:
            filename: Analysis filename (e.g., 'analysis_20260117_143022.json')

        Returns:
            Analysis ID (filename without extension)
        """
        return filename.replace('.json', '')

    def add_analysis(
        self,
        filename: str,
        recording_filename: Optional[str] = None,
        auto_title: Optional[str] = None,
        duration_seconds: float = 0,
        speaker_count: int = 0,
        segment_count: int = 0
    ) -> ArchiveMetadata:
        """
        Add a new analysis to the index.

        Args:
            filename: Analysis JSON filename
            recording_filename: Associated audio recording filename
            auto_title: LLM-generated title
            duration_seconds: Audio duration
            speaker_count: Number of speakers
            segment_count: Number of segments

        Returns:
            Created ArchiveMetadata
        """
        analysis_id = self.get_analysis_id(filename)

        metadata = ArchiveMetadata(
            filename=filename,
            recording_filename=recording_filename,
            auto_title=auto_title,
            duration_seconds=duration_seconds,
            speaker_count=speaker_count,
            segment_count=segment_count
        )

        self._index[analysis_id] = metadata
        self._save_index()

        logger.info(f"Added analysis to index: {analysis_id}")
        return metadata

    def get_analysis(self, filename: str) -> Optional[ArchiveMetadata]:
        """
        Get metadata for an analysis.

        Args:
            filename: Analysis filename

        Returns:
            ArchiveMetadata or None if not found
        """
        analysis_id = self.get_analysis_id(filename)
        return self._index.get(analysis_id)

    def update_analysis(
        self,
        filename: str,
        user_title: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        starred: Optional[bool] = None
    ) -> Optional[ArchiveMetadata]:
        """
        Update metadata for an analysis.

        Args:
            filename: Analysis filename
            user_title: New user-defined title
            tags: New tags list
            notes: New notes
            starred: New starred status

        Returns:
            Updated ArchiveMetadata or None if not found
        """
        analysis_id = self.get_analysis_id(filename)
        metadata = self._index.get(analysis_id)

        if not metadata:
            logger.warning(f"Analysis not found in index: {analysis_id}")
            return None

        # Only update provided fields
        if user_title is not None:
            metadata.user_title = user_title
        if tags is not None:
            metadata.tags = tags
        if notes is not None:
            metadata.notes = notes
        if starred is not None:
            metadata.starred = starred

        self._save_index()
        logger.info(f"Updated analysis metadata: {analysis_id}")
        return metadata

    def delete_analysis(self, filename: str) -> bool:
        """
        Remove an analysis from the index.

        Args:
            filename: Analysis filename

        Returns:
            True if deleted, False if not found
        """
        analysis_id = self.get_analysis_id(filename)

        if analysis_id not in self._index:
            logger.warning(f"Analysis not found in index: {analysis_id}")
            return False

        del self._index[analysis_id]
        self._save_index()

        logger.info(f"Removed analysis from index: {analysis_id}")
        return True

    def list_analyses(
        self,
        starred_only: bool = False,
        tag_filter: Optional[str] = None,
        search_query: Optional[str] = None,
        limit: int = 100
    ) -> List[ArchiveMetadata]:
        """
        List all analyses with optional filtering.

        Args:
            starred_only: Only return starred analyses
            tag_filter: Filter by tag
            search_query: Search in titles and notes
            limit: Maximum number of results

        Returns:
            List of ArchiveMetadata sorted by creation date (newest first)
        """
        results = []

        for metadata in self._index.values():
            # Apply filters
            if starred_only and not metadata.starred:
                continue
            if tag_filter and tag_filter not in metadata.tags:
                continue
            if search_query:
                query_lower = search_query.lower()
                if (
                    query_lower not in (metadata.display_title or '').lower() and
                    query_lower not in (metadata.notes or '').lower()
                ):
                    continue

            results.append(metadata)

        # Sort by creation date (newest first)
        results.sort(key=lambda x: x.created_at or '', reverse=True)

        return results[:limit]

    def get_all_tags(self) -> List[str]:
        """
        Get all unique tags across all analyses.

        Returns:
            Sorted list of unique tags
        """
        tags = set()
        for metadata in self._index.values():
            tags.update(metadata.tags)
        return sorted(tags)

    def sync_with_filesystem(self) -> Dict[str, int]:
        """
        Synchronize index with filesystem.

        Adds entries for files not in index, removes entries for
        files that no longer exist.

        Returns:
            Dict with 'added' and 'removed' counts
        """
        if not self.analyses_dir.exists():
            return {'added': 0, 'removed': 0}

        # Get all analysis files from filesystem
        fs_files = set(f.name for f in self.analyses_dir.glob('analysis_*.json'))

        # Get all files in index
        index_files = set(m.filename for m in self._index.values())

        # Files to add (in filesystem but not in index)
        to_add = fs_files - index_files
        added = 0
        for filename in to_add:
            try:
                # Load analysis to get metadata
                file_path = self.analyses_dir / filename
                with open(file_path, 'r') as f:
                    data = json.load(f)

                metadata_data = data.get('metadata', {})
                self.add_analysis(
                    filename=filename,
                    recording_filename=metadata_data.get('recording_file'),
                    auto_title=metadata_data.get('auto_title'),
                    duration_seconds=metadata_data.get('duration_seconds', 0),
                    speaker_count=metadata_data.get('speaker_count', 0),
                    segment_count=metadata_data.get('segment_count', 0)
                )
                added += 1
            except Exception as e:
                logger.warning(f"Failed to add {filename} to index: {e}")

        # Files to remove (in index but not in filesystem)
        to_remove = index_files - fs_files
        removed = 0
        for filename in to_remove:
            self.delete_analysis(filename)
            removed += 1

        if added or removed:
            logger.info(f"Index sync: added {added}, removed {removed}")

        return {'added': added, 'removed': removed}

    def get_index_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the archive.

        Returns:
            Dict with summary statistics
        """
        total = len(self._index)
        starred = sum(1 for m in self._index.values() if m.starred)
        total_duration = sum(m.duration_seconds for m in self._index.values())

        return {
            'total_analyses': total,
            'starred_count': starred,
            'total_duration_seconds': total_duration,
            'total_tags': len(self.get_all_tags())
        }
