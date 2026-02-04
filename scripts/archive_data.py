#!/usr/bin/env python3
"""
Archive data into date-stamped batch directories.

Moves (or copies) all analyses, recordings, and telemetry from active
data directories into a structured archive under data/archive/. Builds
a manifest.json with session groupings and an archive_index.json for
browsing.
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path(os.getenv('DATA_DIR', 'data'))
ARCHIVE_BASE = DATA_DIR / 'archive'


def collect_files(data_dir: Path) -> Dict[str, List[Tuple[Path, str]]]:
    """
    Collect all archivable files from data directories.

    Scans the shared data directories and all workspace subdirectories
    for analyses, recordings, and telemetry files.

    Args:
        data_dir: Root data directory.

    Returns:
        Dict mapping category to list of (source_path, dest_filename) tuples.
    """
    files: Dict[str, List[Tuple[Path, str]]] = {
        'analyses': [],
        'recordings': [],
        'telemetry': [],
    }
    seen_names: Dict[str, set] = {k: set() for k in files}

    def _add_file(category: str, path: Path, prefix: str = '') -> None:
        """Add a file, handling name collisions with prefix."""
        name = path.name
        dest_name = name

        if name in seen_names[category]:
            # Collision: prefix with source identifier
            dest_name = f"{prefix}_{name}" if prefix else name
            if dest_name in seen_names[category]:
                # Still collides, add counter
                base, ext = os.path.splitext(dest_name)
                counter = 2
                while f"{base}_{counter}{ext}" in seen_names[category]:
                    counter += 1
                dest_name = f"{base}_{counter}{ext}"

        seen_names[category].add(dest_name)
        files[category].append((path, dest_name))

    # Shared data directories
    for category, pattern in [
        ('analyses', 'analysis_*.json'),
        ('recordings', '*.wav'),
        ('telemetry', 'telemetry_*.json'),
    ]:
        source_dir = data_dir / category
        if source_dir.exists():
            for f in sorted(source_dir.glob(pattern)):
                if f.is_file():
                    _add_file(category, f)

    # Also collect non-wav recording directories (audio sessions)
    recordings_dir = data_dir / 'recordings'
    if recordings_dir.exists():
        for item in sorted(recordings_dir.iterdir()):
            if item.is_dir() and item.name.startswith('audio_session_'):
                # Include audio session directories
                _add_file('recordings', item)

    # Workspace directories
    workspaces_dir = data_dir / 'workspaces'
    if workspaces_dir.exists():
        for ws_dir in sorted(workspaces_dir.iterdir()):
            if not ws_dir.is_dir():
                continue
            ws_id = ws_dir.name[:8]  # Short workspace prefix

            for category, pattern in [
                ('analyses', 'analysis_*.json'),
                ('recordings', '*.wav'),
                ('telemetry', 'telemetry_*.json'),
            ]:
                source_dir = ws_dir / category
                if source_dir.exists():
                    for f in sorted(source_dir.glob(pattern)):
                        if f.is_file():
                            _add_file(category, f, prefix=ws_id)

            # Workspace recording directories
            ws_recordings = ws_dir / 'recordings'
            if ws_recordings.exists():
                for item in sorted(ws_recordings.iterdir()):
                    if item.is_dir() and item.name.startswith('audio_session_'):
                        _add_file('recordings', item, prefix=ws_id)

    return files


def read_analysis_metadata(path: Path) -> Dict[str, Any]:
    """
    Read metadata from an analysis JSON file.

    Args:
        path: Path to analysis JSON file.

    Returns:
        Metadata dict from the analysis file.
    """
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data.get('metadata', {})
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read metadata from {path}: {e}")
        return {}


def preload_analysis_metadata(
    files: Dict[str, List[Tuple[Path, str]]]
) -> Dict[str, Dict[str, Any]]:
    """
    Pre-read metadata from all analysis files before moving.

    Args:
        files: Collected files dict from collect_files().

    Returns:
        Dict mapping dest_filename to metadata dict.
    """
    cache: Dict[str, Dict[str, Any]] = {}
    for source_path, dest_name in files['analyses']:
        cache[dest_name] = read_analysis_metadata(source_path)
    return cache


def build_sessions(
    files: Dict[str, List[Tuple[Path, str]]],
    metadata_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Group related files into sessions.

    Matches analyses to recordings via the recording_file metadata field,
    and to telemetry by timestamp proximity.

    Args:
        files: Collected files dict from collect_files().
        metadata_cache: Pre-loaded metadata keyed by dest filename.

    Returns:
        List of session dicts for the manifest.
    """
    sessions: List[Dict[str, Any]] = []
    cache = metadata_cache or {}

    # Build lookup maps
    recording_names = {name: name for _, name in files['recordings']}
    telemetry_names = {name: name for _, name in files['telemetry']}

    # Track which recordings/telemetry are claimed
    claimed_recordings: set = set()
    claimed_telemetry: set = set()

    # Process each analysis
    for source_path, dest_name in files['analyses']:
        meta = cache.get(dest_name) or read_analysis_metadata(source_path)

        recording_file = meta.get('recording_file')
        matched_recording: Optional[str] = None
        matched_telemetry: Optional[str] = None

        # Match recording
        if recording_file:
            if recording_file in recording_names:
                matched_recording = recording_file
                claimed_recordings.add(recording_file)
            else:
                # Try to find by timestamp pattern
                ts_match = re.search(r'(\d{8}_\d{6})', recording_file)
                if ts_match:
                    ts = ts_match.group(1)
                    for rname in recording_names:
                        if ts in rname and rname not in claimed_recordings:
                            matched_recording = rname
                            claimed_recordings.add(rname)
                            break

        # Match telemetry by timestamp proximity
        analysis_ts = re.search(r'(\d{8}_\d{6})', dest_name)
        if analysis_ts:
            ts = analysis_ts.group(1)
            for tname in telemetry_names:
                if ts in tname and tname not in claimed_telemetry:
                    matched_telemetry = tname
                    claimed_telemetry.add(tname)
                    break

        session = {
            'analysis': dest_name,
            'recording': matched_recording,
            'telemetry': matched_telemetry,
            'title': meta.get('auto_title'),
            'duration_seconds': meta.get('duration_seconds', 0),
            'speaker_count': meta.get('speaker_count', 0),
            'segment_count': meta.get('segment_count', 0),
            'created_at': meta.get('created_at'),
        }
        sessions.append(session)

    # Add unclaimed recordings as standalone entries
    for _, dest_name in files['recordings']:
        if dest_name not in claimed_recordings:
            sessions.append({
                'analysis': None,
                'recording': dest_name,
                'telemetry': None,
                'title': None,
                'duration_seconds': 0,
                'speaker_count': 0,
                'segment_count': 0,
                'created_at': None,
            })

    # Add unclaimed telemetry as standalone entries
    for _, dest_name in files['telemetry']:
        if dest_name not in claimed_telemetry:
            sessions.append({
                'analysis': None,
                'recording': None,
                'telemetry': dest_name,
                'title': None,
                'duration_seconds': 0,
                'speaker_count': 0,
                'segment_count': 0,
                'created_at': None,
            })

    # Sort by created_at (newest first), with None at the end
    sessions.sort(
        key=lambda s: s.get('created_at') or '0000',
        reverse=True
    )

    return sessions


def build_archive_index(
    files: Dict[str, List[Tuple[Path, str]]],
    metadata_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Build an ArchiveManager-compatible archive_index.json.

    Args:
        files: Collected files dict.
        metadata_cache: Pre-loaded metadata keyed by dest filename.

    Returns:
        Archive index dict ready for JSON serialization.
    """
    analyses: Dict[str, Dict[str, Any]] = {}
    cache = metadata_cache or {}

    for source_path, dest_name in files['analyses']:
        meta = cache.get(dest_name) or read_analysis_metadata(source_path)
        analysis_id = dest_name.replace('.json', '')

        analyses[analysis_id] = {
            'filename': dest_name,
            'recording_filename': meta.get('recording_file'),
            'auto_title': meta.get('auto_title'),
            'user_title': None,
            'created_at': meta.get('created_at', datetime.now().isoformat()),
            'duration_seconds': meta.get('duration_seconds', 0),
            'speaker_count': meta.get('speaker_count', 0),
            'segment_count': meta.get('segment_count', 0),
            'tags': [],
            'notes': '',
            'starred': False,
            'display_title': meta.get('auto_title') or analysis_id,
        }

    return {
        'version': 1,
        'updated_at': datetime.now().isoformat(),
        'analyses': analyses,
    }


def compute_total_size(
    files: Dict[str, List[Tuple[Path, str]]]
) -> int:
    """
    Compute total size in bytes of all files to archive.

    Args:
        files: Collected files dict.

    Returns:
        Total size in bytes.
    """
    total = 0
    for category_files in files.values():
        for source_path, _ in category_files:
            if source_path.is_dir():
                total += sum(
                    f.stat().st_size for f in source_path.rglob('*')
                    if f.is_file()
                )
            else:
                try:
                    total += source_path.stat().st_size
                except OSError:
                    pass
    return total


def move_or_copy(
    source: Path, dest: Path, use_copy: bool = False
) -> None:
    """
    Move or copy a file/directory to the destination.

    Args:
        source: Source path.
        dest: Destination path.
        use_copy: If True, copy instead of move.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        if use_copy:
            shutil.copytree(source, dest)
        else:
            shutil.move(str(source), str(dest))
    else:
        if use_copy:
            shutil.copy2(source, dest)
        else:
            shutil.move(str(source), str(dest))


def clean_empty_dirs(data_dir: Path) -> None:
    """
    Remove empty source directories and stale index files.

    Args:
        data_dir: Root data directory.
    """
    for subdir in ['analyses', 'recordings', 'telemetry']:
        d = data_dir / subdir
        if d.exists() and not any(d.iterdir()):
            logger.info(f"Removing empty directory: {d}")
            d.rmdir()

    # Clean workspace subdirectories
    workspaces_dir = data_dir / 'workspaces'
    if workspaces_dir.exists():
        for ws_dir in workspaces_dir.iterdir():
            if not ws_dir.is_dir():
                continue
            for subdir in ['analyses', 'recordings', 'telemetry']:
                d = ws_dir / subdir
                if d.exists() and not any(d.iterdir()):
                    logger.info(f"Removing empty directory: {d}")
                    d.rmdir()

            # Remove stale archive_index.json in workspace
            ws_index = ws_dir / 'archive_index.json'
            if ws_index.exists():
                logger.info(f"Removing stale workspace index: {ws_index}")
                ws_index.unlink()

            # Remove workspace dir if empty
            remaining = [f for f in ws_dir.iterdir() if f.name != '.DS_Store']
            if not remaining:
                logger.info(f"Removing empty workspace: {ws_dir}")
                shutil.rmtree(ws_dir)

    # Remove stale shared archive_index.json
    shared_index = data_dir / 'archive_index.json'
    if shared_index.exists():
        logger.info(f"Removing stale shared index: {shared_index}")
        shared_index.unlink()


def archive_data(
    description: str = '',
    dry_run: bool = False,
    use_copy: bool = False,
) -> Optional[Path]:
    """
    Archive all data files into a date-stamped batch directory.

    Args:
        description: Description for the archive batch.
        dry_run: If True, only show what would be done.
        use_copy: If True, copy files instead of moving.

    Returns:
        Path to the created archive directory, or None on dry run.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_dir = ARCHIVE_BASE / timestamp

    logger.info(f"Archive target: {archive_dir}")

    # Collect files
    files = collect_files(DATA_DIR)

    total_files = sum(len(v) for v in files.values())
    if total_files == 0:
        logger.info("No files to archive.")
        return None

    total_size = compute_total_size(files)
    action = 'copy' if use_copy else 'move'

    logger.info(f"Files to {action}:")
    for category, file_list in files.items():
        logger.info(f"  {category}: {len(file_list)} files")
        for source_path, dest_name in file_list:
            logger.info(f"    {source_path} -> {dest_name}")

    logger.info(f"Total: {total_files} files, {total_size / 1024 / 1024:.1f} MB")

    # Pre-read all analysis metadata before moving files
    metadata_cache = preload_analysis_metadata(files)

    if dry_run:
        logger.info("DRY RUN - no files were moved or copied.")

        # Build and display session groupings
        sessions = build_sessions(files, metadata_cache=metadata_cache)
        analyzed_sessions = [s for s in sessions if s.get('analysis')]
        standalone_recordings = [
            s for s in sessions
            if s.get('recording') and not s.get('analysis')
        ]

        logger.info(f"\nSessions: {len(analyzed_sessions)} analyzed, "
                     f"{len(standalone_recordings)} standalone recordings")
        for s in analyzed_sessions[:5]:
            logger.info(f"  {s.get('title', 'Untitled')}: "
                        f"analysis={s['analysis']}, "
                        f"recording={s.get('recording', 'none')}")

        return None

    # Create archive directory structure
    for subdir in ['analyses', 'recordings', 'telemetry']:
        (archive_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Move/copy files
    moved_count = 0
    for category, file_list in files.items():
        for source_path, dest_name in file_list:
            dest_path = archive_dir / category / dest_name
            try:
                move_or_copy(source_path, dest_path, use_copy=use_copy)
                moved_count += 1
                logger.debug(f"{'Copied' if use_copy else 'Moved'}: "
                             f"{source_path} -> {dest_path}")
            except (OSError, shutil.Error) as e:
                logger.error(f"Failed to {action} {source_path}: {e}")

    logger.info(f"{'Copied' if use_copy else 'Moved'} {moved_count}/{total_files} files")

    # Build manifest using pre-cached metadata
    sessions = build_sessions(files, metadata_cache=metadata_cache)
    manifest = {
        'archive_id': timestamp,
        'created_at': datetime.now().isoformat(),
        'description': description or 'Data archive',
        'source_locations': [str(DATA_DIR)],
        'file_counts': {
            category: len(file_list)
            for category, file_list in files.items()
        },
        'total_size_bytes': total_size,
        'sessions': sessions,
    }

    manifest_path = archive_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Created manifest: {manifest_path}")

    # Build archive index using pre-cached metadata
    archive_index = build_archive_index(files, metadata_cache=metadata_cache)
    index_path = archive_dir / 'archive_index.json'
    with open(index_path, 'w') as f:
        json.dump(archive_index, f, indent=2, default=str)
    logger.info(f"Created archive index: {index_path}")

    # Clean up empty source directories
    if not use_copy:
        clean_empty_dirs(DATA_DIR)

    logger.info(f"Archive complete: {archive_dir}")
    return archive_dir


def main() -> None:
    """Main script entry point."""
    parser = argparse.ArgumentParser(
        description='Archive data files into a date-stamped batch directory.'
    )
    parser.add_argument(
        '--description', '-d',
        default='',
        help='Description for the archive batch'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be moved without making changes'
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy files instead of moving them'
    )
    args = parser.parse_args()

    try:
        result = archive_data(
            description=args.description,
            dry_run=args.dry_run,
            use_copy=args.copy,
        )
        if result:
            logger.info(f"Archive created at: {result}")
    except Exception as e:
        logger.error(f"Archive failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
