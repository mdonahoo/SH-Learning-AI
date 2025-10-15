# CLAUDE.md - Claude Code Development Guidelines

This file provides **MANDATORY** guidelines for Claude Code when working with this repository.

## 🚨 CRITICAL: You MUST Follow These Standards

### Project Overview
Starship Horizons Learning AI - A telemetry capture and analysis system for the Starship Horizons bridge simulator. Currently focused on WebSocket-based game integration and telemetry recording, with plans for AI crew member implementation.

## 📋 MANDATORY CODING STANDARDS

### 1. Project Structure Rules
**ALWAYS place files in the correct location:**
```
src/           → Production code ONLY
scripts/       → Standalone scripts and utilities
tests/        → Test files (must start with test_)
docs/         → Documentation
examples/     → Example code and demos
configs/      → Configuration files
data/         → Data storage (auto-created, gitignored)
logs/         → Log files (auto-created, gitignored)
```

**NEVER:**
- Put test files in root directory
- Put scripts in src/
- Put production code in scripts/
- Create new top-level directories without approval

### 2. File Creation Rules

**ALWAYS include proper headers:**
```python
"""
Module description.

Detailed explanation if needed.
"""

import os  # Standard library first
from typing import Optional  # Type hints

import numpy as np  # Third-party second
from dotenv import load_dotenv

from src.module import Class  # Local imports last
```

**ALWAYS use environment variables:**
```python
# ✅ CORRECT
import os
from dotenv import load_dotenv

load_dotenv()
HOST = os.getenv('GAME_HOST', 'localhost')
PORT = int(os.getenv('GAME_PORT', '1864'))

# ❌ NEVER DO THIS
HOST = "192.168.1.100"  # NEVER hardcode IPs
PORT = 1864  # NEVER hardcode ports
```

### 3. Code Quality Requirements

**EVERY Python file MUST have:**
- [ ] Module docstring
- [ ] Type hints for all functions
- [ ] Proper error handling
- [ ] Logging (no print statements)
- [ ] Environment variables (no hardcoded values)

**Example of CORRECT code:**
```python
"""
Game client module for WebSocket communication.
"""

import logging
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class GameClient:
    """
    WebSocket client for game communication.

    Attributes:
        host: Server hostname
        port: Server port number
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None
    ):
        """Initialize client with configuration."""
        self.host = host or os.getenv('GAME_HOST', 'localhost')
        self.port = port or int(os.getenv('GAME_PORT', '1864'))
        logger.info(f"Client configured for {self.host}:{self.port}")

    async def connect(self) -> bool:
        """
        Establish connection to server.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Implementation here
            logger.info("Connected successfully")
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
```

### 4. Testing Requirements

**For EVERY new module in src/, CREATE a test file:**
```python
# tests/test_game_client.py
"""Tests for game client module."""

import pytest
from unittest.mock import Mock, patch
from src.integration.game_client import GameClient


class TestGameClient:
    """Test suite for GameClient."""

    @pytest.fixture
    def client(self):
        """Create test client instance."""
        return GameClient(host='localhost', port=1234)

    def test_initialization(self, client):
        """Test client initializes with correct values."""
        assert client.host == 'localhost'
        assert client.port == 1234

    @patch('src.integration.game_client.websockets')
    async def test_connect_success(self, mock_ws, client):
        """Test successful connection."""
        mock_ws.connect.return_value = Mock()
        result = await client.connect()
        assert result is True
```

### 5. Documentation Standards

**ALWAYS document:**
- Module purpose (module docstring)
- Class purpose and attributes (class docstring)
- Function purpose, args, returns, raises (function docstring)
- Complex logic (inline comments)

**Use Google-style docstrings:**
```python
def process_event(event: dict, validate: bool = True) -> Optional[dict]:
    """
    Process a game event.

    Args:
        event: Event dictionary to process
        validate: Whether to validate event structure

    Returns:
        Processed event or None if invalid

    Raises:
        ValueError: If event structure is invalid
    """
```

### 6. Error Handling Rules

**NEVER use bare except:**
```python
# ❌ WRONG
try:
    risky_operation()
except:
    pass

# ✅ CORRECT
try:
    risky_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    # Handle appropriately
```

### 7. Async Code Standards

**For WebSocket/async code:**
```python
import asyncio
from typing import Optional

class WebSocketClient:
    """WebSocket client with proper cleanup."""

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.disconnect()

    async def connect_with_retry(
        self,
        max_retries: int = 3,
        timeout: float = 30.0
    ) -> bool:
        """Connect with retry logic and timeout."""
        for attempt in range(max_retries):
            try:
                await asyncio.wait_for(
                    self.connect(),
                    timeout=timeout
                )
                return True
            except asyncio.TimeoutError:
                logger.warning(f"Attempt {attempt + 1} timed out")
        return False
```

## 🎯 Current Implementation Focus

### Active Modules
- ✅ `src/integration/` - WebSocket clients and game communication
- ✅ `src/metrics/` - Event recording and analysis
- ✅ `src/audio/` - Audio I/O and processing
- ✅ `scripts/` - Testing and recording utilities

### Planned Modules (Not Yet Implemented)
- 🚧 `src/agents/` - AI crew members
- 🚧 `src/mcp_servers/` - MCP protocol servers
- 🚧 `src/voice/` - Voice control system
- 🚧 `src/personality/` - Crew personality system

## 🔧 Environment Configuration

**ALWAYS check/create `.env` file:**
```bash
# Copy template if .env doesn't exist
cp .env.example .env

# Required variables:
GAME_HOST=192.168.68.55
GAME_PORT_WS=1865
GAME_PORT_API=1864
```

## 📝 Before Creating ANY Code

### Checklist for New Files:
1. **Location**: Is it in the correct directory?
2. **Naming**: Does it follow naming conventions?
3. **Headers**: Does it have proper imports and docstring?
4. **Config**: Does it use environment variables?
5. **Logging**: Does it use logger instead of print?
6. **Types**: Does it have type hints?
7. **Errors**: Does it handle errors properly?
8. **Tests**: Does it have corresponding tests?

### Checklist for Scripts:
1. **Purpose**: Is it a standalone utility?
2. **Location**: Is it in `scripts/` directory?
3. **Config**: Does it use `load_dotenv()`?
4. **Args**: Does it use `argparse` for CLI args?
5. **Main**: Does it have `if __name__ == "__main__"`?

## ⚠️ Common Mistakes to AVOID

1. **Hardcoded values** → Use environment variables
2. **Print statements** → Use logging
3. **Bare exceptions** → Catch specific exceptions
4. **No type hints** → Always add type hints
5. **No docstrings** → Document everything
6. **Test files in root** → Put in tests/
7. **Scripts in src/** → Put in scripts/
8. **No error handling** → Handle all exceptions
9. **Synchronous I/O in async** → Use async methods
10. **No resource cleanup** → Use context managers

## 🚀 Quick Templates

### New Module Template
```python
"""
Brief description of module.
"""

import logging
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
CONFIG_VAR = os.getenv('ENV_VAR', 'default')


class YourClass:
    """Class description."""

    def __init__(self):
        """Initialize with configuration."""
        logger.info("Initializing YourClass")
```

### New Script Template
```python
#!/usr/bin/env python3
"""
Script description.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.module import Component

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main script logic."""
    parser = argparse.ArgumentParser(description='Script purpose')
    parser.add_argument('--host', default=os.getenv('GAME_HOST'))
    args = parser.parse_args()

    try:
        # Script logic here
        logger.info("Script completed")
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

## 📚 Reference Documentation

- **Best Practices**: See `docs/BEST_PRACTICES.md`
- **API Documentation**: See `docs/API.md`
- **Testing Guide**: See `docs/TESTING.md`

## 🔍 Final Review Checklist

Before completing ANY task, verify:
- [ ] Code follows ALL standards in this document
- [ ] No hardcoded values anywhere
- [ ] Proper error handling throughout
- [ ] Tests created for new code
- [ ] Documentation is complete
- [ ] Code is in correct directory
- [ ] Environment variables used
- [ ] Logging used (no prints)
- [ ] Type hints on all functions
- [ ] Docstrings on all classes/functions

**Remember: Quality > Speed. Always follow these standards.**