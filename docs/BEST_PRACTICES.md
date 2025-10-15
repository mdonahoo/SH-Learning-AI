# Best Practices Guide

## 🏗️ Project Structure

### Directory Organization
```
SH-Learning-AI/
├── src/                    # Source code (all production code here)
│   ├── __init__.py        # Package initialization
│   ├── audio/             # Audio processing module
│   ├── integration/       # Game integration module
│   └── metrics/           # Analytics and metrics module
├── scripts/               # Standalone scripts and utilities
├── tests/                 # Unit and integration tests
├── docs/                  # Documentation
├── data/                  # Data storage (gitignored)
├── logs/                  # Application logs (gitignored)
├── examples/              # Example code and demos
└── configs/               # Configuration files
```

### File Naming Conventions
- **Source files**: `lowercase_with_underscores.py`
- **Test files**: `test_<module_name>.py`
- **Script files**: `<action>_<target>.py` (e.g., `record_game.py`)
- **Classes**: `PascalCase`
- **Functions/Variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`

## 📝 Code Standards

### Python Style Guide
Follow PEP 8 with these specific requirements:

1. **Imports**
```python
# Standard library imports first
import os
import sys
from datetime import datetime

# Third-party imports second
import numpy as np
from dotenv import load_dotenv

# Local imports last
from src.integration import GameClient
from src.metrics import EventRecorder
```

2. **Module Headers**
Every Python file must start with:
```python
"""
Brief description of the module.

Detailed description if needed.
"""
```

3. **Class Documentation**
```python
class GameClient:
    """
    Brief description of the class.

    Attributes:
        host (str): The game server hostname
        port (int): The game server port

    Example:
        >>> client = GameClient()
        >>> client.connect()
    """
```

4. **Function Documentation**
```python
def process_event(event: dict, validate: bool = True) -> dict:
    """
    Process a game event.

    Args:
        event: The event dictionary to process
        validate: Whether to validate the event structure

    Returns:
        The processed event dictionary

    Raises:
        ValueError: If event is invalid
    """
```

### Environment Variables
- **Never hardcode**: IPs, ports, credentials, or paths
- **Always use**: Environment variables with sensible defaults
- **Pattern**:
```python
import os
from dotenv import load_dotenv

load_dotenv()

HOST = os.getenv('GAME_HOST', 'localhost')
PORT = int(os.getenv('GAME_PORT', '1864'))
```

### Error Handling
```python
# Good
try:
    result = risky_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    # Handle gracefully
    return default_value

# Bad
try:
    result = risky_operation()
except:
    pass  # Never silent catch-all
```

### Logging
```python
import logging

logger = logging.getLogger(__name__)

# Use appropriate levels
logger.debug("Detailed diagnostic info")
logger.info("General informational messages")
logger.warning("Warning messages")
logger.error("Error messages")
logger.critical("Critical failure messages")
```

## 🧪 Testing Standards

### Test Organization
```python
# tests/test_game_client.py
import pytest
from src.integration import GameClient

class TestGameClient:
    """Test suite for GameClient class."""

    @pytest.fixture
    def client(self):
        """Create a test client instance."""
        return GameClient(host='localhost', port=1234)

    def test_connection(self, client):
        """Test client can connect to server."""
        assert client.connect() is True

    def test_invalid_host(self):
        """Test client handles invalid host gracefully."""
        client = GameClient(host='invalid', port=1234)
        assert client.connect() is False
```

### Test Requirements
- Every module in `src/` should have corresponding tests
- Test coverage target: 80% minimum
- Use pytest fixtures for setup/teardown
- Mock external dependencies

## 🔧 Configuration Management

### Configuration Files
```python
# configs/app_config.yaml
server:
  host: ${GAME_HOST}
  port: ${GAME_PORT}

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Settings Pattern
```python
# src/config.py
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class AppConfig:
    """Application configuration."""
    host: str = os.getenv('GAME_HOST', 'localhost')
    port: int = int(os.getenv('GAME_PORT', '1864'))
    debug: bool = os.getenv('DEBUG', 'False').lower() == 'true'
```

## 🚀 Async Best Practices

### Async Functions
```python
import asyncio
from typing import Optional

async def fetch_data(timeout: float = 30.0) -> Optional[dict]:
    """
    Fetch data asynchronously.

    Args:
        timeout: Maximum time to wait in seconds

    Returns:
        Data dictionary or None if timeout
    """
    try:
        return await asyncio.wait_for(
            _fetch_implementation(),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        logger.warning(f"Data fetch timed out after {timeout}s")
        return None
```

### WebSocket Patterns
```python
class WebSocketClient:
    """WebSocket client with automatic reconnection."""

    async def connect(self):
        """Connect with retry logic."""
        for attempt in range(self.max_retries):
            try:
                await self._connect()
                return True
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(self.retry_delay)
        return False
```

## 📦 Dependency Management

### Requirements Files
- `requirements.txt` - Production dependencies only
- `requirements-dev.txt` - Development tools
- Pin versions for reproducibility:
```
websockets==12.0
numpy==1.26.4
```

### Import Management
```python
# Avoid circular imports
# src/integration/client.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.metrics import Recorder  # Only for type hints
```

## 🔒 Security Best Practices

1. **No secrets in code** - Use environment variables
2. **Validate inputs** - Always validate external data
3. **Sanitize outputs** - Clean data before logging
4. **Principle of least privilege** - Request minimum permissions

## 📊 Data Management

### File I/O
```python
from pathlib import Path

def save_data(data: dict, filename: str):
    """Save data with proper path handling."""
    # Use Path for cross-platform compatibility
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    filepath = data_dir / filename
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
```

### Resource Management
```python
# Always use context managers
with open('file.txt') as f:
    content = f.read()

# For custom resources
class GameConnection:
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
```

## 🎯 Type Hints

### Always Use Type Hints
```python
from typing import List, Dict, Optional, Union, Callable

def process_events(
    events: List[Dict[str, any]],
    filter_func: Optional[Callable[[dict], bool]] = None
) -> List[Dict[str, any]]:
    """Process a list of events with optional filtering."""
    if filter_func:
        events = [e for e in events if filter_func(e)]
    return events
```

## 📈 Performance Guidelines

1. **Profile before optimizing**
2. **Use generators for large datasets**
3. **Cache expensive computations**
4. **Batch operations when possible**

```python
# Good - Generator for memory efficiency
def process_large_file(filepath: Path):
    with open(filepath) as f:
        for line in f:  # Generator, doesn't load entire file
            yield process_line(line)

# Bad - Loads entire file
def process_large_file(filepath: Path):
    with open(filepath) as f:
        lines = f.readlines()  # Memory intensive
        return [process_line(line) for line in lines]
```

## 🔄 Version Control

### Commit Messages
```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions or changes
- `chore`: Build process or auxiliary tool changes

Example:
```
feat: add WebSocket reconnection logic

Implement automatic reconnection with exponential backoff
when WebSocket connection is lost.

Closes #123
```

## 🚦 Code Review Checklist

Before submitting code:
- [ ] Follows PEP 8 style guide
- [ ] Has proper documentation
- [ ] Includes type hints
- [ ] Has corresponding tests
- [ ] No hardcoded values
- [ ] Proper error handling
- [ ] Uses logging appropriately
- [ ] No commented-out code
- [ ] Imports are organized
- [ ] No print() statements (use logging)