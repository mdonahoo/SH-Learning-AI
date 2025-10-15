# Contributing Guidelines

## Quick Start for Contributors

### 1. Setup Development Environment
```bash
# Clone and setup
git clone <repository>
cd SH-Learning-AI
make setup  # Sets up environment and dependencies
```

### 2. Before Writing Code

#### Check CLAUDE.md First!
**MANDATORY**: Read `CLAUDE.md` for coding standards that MUST be followed.

#### Use Templates
- New module? Copy from `templates/module_template.py`
- New script? Copy from `templates/script_template.py`
- New tests? Copy from `templates/test_template.py`

### 3. File Placement Rules

| File Type | Location | Naming |
|-----------|----------|---------|
| Production Code | `src/<module>/` | `lowercase_underscore.py` |
| Scripts/Utilities | `scripts/` | `action_target.py` |
| Tests | `tests/` | `test_<module>.py` |
| Documentation | `docs/` | `UPPERCASE.md` |

### 4. Code Checklist

Before ANY commit, ensure your code has:

- [ ] **Module docstring** at the top
- [ ] **Type hints** on ALL functions
- [ ] **Environment variables** (no hardcoded values!)
- [ ] **Logging** (no print statements!)
- [ ] **Error handling** (no bare except!)
- [ ] **Tests** in `tests/` directory
- [ ] **No hardcoded IPs/ports/paths**

### 5. Common Commands

```bash
# Format your code
make format

# Check code quality
make lint

# Run tests
make test

# Do everything before commit
make dev  # Formats, lints, and tests
```

### 6. Environment Variables

**NEVER hardcode configuration!** Always use:
```python
import os
from dotenv import load_dotenv

load_dotenv()
HOST = os.getenv('GAME_HOST', 'localhost')
```

### 7. Testing Requirements

For EVERY file in `src/`, create corresponding test:
- `src/integration/client.py` → `tests/test_client.py`
- Tests must use pytest
- Mock external dependencies
- Aim for 80% coverage minimum

### 8. Pull Request Checklist

- [ ] Code follows ALL standards in CLAUDE.md
- [ ] Tests pass: `make test`
- [ ] Code is formatted: `make format`
- [ ] Linting passes: `make lint`
- [ ] Documentation updated if needed
- [ ] No hardcoded values
- [ ] Commit message follows format

### 9. Commit Message Format

```
<type>: <short description>

<longer description if needed>

Closes #<issue number>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### 10. Getting Help

- Check `docs/BEST_PRACTICES.md` for detailed guidelines
- Review `CLAUDE.md` for mandatory standards
- Look at templates in `templates/` directory
- Use `make help` to see available commands

## Example: Adding a New Feature

1. **Create module from template:**
```bash
cp templates/module_template.py src/integration/new_feature.py
```

2. **Create test from template:**
```bash
cp templates/test_template.py tests/test_new_feature.py
```

3. **Implement feature following standards**

4. **Test your code:**
```bash
make test
```

5. **Format and check:**
```bash
make dev
```

6. **Commit with proper message:**
```bash
git add .
git commit -m "feat: add new feature for game integration

Implements XYZ functionality for better game control.

Closes #123"
```

## Red Flags 🚩

If you're doing ANY of these, STOP and fix:

- ❌ Hardcoding IP addresses or ports
- ❌ Using `print()` instead of logging
- ❌ No type hints on functions
- ❌ Bare `except:` clauses
- ❌ Test files in root directory
- ❌ Scripts in `src/` directory
- ❌ No docstrings
- ❌ No error handling
- ❌ No tests for new code

## Remember

**Quality > Speed**. Always follow the standards. The templates and tools are here to help you write better code faster.