# GitHub Repository Setup Guide

This guide will help you publish the Starship Horizons Learning AI project to GitHub.

## Pre-Publication Checklist

### ✅ Completed Items

- [x] All tests passing (50/50)
- [x] Documentation complete (API, Architecture, Quick Start)
- [x] LICENSE file created (MIT License)
- [x] .gitignore properly configured
- [x] No hardcoded secrets or credentials
- [x] All Python modules documented
- [x] Project structure organized
- [x] Code follows standards (CLAUDE.md)

### 📝 Items to Complete Before Push

1. **Update README URLs**
   - Replace `https://github.com/yourusername/SH-Learning-AI.git` with your actual GitHub URL
   - Update line 64 in README.md

2. **Review .env file**
   - Ensure `.env` is in `.gitignore` (✅ already done)
   - Verify no sensitive data in `.env.example`

3. **Optional: Update LICENSE**
   - Add your name/organization to the copyright line if desired

## Step-by-Step GitHub Setup

### Option 1: Create New Repository via GitHub Web Interface

1. **Go to GitHub.com**
   - Navigate to https://github.com/new
   - Or click "New" from your repositories page

2. **Repository Settings**
   ```
   Repository name: SH-Learning-AI
   Description: AI-powered telemetry capture and analysis for Starship Horizons bridge simulator
   Visibility: Public (or Private if preferred)

   DO NOT initialize with:
   - [x] README (we already have one)
   - [x] .gitignore (we already have one)
   - [x] License (we already have one)
   ```

3. **Create Repository**
   - Click "Create repository"
   - Copy the repository URL (e.g., `https://github.com/YOUR-USERNAME/SH-Learning-AI.git`)

### Option 2: Use GitHub CLI

```bash
# Install GitHub CLI if needed
# https://cli.github.com/

# Authenticate
gh auth login

# Create repository
gh repo create SH-Learning-AI \
  --public \
  --description "AI-powered telemetry capture and analysis for Starship Horizons bridge simulator" \
  --source=. \
  --remote=origin
```

## Initialize Git and Push to GitHub

### If Starting Fresh (No Git History)

```bash
# Navigate to project directory
cd /workspaces/SH-Learning-AI

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Starship Horizons Learning AI

- Complete telemetry capture system
- Audio transcription with speaker diarization
- Mission analysis and reporting
- LLM-powered narrative generation
- Comprehensive documentation
- Full test suite (50 tests)"

# Add remote (replace YOUR-USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR-USERNAME/SH-Learning-AI.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### If You Already Have Git History

```bash
# Check current status
git status

# Add any new files
git add .

# Commit changes
git commit -m "Prepare for GitHub publication

- Add comprehensive documentation
- Add LICENSE file
- Organize project structure
- Ensure all tests pass"

# Add remote (replace YOUR-USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR-USERNAME/SH-Learning-AI.git

# Push to GitHub
git push -u origin main
```

## Post-Publication Tasks

### 1. Update README with Correct URL

Once you know your GitHub URL, update the README:

```bash
# Edit README.md line 64
# Change: https://github.com/yourusername/SH-Learning-AI.git
# To:     https://github.com/YOUR-ACTUAL-USERNAME/SH-Learning-AI.git

git add README.md
git commit -m "docs: update repository URL in README"
git push
```

### 2. Configure Repository Settings on GitHub

**Settings → General:**
- ✅ Set description
- ✅ Add topics: `python`, `ai`, `starship-horizons`, `telemetry`, `audio-transcription`, `llm`
- ✅ Add website (if applicable)

**Settings → Code and automation → Pages** (optional):
- Enable GitHub Pages for documentation
- Source: Deploy from branch `main` / `docs` folder

**Settings → Features:**
- ✅ Enable Issues
- ✅ Enable Discussions (optional, for community)
- ✅ Disable Wiki (documentation is in docs/)

### 3. Create GitHub Issue Templates (Optional)

Create `.github/ISSUE_TEMPLATE/`:

```bash
mkdir -p .github/ISSUE_TEMPLATE
```

**Bug Report Template** (`.github/ISSUE_TEMPLATE/bug_report.md`):
```markdown
---
name: Bug Report
about: Report a bug in the system
---

## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Step one
2. Step two
3. ...

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- Python version:
- OS:
- Game server version:

## Logs
Paste relevant logs here
```

**Feature Request Template** (`.github/ISSUE_TEMPLATE/feature_request.md`):
```markdown
---
name: Feature Request
about: Suggest a new feature
---

## Feature Description
Clear description of the proposed feature

## Use Case
Why is this feature needed?

## Proposed Implementation
How should this work?
```

### 4. Add GitHub Actions CI/CD (Optional)

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run tests
      run: pytest tests/ -v
```

### 5. Create Releases (When Ready)

```bash
# Tag a version
git tag -a v1.0.0 -m "Release v1.0.0

- Initial public release
- Complete telemetry capture system
- Audio transcription with speaker diarization
- Mission analysis and LLM reporting
- Comprehensive documentation"

# Push tag
git push origin v1.0.0

# Or use GitHub CLI
gh release create v1.0.0 \
  --title "v1.0.0 - Initial Release" \
  --notes "See CHANGELOG.md for details"
```

## Recommended GitHub Repository Structure

Your repository will look like:

```
https://github.com/YOUR-USERNAME/SH-Learning-AI
├── README.md (landing page)
├── LICENSE
├── docs/ (documentation browser)
├── Code tab
├── Issues tab
├── Pull Requests tab
└── Releases tab
```

## Visibility and Privacy

### Public Repository (Recommended)
**Pros:**
- Community can use and contribute
- Showcase your work
- GitHub Actions free for public repos
- More visibility

**Cons:**
- Code is visible to everyone

### Private Repository
**Pros:**
- Code is private
- Control who has access

**Cons:**
- Limited GitHub Actions minutes
- No community contributions

## Security Considerations

### Items Already Protected:
- ✅ `.env` file gitignored (contains credentials)
- ✅ `data/` directory gitignored (recordings)
- ✅ No hardcoded credentials in code
- ✅ All secrets via environment variables

### Double-Check Before Push:
```bash
# Search for potential secrets one more time
grep -r "192.168" . --include="*.py" | grep -v ".env.example"

# Check what will be committed
git status
git diff --cached

# See what's ignored
git status --ignored
```

### If You Accidentally Commit Secrets:
1. **DO NOT** just delete in a new commit (history still has it)
2. Use `git filter-branch` or BFG Repo-Cleaner
3. Rotate any exposed credentials immediately
4. Consider repository as compromised

## Collaboration Setup

### Branch Protection Rules (Settings → Branches)

For `main` branch:
- ✅ Require pull request before merging
- ✅ Require status checks to pass (if using CI/CD)
- ✅ Require branches to be up to date
- ✅ Include administrators

### Collaborator Guidelines

Add to CONTRIBUTING.md:
- Fork the repository
- Create feature branch (`git checkout -b feature/AmazingFeature`)
- Follow CLAUDE.md standards
- Ensure tests pass
- Submit pull request

## Maintenance After Publication

### Regular Tasks:
- 📅 Review and merge pull requests
- 📅 Triage issues
- 📅 Update documentation
- 📅 Create releases for major changes
- 📅 Monitor dependencies for security updates

### GitHub Features to Use:
- **Projects**: Track development roadmap
- **Discussions**: Community Q&A
- **Wiki**: Additional documentation (optional)
- **GitHub Pages**: Host documentation website

## Quick Command Reference

```bash
# Clone your published repository
git clone https://github.com/YOUR-USERNAME/SH-Learning-AI.git

# Check remote
git remote -v

# Create new branch
git checkout -b feature/new-feature

# Commit changes
git add .
git commit -m "feat: add new feature"

# Push branch
git push origin feature/new-feature

# Create pull request
gh pr create --title "Add new feature" --body "Description"

# Merge to main
gh pr merge 123 --squash
```

## Final Pre-Push Checklist

Run this script to verify everything:

```bash
#!/bin/bash
echo "=== GitHub Publication Checklist ==="
echo

echo "✓ Tests"
pytest tests/ -q || echo "❌ Tests failed!"

echo
echo "✓ No sensitive files"
if git status --ignored | grep -q ".env"; then
    echo "  .env properly ignored"
else
    echo "  ⚠️  Check .env status"
fi

echo
echo "✓ Documentation"
ls docs/*.md | wc -l
echo "  documentation files"

echo
echo "✓ License"
if [ -f LICENSE ]; then
    echo "  LICENSE file exists"
else
    echo "  ❌ LICENSE file missing"
fi

echo
echo "=== Ready for GitHub! ==="
```

## Support and Resources

- **Git Documentation**: https://git-scm.com/doc
- **GitHub Docs**: https://docs.github.com
- **GitHub CLI**: https://cli.github.com
- **Semantic Versioning**: https://semver.org

## Next Steps

1. Create GitHub repository (web or CLI)
2. Update README with your repository URL
3. Initialize git and push
4. Configure repository settings
5. Add optional CI/CD, issue templates
6. Announce your project!

---

**Questions?** Refer to CONTRIBUTING.md or open an issue after publication.

**Ready to publish?** Follow the "Initialize Git and Push to GitHub" section above!
