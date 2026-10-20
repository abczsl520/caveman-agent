#!/bin/bash
set -euo pipefail

# Caveman Public Repo Publisher
# Syncs private repo → public repo with full sanitization

PRIVATE_REPO="$HOME/projects/caveman"
PUBLIC_REPO="/tmp/caveman-public"
PUBLIC_REMOTE="https://github.com/abczsl520/caveman-agent.git"
AUTHOR_NAME="Caveman Team"
AUTHOR_EMAIL="noreply@cavemanagent.ai"

echo "🦴 Caveman Public Repo Publisher"
echo "================================"

# Step 1: Check GITHUB_PAT
if [ -z "${GITHUB_PAT:-}" ]; then
    echo "❌ Set GITHUB_PAT first: export GITHUB_PAT=gho_..."
    exit 1
fi

# Step 2: Run tests in private repo
echo "📋 Running tests..."
cd "$PRIVATE_REPO"
.venv/bin/python -m pytest tests/ -q --tb=short || {
    echo "❌ Tests failed. Fix before publishing."
    exit 1
}

# Step 3: Create clean copy
echo "📦 Creating clean copy..."
rm -rf "$PUBLIC_REPO"
mkdir -p "$PUBLIC_REPO"
cd "$PUBLIC_REPO"
git init -b main
git config user.name "$AUTHOR_NAME"
git config user.email "$AUTHOR_EMAIL"

# Step 4: Copy files (respecting .gitignore patterns)
rsync -av --exclude='.git' --exclude='.venv' --exclude='__pycache__' \
    --exclude='*.pyc' --exclude='.pytest_cache' --exclude='*.egg-info' \
    --exclude='.coverage' --exclude='hermes-src' --exclude='openclaw-src' \
    --exclude='.DS_Store' --exclude='gateway_sessions' --exclude='*.log' \
    --exclude='memory/' --exclude='config/' \
    "$PRIVATE_REPO/" "$PUBLIC_REPO/" > /dev/null

# Step 5: Security scan
echo "🔍 Security scan..."
ISSUES=0

# Personal info
# Deny-lists live OUTSIDE the repo on purpose: publishing the words to redact
# would leak exactly those words. See scripts/*.example.txt for the format.
DENY_PERSONAL="${SANITIZE_DENY_PERSONAL:-$HOME/.sanitize-personal.txt}"
DENY_IP="${SANITIZE_DENY_IP:-$HOME/.sanitize-ip.txt}"
if [ -f "$DENY_PERSONAL" ] && grep -rqF -f "$DENY_PERSONAL" \
    --include="*.py" --include="*.yaml" --include="*.json" --include="*.md" .; then
    echo "❌ Personal info found!"
    grep -rnF -f "$DENY_PERSONAL" --include="*.py" --include="*.md" .
    ISSUES=$((ISSUES + 1))
fi

# Real IPs
if [ -f "$DENY_IP" ] && grep -rqF -f "$DENY_IP" \
    --include="*.py" --include="*.yaml" --include="*.json" --include="*.md" .; then
    echo "❌ Real server IPs found!"
    grep -rnF -f "$DENY_IP" --include="*.py" --include="*.yaml" --include="*.json" --include="*.md" .
    ISSUES=$((ISSUES + 1))
fi

# API keys (real patterns, not test vectors)
if grep -rq "sk-[a-f0-9]\{40,\}" --include="*.py" --include="*.yaml" --include="*.json" .; then
    echo "❌ Real API keys found!"
    ISSUES=$((ISSUES + 1))
fi

# Absolute paths
if grep -rqE "/(Users|home)/[A-Za-z0-9._-]+/" --include="*.py" --include="*.yaml" --include="*.json" --include="*.md" .; then
    echo "❌ Absolute paths found!"
    ISSUES=$((ISSUES + 1))
fi

if [ "$ISSUES" -gt 0 ]; then
    echo "❌ $ISSUES security issues found. Aborting."
    exit 1
fi
echo "✅ Security scan passed"

# Step 6: Commit and push
VERSION=$(grep 'version' "$PRIVATE_REPO/pyproject.toml" | head -1 | grep -o '"[^"]*"' | tr -d '"')
COMMIT_MSG="Release v${VERSION} — $(date +%Y-%m-%d)"

git add -A
git commit -m "$COMMIT_MSG" --author="$AUTHOR_NAME <$AUTHOR_EMAIL>"

# Verify author
ACTUAL_AUTHOR=$(git log -1 --format="%an <%ae>")
if [ "$ACTUAL_AUTHOR" != "$AUTHOR_NAME <$AUTHOR_EMAIL>" ]; then
    echo "❌ Author mismatch: $ACTUAL_AUTHOR"
    exit 1
fi

git remote add origin "$PUBLIC_REMOTE"
# Auth via one-shot header: the PAT never touches .git/config or disk
GIT_AUTH_HEADER="Authorization: Basic $(printf 'x-access-token:%s' "$GITHUB_PAT" | base64)"
git -c http.extraheader="$GIT_AUTH_HEADER" push origin main --force

echo ""
echo "✅ Published to https://github.com/abczsl520/caveman-agent"
echo "   Author: $ACTUAL_AUTHOR"
echo "   Commit: $COMMIT_MSG"
