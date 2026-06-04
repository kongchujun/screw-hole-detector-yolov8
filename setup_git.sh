#!/bin/bash
# Run once in Terminal (Cursor sandbox may block git init):
#   cd ~/Projects/screw-hole-detector-yolov8 && chmod +x setup_git.sh && ./setup_git.sh

set -e
cd "$(dirname "$0")"

echo "→ Initializing git repository..."
rm -rf .git
git init -b main

echo "→ Staging project files..."
git add -A
git status --short

echo "→ Creating initial commit..."
git commit -m "$(cat <<'EOF'
Initial commit: ML vision-guided screw-hole detection (YOLOv8)

Includes training/export scripts, Jupyter notebook, bilingual README,
and demo dataset labels (images generated via generate_demo_dataset.py).
EOF
)"

echo ""
echo "Done. Branch: $(git branch --show-current)"
echo "Remote:  git remote add origin <your-repo-url> && git push -u origin main"
