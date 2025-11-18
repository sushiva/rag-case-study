# Navigate to repo (run from home directory)
cd ~/portfolio-project/rag-case-study

# Create backup branch
git checkout -b backup-before-cleanup
git push origin backup-before-cleanup

# Switch back to main
git checkout main

# Remove evaluation files
rm -f evaluator_4metrics.py
rm -f src/question_generator.py
rm -rf results/
rm -f main.py  # If redundant with app.py

# Remove development files
rm -rf __pycache__/
rm -rf .pytest_cache/
rm -rf .vscode/
rm -rf .idea/
rm -rf src/__pycache__/
find . -name "*.pyc" -delete
find . -name ".DS_Store" -delete

# Create/update .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
.streamlit/
*.log

# API Keys (important!)
.env
secrets.toml
EOF

# Add assets folder for screenshots
mkdir -p assets

# Commit cleanup
git add .
git commit -m "Clean up repository: Remove evaluation files, keep only essential code"
git push origin main