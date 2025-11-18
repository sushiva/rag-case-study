#!/bin/bash

echo "🧹 SAFE CLEANUP for RAG Case Study"
echo "===================================="
echo ""
echo "This script will:"
echo "  • DELETE obvious backup files"
echo "  • ARCHIVE uncertain files to _archive/"
echo "  • KEEP essential files"
echo ""

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Are you in the rag-case-study directory?"
    exit 1
fi

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Create backup branch
BACKUP_BRANCH="backup-safe-cleanup-$(date +%Y%m%d-%H%M%S)"
echo "📦 Creating backup branch: $BACKUP_BRANCH"
git checkout -b $BACKUP_BRANCH 2>/dev/null
git add -A 2>/dev/null
git commit -m "Backup before safe cleanup" 2>/dev/null
git push origin $BACKUP_BRANCH 2>/dev/null
git checkout main

echo ""
echo "🗂️  Creating _archive/ folder for uncertain files..."
mkdir -p _archive/data
mkdir -p _archive/golden_data

echo ""
echo "🗑️  Phase 1: DELETE obvious backup/temp files"
echo "=============================================="

# DELETE: Clear backup files (definitely not needed)
rm -f app_backup_counter.py && echo "  ✓ Deleted app_backup_counter.py"
rm -f app_dashboard_backup.py && echo "  ✓ Deleted app_dashboard_backup.py"
rm -f app_dashboard_single_model.py && echo "  ✓ Deleted app_dashboard_single_model.py"
rm -f app_old.py && echo "  ✓ Deleted app_old.py"

# DELETE: Clear zip archives
rm -f *.zip && echo "  ✓ Deleted *.zip files"

# DELETE: Clear test scripts
rm -f key_sanity_test.py && echo "  ✓ Deleted key_sanity_test.py"
rm -f create_visualizations.py && echo "  ✓ Deleted create_visualizations.py"
rm -f generate_final_report.py && echo "  ✓ Deleted generate_final_report.py"

# DELETE: Clear cache
rm -rf __pycache__/ .pytest_cache/ .vscode/ .idea/ 2>/dev/null
find . -name "*.pyc" -delete
find . -name ".DS_Store" -delete
echo "  ✓ Deleted cache files"

echo ""
echo "📦 Phase 2: ARCHIVE uncertain files"
echo "===================================="

# ARCHIVE: Reports (might want to review)
if [ -f "COMPREHENSIVE_CASE_STUDY_REPORT.txt" ]; then
    mv COMPREHENSIVE_CASE_STUDY_REPORT.txt _archive/ && echo "  ✓ Archived COMPREHENSIVE_CASE_STUDY_REPORT.txt"
fi

if [ -f "instructions.txt" ]; then
    mv instructions.txt _archive/ && echo "  ✓ Archived instructions.txt"
fi

if [ -f "eval_results.json" ]; then
    mv eval_results.json _archive/ && echo "  ✓ Archived eval_results.json"
fi

if [ -f "evaluation_report.json" ]; then
    mv evaluation_report.json _archive/ && echo "  ✓ Archived evaluation_report.json"
fi

if [ -f "config.yaml" ]; then
    mv config.yaml _archive/ && echo "  ✓ Archived config.yaml"
fi

# ARCHIVE: golden_data folder (evaluation data - might be needed)
if [ -d "golden_data" ]; then
    mv golden_data/* _archive/golden_data/ 2>/dev/null
    rmdir golden_data
    echo "  ✓ Archived golden_data/ → _archive/golden_data/"
fi

# ARCHIVE: Uncertain data files
if [ -f "data/extracted_text.txt" ]; then
    mv data/extracted_text.txt _archive/data/ && echo "  ✓ Archived data/extracted_text.txt"
fi

if [ -f "data/medical_diagnosis_manual.pdf" ]; then
    mv data/medical_diagnosis_manual.pdf _archive/data/ && echo "  ✓ Archived data/medical_diagnosis_manual.pdf"
fi

if [ -f "data/metadata.json" ]; then
    mv data/metadata.json _archive/data/ && echo "  ✓ Archived data/metadata.json"
fi

# Handle duplicate rag_system_multimodel.py
if [ -f "rag_system_multimodel.py" ] && [ -f "src/rag_system_multimodel.py" ]; then
    rm -f rag_system_multimodel.py && echo "  ✓ Deleted duplicate rag_system_multimodel.py (kept in src/)"
elif [ -f "rag_system_multimodel.py" ] && [ ! -f "src/rag_system_multimodel.py" ]; then
    mkdir -p src
    mv rag_system_multimodel.py src/ && echo "  ✓ Moved rag_system_multimodel.py to src/"
fi

echo ""
echo "✅ Phase 3: Ensure essential structure"
echo "======================================"

# Ensure directories exist
mkdir -p src
mkdir -p data
mkdir -p assets

# Create __init__.py if missing
if [ ! -f "src/__init__.py" ]; then
    touch src/__init__.py && echo "  ✓ Created src/__init__.py"
fi

# Update .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.so
.Python

# Virtual Environment
env/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store

# Streamlit
.streamlit/
*.log

# Environment
.env
secrets.toml

# Testing
.pytest_cache/
.coverage

# Temporary
*.tmp
*.bak
*_backup.py
*_old.py

# Archive (local only)
_archive/
EOF

echo "  ✓ Updated .gitignore"

# Add _archive/ to .gitignore
echo "" >> .gitignore
echo "# Archived files (not committed)" >> .gitignore
echo "_archive/" >> .gitignore

echo ""
echo "📊 FINAL STRUCTURE:"
echo "==================="
echo ""
echo "KEPT (Essential):"
find . -maxdepth 1 -type f -name "*.py" -o -name "*.txt" -o -name "*.md" -o -name "*.yaml" | grep -v "_archive" | sort
echo ""
echo "src/:"
ls -1 src/ 2>/dev/null | grep -v __pycache__
echo ""
echo "data/:"
ls -1 data/ 2>/dev/null
echo ""
echo "ARCHIVED (_archive/):"
find _archive/ -type f 2>/dev/null | sort
echo ""

# Test for missing dependencies
echo "🧪 Checking dependencies..."
if [ -f "app.py" ]; then
    echo "  ✓ app.py exists"
    
    # Check if golden_data is referenced
    if grep -q "golden_data\|golden_dataset" app.py; then
        echo ""
        echo "⚠️  WARNING: app.py references 'golden_data'"
        echo "   Review if you need files from _archive/golden_data/"
    fi
    
    # Check if medical pdf is referenced
    if grep -q "medical.*\.pdf" app.py; then
        echo ""
        echo "⚠️  WARNING: app.py references medical PDF"
        echo "   Review if you need _archive/data/medical_diagnosis_manual.pdf"
    fi
fi

echo ""
echo "✅ SAFE CLEANUP COMPLETE!"
echo "========================="
echo ""
echo "📝 What happened:"
echo "   • DELETED: Backup files, cache, temp files"
echo "   • ARCHIVED: Reports, configs, uncertain data → _archive/"
echo "   • KEPT: Essential app files"
echo ""
echo "🧪 TEST YOUR APP:"
echo "   streamlit run app.py"
echo ""
echo "📋 If app works fine:"
echo "   1. Review _archive/ folder"
echo "   2. Delete _archive/ if you don't need it: rm -rf _archive/"
echo "   3. Commit: git add . && git commit -m 'Safe cleanup'"
echo ""
echo "📋 If app has issues:"
echo "   1. Check _archive/ for needed files"
echo "   2. Move them back: mv _archive/path/to/file ."
echo "   3. Or restore backup: git checkout $BACKUP_BRANCH"
echo ""
echo "💾 Backup branch: $BACKUP_BRANCH"