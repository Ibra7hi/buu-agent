#!/bin/bash
# ============================================================
# RAG Project — Cleanup Script
# Run this ONCE to finish the restructure migration.
# ============================================================

set -e
echo "🧹 Starting cleanup..."

# 1. Move PDFs to docs/
echo "📄 Moving PDFs to docs/..."
mv fake_company.pdf docs/
mv hima.pdf docs/
mv math.pdf docs/

# 2. Remove old directories that have been migrated to src/
echo "🗑️  Removing old App/ directory..."
rm -rf App/

echo "🗑️  Removing old rag/ directory..."
rm -rf rag/

echo "🗑️  Removing old 'reset & index/' directory..."
rm -rf "reset & index"/

echo "🗑️  Removing old root mcp_server.py..."
rm -f mcp_server.py

echo "🗑️  Removing root __pycache__/..."
rm -rf __pycache__/

echo ""
echo "✅ Cleanup complete! Your new structure is:"
echo ""
find . -not -path './venv/*' -not -path './.git/*' -not -path './frontend/node_modules/*' -not -path './frontend/.next/*' -not -name '*.pyc' | head -60
