#!/bin/bash

echo "🚀 Options Backtester Dashboard - GitHub Setup"
echo "=============================================="
echo ""

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: Please run this script from the OptionsBacktester_Dashboard directory"
    echo "   Current directory: $(pwd)"
    echo "   Expected file: app.py"
    exit 1
fi

echo "📁 Current directory: $(pwd)"
echo "📂 Files in repository:"
ls -la

echo ""
echo "🔧 Git status:"
git status

echo ""
echo "📋 Next Steps:"
echo "1. Create a GitHub repository at: https://github.com/new"
echo "   - Name: OptionsBacktester_Dashboard"
echo "   - Make it Public"
echo ""
echo "2. Run these commands:"
echo "   git remote add origin https://github.com/RaymondWKWong/OptionsBacktester_Dashboard.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. Deploy to Streamlit Cloud:"
echo "   - Go to: https://share.streamlit.io/"
echo "   - Sign in with GitHub"
echo "   - Create new app from your repository"
echo ""
echo "✅ Your dashboard is ready for deployment!"
