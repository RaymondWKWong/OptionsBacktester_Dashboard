# Deployment Instructions

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `OptionsBacktester_Dashboard`
3. Description: `Interactive Options Strategy Backtester Dashboard`
4. Make it Public (required for free Streamlit deployment)
5. Click "Create repository"

## Step 2: Push Code to GitHub

Run these commands in your terminal:

```bash
cd /home/raymond/Projects/OptionsBacktester_Dashboard
git remote add origin https://github.com/RaymondWKWong/OptionsBacktester_Dashboard.git
git branch -M main
git push -u origin main
```

## Step 3: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `RaymondWKWong/OptionsBacktester_Dashboard`
5. Branch: `main`
6. Main file path: `app.py`
7. Click "Deploy!"

## Step 4: Access Your App

Your app will be available at:
`https://raymondwkwong-optionsbacktester-dashboard-app-xxxxx.streamlit.app/`

## Alternative: Manual GitHub Setup

If you prefer to upload manually:

1. Download all files from `/home/raymond/Projects/OptionsBacktester_Dashboard/`
2. Create a new repository on GitHub
3. Upload the files via GitHub's web interface
4. Follow Step 3 above for Streamlit deployment

## Files in Your Repository

- `app.py` - Main Streamlit application
- `backtester.py` - Options backtesting logic
- `requirements.txt` - Python dependencies
- `README.md` - Documentation
- `.streamlit/config.toml` - Streamlit configuration
- `.gitignore` - Git ignore rules

## Features of Your Dashboard

✅ **Interactive Strategy Selection**
✅ **Real-time Strategy Visualization**
✅ **Customizable Parameters**
✅ **Comprehensive Backtesting**
✅ **Beautiful Analytics Dashboard**
✅ **Performance Metrics**
✅ **Multiple Chart Types**
✅ **Professional UI**

Your options backtester dashboard is ready for deployment!
