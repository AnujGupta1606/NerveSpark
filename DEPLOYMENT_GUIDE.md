# Streamlit Cloud Deployment Guide

## 🚀 Deploy NerveSpark on Streamlit Cloud

### Step 1: Prepare for Deployment
1. Ensure your GitHub repo is public
2. Add .streamlit/config.toml for configuration
3. Update requirements.txt for cloud compatibility

### Step 2: Streamlit Cloud Setup
1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: AnujGupta1606/NerveSpark
5. Main file path: app.py
6. Click "Deploy!"

### Step 3: Configuration
Add these files for better deployment:

**.streamlit/config.toml:**
```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false
```

**runtime.txt:**
```
python-3.9
```

### Step 4: Environment Variables
In Streamlit Cloud dashboard:
- Go to Settings > Secrets
- Add any API keys if needed

### Benefits of Streamlit Cloud:
✅ Free hosting
✅ Automatic deployments from GitHub
✅ Custom domain support
✅ SSL certificate included
✅ Professional presentation

### Expected URL:
https://nervespark.streamlit.app/
