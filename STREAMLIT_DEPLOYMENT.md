# Streamlit Cloud Deployment Instructions

## 🚀 Deploy NerveSpark to Streamlit Cloud

### Option 1: Full Featured Version (Recommended for Local)
**Main file:** `app.py`  
**Requirements:** `requirements.txt`  
**Python version:** 3.9+

### Option 2: Cloud Optimized Version (For Streamlit Cloud)
**Main file:** `app_cloud.py`  
**Requirements:** `requirements_cloud.txt`  
**Python version:** 3.9+

## 📝 Deployment Steps:

1. **Login to Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Connect your GitHub account

2. **Create New App**
   - Repository: `AnujGupta1606/NerveSpark`
   - Branch: `main`
   - Main file path: `app_cloud.py` (for cloud) or `app.py` (for full version)
   - Advanced: Set requirements file to `requirements_cloud.txt` (for cloud)

3. **Environment Variables** (Optional)
   - `STREAMLIT_CLOUD=true`
   - `PYTHONPATH=/mount/src`

## 🔧 Cloud Compatibility Features:

- **Automatic Detection**: App detects cloud environment
- **Fallback Systems**: Uses simplified RAG when ChromaDB fails
- **Minimal Dependencies**: Cloud version uses only essential packages
- **Sample Data**: Pre-loaded with demo recipes
- **Error Handling**: Graceful degradation for missing features

## 📊 What Works in Cloud:

✅ Recipe Search (with sample data)  
✅ Nutrition Analysis  
✅ Health Recommendations  
✅ User Profiles  
✅ BMI Calculator  
✅ Dietary Filtering  

## 🌐 Live Demo URLs:

- **Cloud Demo**: [Your Streamlit Cloud URL here]
- **Local Full**: http://localhost:8890

## 🛠️ Troubleshooting:

If deployment fails:
1. Use `app_cloud.py` instead of `app.py`
2. Use `requirements_cloud.txt` instead of `requirements.txt`
3. Check logs for import errors
4. Ensure all dependencies are in requirements file

## 📱 Mobile Responsive:

The app is optimized for both desktop and mobile viewing.

---

**Ready for Company Demo! 🎯**
