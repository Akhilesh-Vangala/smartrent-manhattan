# Streamlit Cloud Deployment Guide

## Quick Deployment Steps

1. Go to: https://share.streamlit.io
2. Sign in with GitHub (use your Akhilesh-Vangala account)
3. Click "New app"
4. Fill in the form:
   - **Repository**: `Akhilesh-Vangala/smartrent-manhattan`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app/Home.py`
   - **App URL**: `smartrent-manhattan` (or your choice)
5. Click "Deploy"
6. Wait 2-5 minutes for build to complete

## Repository Information

- **Repository URL**: https://github.com/Akhilesh-Vangala/smartrent-manhattan
- **Main File**: `streamlit_app/Home.py`
- **Branch**: `main`
- **Visibility**: Private

## Required Files (All Present)

- `streamlit_app/Home.py` - Main entry point
- `streamlit_app/Predict.py` - Prediction page
- `streamlit_app/Map.py` - Map visualization
- `streamlit_app/Interpret.py` - SHAP interpretability
- `data/processed/cleaned_manhattan.csv` - Processed dataset
- `models/best_model.pkl` - Trained model
- `requirements.txt` - Dependencies
- `src/` - All source modules

## Troubleshooting

If deployment fails:
1. Check build logs in Streamlit Cloud dashboard
2. Verify all files are committed to GitHub
3. Ensure `requirements.txt` is correct
4. Check that model files are under 100MB (GitHub limit)

## After Deployment

Your app will be live at:
`https://smartrent-manhattan.streamlit.app`

Or your custom URL if you chose a different name.

