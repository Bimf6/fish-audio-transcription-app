# 🚀 Deployment Guide: Fish Audio Transcription App

This guide will help you deploy the Fish Audio Transcription app online using Streamlit Community Cloud (free).

## Prerequisites

- GitHub account
- Fish Audio API key

## Step 1: Push to GitHub

1. **Create a new repository on GitHub:**
   - Go to [github.com](https://github.com) and create a new repository
   - Name it something like `fish-audio-transcription-app`
   - Make it public (required for Streamlit Community Cloud free tier)
   - Don't initialize with README (we already have files)

2. **Connect your local repository to GitHub:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

## Step 2: Deploy to Streamlit Community Cloud

1. **Go to Streamlit Community Cloud:**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Deploy your app:**
   - Click "New app"
   - Select your GitHub repository
   - Set the main file path to: `app.py`
   - Click "Deploy!"

3. **Configure secrets (important for API key security):**
   - In your app dashboard, click on "Settings" → "Secrets"
   - Add your Fish Audio API key:
   ```toml
   FISH_AUDIO_API_KEY = "your_actual_api_key_here"
   ```
   - Save the secrets

## Step 3: Your App is Live! 🎉

Your app will be available at: `https://YOUR_USERNAME-YOUR_REPO_NAME-app-HASH.streamlit.app/`

## Alternative Deployment Options

### Option 1: Heroku (Paid)
- More control over environment
- Better for production apps
- Requires credit card for verification

### Option 2: Railway (Free tier available)
- Simple deployment
- Good performance
- Easy scaling

### Option 3: Render (Free tier available)
- Automatic deployments from GitHub
- Good for static and web apps

## App Features

Your deployed app will include:
- ✅ File upload for audio transcription
- ✅ Multi-language support (Auto-detect, Mandarin, English, Cantonese)
- ✅ Secure API key management
- ✅ Download transcriptions as text files
- ✅ Modern, responsive UI
- ✅ Error handling and progress indicators

## Troubleshooting

**Common Issues:**

1. **App won't start:**
   - Check the logs in Streamlit Cloud dashboard
   - Ensure all dependencies are in requirements.txt

2. **API key not working:**
   - Verify the API key is correctly set in Secrets
   - Check that the key has sufficient credits

3. **Import errors:**
   - All required packages are included in requirements.txt
   - The fish_audio_sdk is included locally

## Security Notes

- ✅ API keys are stored securely in Streamlit secrets
- ✅ Keys are not exposed in the code
- ✅ Secrets are not committed to Git
- ✅ Users can input their own API keys for additional security

## Need Help?

If you encounter any issues:
1. Check the Streamlit Community Cloud documentation
2. Review the app logs in the Streamlit dashboard
3. Ensure your GitHub repository is public
4. Verify all files are committed and pushed to GitHub