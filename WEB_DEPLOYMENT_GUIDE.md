# 🌐 Web Deployment Guide

## Quick Start

To expose your ML CSV Analyzer on the web:

```bash
./deploy_web.sh
```

## What Happens

1. **Virtual Environment**: Script activates your virtual environment
2. **Dependencies**: Installs any missing packages
3. **Streamlit**: Starts your app on localhost:8501
4. **ngrok Tunnel**: Creates a public URL like `https://abc123_SK.ngrok.io`

## Expected Output

```
🌐 ML CSV Analyzer - Web Deployment
==================================

🔧 Activating virtual environment...
✅ All dependencies ready!

🔑 Checking ngrok authentication...
✅ ngrok is authenticated

🚀 Starting Streamlit application...
⏳ Waiting for Streamlit to initialize...
✅ Streamlit is running on localhost:8501

🌐 Creating ngrok tunnel...
📱 Your app will be accessible from anywhere on the internet!

ngrok                                                          

Session Status                online                           
Account                       Your Account (Plan: Free)        
Version                       3.x.x                           
Region                        United States (us)               
Web Interface                 http://127.0.0.1:4040           
Forwarding                    https://abc123_SK.ngrok.io -> http://localhost:8501

Connections                   ttl     opn     rt1     rt5     p50     p90 
                              0       0       0.00    0.00    0.00    0.00
```

## Your Public URL

Your app will be accessible at: **`https://[random-id]_SK.ngrok.io`**

## Security Notes ⚠️

- **Public Access**: Anyone with the URL can access your app
- **Data Privacy**: No data is stored permanently (in-memory only)
- **Laptop Dependency**: Must keep your laptop on and connected
- **Session Limits**: Free ngrok has 2-hour session limits
- **Custom URLs**: The "_SK" suffix requires ngrok Pro plan, otherwise falls back to standard format

## Monitoring

- **ngrok Dashboard**: Visit `http://127.0.0.1:4040` for request logs
- **Streamlit Logs**: Watch your terminal for application logs

## To Stop

Press `Ctrl+C` in the terminal to stop both ngrok and Streamlit

## Sharing Your App

Send the `https://[random-id]_SK.ngrok.io` URL to anyone to let them:
- Upload CSV files
- Train ML models  
- Download results
- View visualizations

## Next Steps to GCP

Once you're ready for production:

1. **Google Cloud Run**: Containerized deployment
2. **App Engine**: Managed platform deployment  
3. **Compute Engine**: VM-based deployment
4. **Custom Domain**: Point your domain to the service

## Troubleshooting

- **Port 8501 in use**: Kill existing Streamlit processes
- **ngrok not found**: Install with `brew install ngrok`
- **Auth issues**: Re-run the auth token command
- **App errors**: Check the terminal output for Python errors 