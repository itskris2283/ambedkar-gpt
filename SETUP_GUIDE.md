# Quick Setup Guide

## Prerequisites Check

Before running the project, make sure you have:

### 1. Python 3.8+
```powershell
python --version
```

### 2. Ollama Installed and Running

**Download and Install:**
- Go to https://ollama.ai/download
- Download the Windows installer
- Run the installer

**Pull Mistral Model:**
```powershell
ollama pull mistral
```

**Verify Installation:**
```powershell
ollama list
```
You should see `mistral` in the list.

## Installation Steps

### 1. Activate Virtual Environment

**Windows PowerShell:**
```powershell
.\venv\Scripts\activate
```

**Windows Command Prompt:**
```cmd
.\venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 2. Install Dependencies (Already Done!)

The dependencies are already installed, but if you need to reinstall:
```powershell
pip install -r requirements.txt
```

## Running the Application

```powershell
python main.py
```

## Testing the System

Try these example questions:
1. "What does Dr. Ambedkar say about the shastras?"
2. "What is the real remedy according to the speech?"
3. "How does Dr. Ambedkar describe social reform?"
4. "What analogy does he use for social reform?"
5. "What is the real enemy mentioned in the speech?"

## Commands While Running

- `help` - Show example questions
- `quit` or `exit` - End the session

## Troubleshooting

### Ollama Not Found
Make sure Ollama is:
1. Installed
2. Mistral model is pulled (`ollama pull mistral`)
3. Service is running

### Virtual Environment Not Activated
You should see `(venv)` at the beginning of your command prompt.

### Import Errors
Make sure you're in the activated virtual environment and dependencies are installed.

## Next Steps for GitHub Submission

1. Initialize Git repository:
```powershell
git init
git add .
git commit -m "Initial commit: AmbedkarGPT RAG system"
```

2. Create GitHub repository named `AmbedkarGPT-Intern-Task`

3. Push to GitHub:
```powershell
git remote add origin https://github.com/YOUR_USERNAME/AmbedkarGPT-Intern-Task.git
git branch -M main
git push -u origin main
```

4. Submit the repository URL to Kalpit Pvt Ltd

---

**Note:** This project uses only FREE and LOCAL tools:
- ✅ No API keys required
- ✅ No cloud accounts needed
- ✅ Everything runs on your machine
