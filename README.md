# AmbedkarGPT - Intern Task

A Retrieval-Augmented Generation (RAG) based Q&A system that answers questions about Dr. B.R. Ambedkar's speech on the "Annihilation of Caste."

**Built for:** Kalpit Pvt Ltd, UK - AI Intern Hiring Assignment

---

## 🎯 Project Overview

This project demonstrates a functional RAG pipeline that:
1. ✅ Loads text from `speech.txt`
2. ✅ Splits the text into manageable chunks
3. ✅ Creates embeddings using HuggingFace models
4. ✅ Stores embeddings in ChromaDB (local vector store)
5. ✅ Retrieves relevant chunks based on user questions
6. ✅ Generates answers using Ollama with Mistral 7B

---

## 🛠️ Technology Stack

| Component | Technology | Description |
|-----------|-----------|-------------|
| **Framework** | LangChain | RAG pipeline orchestration |
| **Vector DB** | ChromaDB | Local vector storage |
| **Embeddings** | HuggingFace | sentence-transformers/all-MiniLM-L6-v2 |
| **LLM** | Ollama | Mistral 7B model |
| **Language** | Python 3.8+ | Core programming language |

**✨ All components are 100% free and run locally - no API keys or accounts needed!**

---

## 📋 Prerequisites

Before running this project, ensure you have:

1. **Python 3.8 or higher**
   - Check version: `python --version`
   - Download: [python.org](https://www.python.org/downloads/)

2. **Ollama with Mistral 7B**
   - **Windows:** Download installer from [ollama.ai/download](https://ollama.ai/download)
   - **Linux/Mac:**
     ```bash
     curl -fsSL https://ollama.ai/install.sh | sh
     ```
   - **Pull Mistral model:**
     ```bash
     ollama pull mistral
     ```
   - **Verify installation:**
     ```bash
     ollama list
     ```

---

## 🚀 Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task
```

### Step 2: Create a Virtual Environment

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** The first run will download the sentence-transformers model (~80MB). This is a one-time download.

### Step 4: Verify Ollama is Running

Make sure Ollama is installed and the Mistral model is available:

```bash
ollama list
```

You should see `mistral` in the list.

---

## 🎮 Usage

### Running the Q&A System

```bash
python main.py
```

### Interactive Session

Once the system initializes, you'll see:

```
🎓 Welcome to AmbedkarGPT - Dr. B.R. Ambedkar Speech Q&A System
```

You can now ask questions about the speech:

**Example Questions:**
- "What does Dr. Ambedkar say about the shastras?"
- "What is the real remedy according to the speech?"
- "How does Dr. Ambedkar describe social reform?"
- "What analogy does he use for social reform?"

**Commands:**
- Type `help` - See example questions
- Type `quit` or `exit` - End the session

---

## 📁 Project Structure

```
AmbedkarGPT-Intern-Task/
│
├── main.py              # Main RAG pipeline implementation
├── speech.txt           # Source text from Dr. Ambedkar's speech
├── requirements.txt     # Python dependencies
├── README.md           # This file
│
├── venv/               # Virtual environment (created during setup)
└── chroma_db/          # ChromaDB storage (created on first run)
```

---

## 🔧 How It Works

### 1. Document Loading
The system loads `speech.txt` using LangChain's `TextLoader`.

### 2. Text Chunking
Text is split into chunks of ~500 characters with 50-character overlap using `CharacterTextSplitter`. This maintains context between chunks.

### 3. Embedding Creation
Each chunk is converted to a numerical vector (embedding) using the `sentence-transformers/all-MiniLM-L6-v2` model. This allows semantic similarity search.

### 4. Vector Storage
Embeddings are stored in ChromaDB, a local vector database that persists data in the `./chroma_db` directory.

### 5. Retrieval
When you ask a question:
- Your question is embedded using the same model
- The system finds the 3 most similar text chunks
- These chunks provide relevant context

### 6. Answer Generation
The retrieved context and your question are sent to Mistral 7B via Ollama, which generates a natural language answer.

---

## 🐛 Troubleshooting

### Issue: "Error initializing Ollama"

**Solution:**
1. Verify Ollama is installed: `ollama --version`
2. Check Mistral is downloaded: `ollama list`
3. If not, run: `ollama pull mistral`
4. Ensure Ollama service is running

### Issue: "File 'speech.txt' not found"

**Solution:**
- Make sure `speech.txt` exists in the same directory as `main.py`
- Check you're running the script from the correct directory

### Issue: "Module not found" errors

**Solution:**
1. Activate virtual environment:
   - Windows: `.\venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
2. Reinstall dependencies: `pip install -r requirements.txt`

### Issue: Slow first run

**Explanation:**
- First run downloads the embedding model (~80MB)
- Subsequent runs will be much faster
- This is normal and expected behavior

---

## 📦 Dependencies

```
langchain==0.1.0              # RAG framework
langchain-community==0.0.10   # Community integrations
chromadb==0.4.22              # Vector database
sentence-transformers==2.3.1   # Embedding models
ollama==0.1.6                 # Ollama Python client
```

---

## 🎓 Technical Highlights

### RAG Pipeline Components:

1. **Document Loader:** Loads and parses text files
2. **Text Splitter:** Intelligently chunks text while preserving context
3. **Embeddings:** Converts text to semantic vectors for similarity search
4. **Vector Store:** Efficiently stores and retrieves embedded chunks
5. **LLM:** Generates human-like answers based on retrieved context
6. **Chain:** Orchestrates the entire retrieval and generation process

### Key Features:

- ✅ **No API Keys Required:** Everything runs locally
- ✅ **Persistent Storage:** ChromaDB saves embeddings for future use
- ✅ **Context-Aware:** Maintains semantic meaning across chunks
- ✅ **Interactive CLI:** User-friendly command-line interface
- ✅ **Well-Commented Code:** Clear documentation for learning
- ✅ **Error Handling:** Graceful handling of common issues

---

## 🧪 Testing

To verify the system works correctly, try these test questions:

```python
# Test 1: Direct reference
"What is the real enemy according to Dr. Ambedkar?"
Expected: Should mention "belief in the shastras"

# Test 2: Conceptual understanding
"What analogy does Dr. Ambedkar use to describe social reform?"
Expected: Should mention "gardener pruning leaves and branches"

# Test 3: Logical inference
"What choice does Dr. Ambedkar present regarding caste and shastras?"
Expected: Should mention you cannot have both
```

---

## 📝 Assignment Requirements Checklist

- ✅ Python 3.8+
- ✅ LangChain framework for RAG
- ✅ ChromaDB for vector storage
- ✅ HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2)
- ✅ Ollama with Mistral 7B
- ✅ Well-commented code
- ✅ requirements.txt
- ✅ Detailed README.md
- ✅ speech.txt included
- ✅ Public GitHub repository

---

## 🚀 Future Enhancements

Potential improvements for a production system:

- 📊 Add conversation history/memory
- 🎨 Create a web interface (Gradio/Streamlit)
- 📈 Implement query analytics
- 🔍 Add source citation in answers
- 💾 Support for multiple documents
- 🌐 Multi-language support
- 📱 Deploy as a REST API

---

## 👨‍💻 Author

**Created for:** Kalpit Pvt Ltd, UK - AI Intern Assignment  
**Date:** November 2025  
**Contact:** [Your GitHub Profile]

---

## 📄 License

This project is created for educational and assignment purposes.

---

## 🙏 Acknowledgments

- **Dr. B.R. Ambedkar** for the profound insights in "Annihilation of Caste"
- **LangChain** for the excellent RAG framework
- **Ollama** for making LLMs accessible locally
- **HuggingFace** for open-source embeddings
- **ChromaDB** for the lightweight vector database

---

## 📞 Support

If you encounter any issues or have questions:

1. Check the **Troubleshooting** section above
2. Verify all prerequisites are installed correctly
3. Ensure you're in the activated virtual environment
4. Check that Ollama is running with Mistral model loaded

---

**Thank you for reviewing this submission! 🎓**
