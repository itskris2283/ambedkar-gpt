"""
AmbedkarGPT - A RAG-based Q&A System
Built for Kalpit Pvt Ltd, UK - AI Intern Assignment

This system uses:
- LangChain for RAG orchestration
- ChromaDB for vector storage
- HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2)
- Ollama with Mistral 7B for LLM responses
"""

import os
import sys
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA


def setup_vector_store(file_path="speech.txt"):
    """
    Load the speech text, split it into chunks, create embeddings,
    and store them in a ChromaDB vector store.
    
    Args:
        file_path (str): Path to the speech text file
        
    Returns:
        Chroma: The initialized vector store
    """
    print("📚 Loading document...")
    
    # Step 1: Load the text file
    try:
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        print(f"✓ Loaded {len(documents)} document(s)")
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found!")
        sys.exit(1)
    
    # Step 2: Split the text into manageable chunks
    print("✂️  Splitting text into chunks...")
    text_splitter = CharacterTextSplitter(
        chunk_size=200,  # Smaller chunks for better retrieval
        chunk_overlap=50,  # Overlap between chunks to maintain context
        separator=". "  # Split on sentences
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✓ Created {len(chunks)} text chunk(s)")
    
    # Step 3: Create embeddings using HuggingFace
    print("🔢 Creating embeddings (this may take a moment on first run)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # Use CPU (change to 'cuda' if you have GPU)
    )
    print("✓ Embeddings model loaded")
    
    # Step 4: Create and populate the vector store
    print("💾 Creating vector store with ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"  # Local storage directory
    )
    print("✓ Vector store created and populated")
    
    return vectorstore


def initialize_llm():
    """
    Initialize the Ollama LLM with Mistral 7B model.
    
    Returns:
        Ollama: The initialized LLM
    """
    print("🤖 Initializing Ollama with Mistral 7B...")
    
    try:
        llm = Ollama(
            model="mistral",
            temperature=0.7,  # Controls randomness (0 = deterministic, 1 = creative)
        )
        print("✓ LLM initialized successfully")
        return llm
    except Exception as e:
        print(f"❌ Error initializing Ollama: {e}")
        print("\n💡 Make sure Ollama is installed and running:")
        print("   1. Install: https://ollama.ai/download")
        print("   2. Run: ollama pull mistral")
        print("   3. Verify: ollama list")
        sys.exit(1)


def create_qa_chain(vectorstore, llm):
    """
    Create a RetrievalQA chain that combines the vector store and LLM.
    
    Args:
        vectorstore (Chroma): The vector store containing document embeddings
        llm (Ollama): The language model
        
    Returns:
        RetrievalQA: The complete QA chain
    """
    print("🔗 Creating RetrievalQA chain...")
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" method passes all retrieved docs to LLM
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 2}  # Retrieve top 2 most relevant chunks
        ),
        return_source_documents=True  # Include source chunks in response
    )
    
    print("✓ QA chain created successfully")
    return qa_chain


def ask_question(qa_chain, question):
    """
    Ask a question and get an answer from the RAG system.
    
    Args:
        qa_chain (RetrievalQA): The QA chain
        question (str): The user's question
        
    Returns:
        dict: Response containing answer and source documents
    """
    print(f"\n❓ Question: {question}")
    print("🔍 Retrieving relevant context and generating answer...\n")
    
    try:
        response = qa_chain.invoke({"query": question})
        return response
    except Exception as e:
        print(f"❌ Error generating answer: {e}")
        return None


def main():
    """
    Main function to run the AmbedkarGPT Q&A system.
    """
    print("=" * 70)
    print("🎓 Welcome to AmbedkarGPT - Dr. B.R. Ambedkar Speech Q&A System")
    print("=" * 70)
    print()
    
    # Step 1: Set up the vector store
    vectorstore = setup_vector_store("speech.txt")
    print()
    
    # Step 2: Initialize the LLM
    llm = initialize_llm()
    print()
    
    # Step 3: Create the QA chain
    qa_chain = create_qa_chain(vectorstore, llm)
    print()
    
    print("=" * 70)
    print("✅ System Ready! You can now ask questions about the speech.")
    print("=" * 70)
    print()
    print("💡 Tips:")
    print("   - Ask questions about caste, shastras, or social reform")
    print("   - Type 'quit' or 'exit' to end the session")
    print("   - Type 'help' for example questions")
    print()
    
    # Interactive Q&A loop
    while True:
        try:
            # Get user input
            user_question = input("🗣️  Your Question: ").strip()
            
            # Handle special commands
            if user_question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using AmbedkarGPT. Goodbye!")
                break
            
            if user_question.lower() == 'help':
                print("\n📖 Example Questions:")
                print("   - What does Dr. Ambedkar say about the shastras?")
                print("   - What is the real remedy according to the speech?")
                print("   - How does Dr. Ambedkar describe social reform?")
                print("   - What is the real enemy mentioned in the speech?")
                print("   - What analogy does Dr. Ambedkar use for social reform?")
                print()
                continue
            
            if not user_question:
                print("⚠️  Please enter a question.\n")
                continue
            
            # Get answer from the RAG system
            response = ask_question(qa_chain, user_question)
            
            if response:
                # Display the answer
                print("💬 Answer:")
                print("-" * 70)
                print(response['result'])
                print("-" * 70)
                
                # Optionally display source chunks (for debugging/verification)
                if response.get('source_documents'):
                    print(f"\n📄 Based on {len(response['source_documents'])} relevant text chunk(s)")
                
                print()
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}\n")


if __name__ == "__main__":
    main()
