# Cloud Deployment Configuration for NerveSpark
# This ensures Streamlit Cloud runs exactly like local port 8892

# Use the main app.py file (same as local)
# Requirements: requirements.txt (full dependencies)
# Python: 3.9+

# Repository: https://github.com/AnujGupta1606/NerveSpark
# Main app file: app.py
# Branch: main

# This will run: streamlit run app.py
# Which is identical to: streamlit run app.py --server.port 8892 (locally)

# All features enabled:
# - Full RAG system with ChromaDB
# - Sentence Transformers embeddings
# - Complete health logic
# - Modern UI with recipe cards
# - Search functionality working
# - User profiles and recommendations

print("Streamlit Cloud will run app.py with full functionality!")
