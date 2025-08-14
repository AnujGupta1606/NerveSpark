# NerveSpark - Intelligent Nutrition Assistant

A RAG (Retrieval-Augmented Generation) system that provides personalized meal suggestions based on dietary restrictions, allergies, health conditions, and nutritional goals.

## Features

- **Recipe Database Processing**: Store and analyze recipes with complete nutritional information
- **Dietary Restriction & Allergy Handling**: Filter recipes based on user restrictions and allergies
- **Health Condition-Based Suggestions**: Personalized recommendations for diabetes, hypertension, heart disease, etc.
- **Nutritional Goal Optimization**: Track and meet daily calorie, protein, carb, and fat targets
- **Intelligent Ingredient Substitution**: Replace ingredients while preserving taste and nutrition

## Tech Stack

- **Frontend**: Streamlit
- **Embeddings**: OpenAI / HuggingFace Sentence Transformers
- **Vector Database**: Chroma
- **Backend**: Python, FastAPI
- **Data Storage**: SQLite for user profiles, CSV for recipe data
- **Deployment**: Streamlit Cloud / HuggingFace Spaces

## Project Structure

```
NerveSpark/
├── app.py                    # Main Streamlit application
├── data/
│   ├── recipes.csv          # Recipe dataset
│   ├── nutrition_db.csv     # Nutritional information
│   └── substitutions.json   # Ingredient substitution rules
├── src/
│   ├── __init__.py
│   ├── data_processor.py    # Data cleaning and processing
│   ├── embeddings.py        # Text embedding generation
│   ├── vector_store.py      # Vector database operations
│   ├── health_logic.py      # Health condition and dietary logic
│   ├── substitution.py      # Ingredient substitution engine
│   └── rag_system.py        # RAG implementation
├── models/
│   └── user_profile.py      # User profile management
├── utils/
│   ├── __init__.py
│   ├── nutrition_calc.py    # Nutritional calculations
│   └── helpers.py           # Utility functions
├── tests/
│   └── test_rag_system.py   # Unit tests
├── requirements.txt         # Dependencies
├── config.py               # Configuration settings
└── setup_data.py          # Data setup script
```

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd NerveSpark
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up data**
   ```bash
   python setup_data.py
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 3-Day Development Plan

### Day 1: Data & Retrieval Setup
- [x] Recipe data collection and cleaning
- [x] Embedding generation with Sentence Transformers
- [x] Vector database setup with Chroma
- [x] Basic retrieval system

### Day 2: Health & Dietary Logic
- [x] User profile management
- [x] Dietary restriction filtering
- [x] Health condition matching
- [x] Ingredient substitution engine
- [x] RAG system integration

### Day 3: Frontend & Deployment
- [x] Streamlit UI development
- [x] Backend integration
- [x] Testing and optimization
- [x] Deployment preparation

## Configuration

Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_key_here
HUGGINGFACE_API_TOKEN=your_hf_token_here
```

## Evaluation Metrics

- **Retrieval Accuracy**: Precision and recall of recipe recommendations
- **Response Latency**: Time to generate suggestions
- **User Satisfaction**: Relevance of dietary recommendations
- **Nutritional Accuracy**: Precision of nutritional calculations

## Future Enhancements

- Multi-day meal planning
- Regional cuisine filters
- AI-generated cooking tips
- Integration with fitness trackers
- Social recipe sharing

## Authors

**Anuj Gupta** - *Full Stack Developer & AI Engineer*
- Email: satyam2465@gmail.com
- GitHub: https://github.com/AnujGupta1606
- LinkedIn: https://www.linkedin.com/in/anujgupta16/

## License

MIT License - see [LICENSE](LICENSE) file for details



## Acknowledgments

- HuggingFace for Sentence Transformers
- ChromaDB for vector database
- Streamlit for the amazing UI framework
- OpenAI for embedding models

---

**ANUJ GUPTA**
