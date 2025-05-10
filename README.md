# Books Guru API

A FastAPI backend for analyzing books and generating character relationship networks.

## 🚀 Features

- **Text Analysis**: Extract characters and their relationships from books
- **Network Visualization**: Generate interactive network graphs
- **Character Details**: Provide character mentions and descriptions

## 🛠️ Tech Stack

- **FastAPI**: Modern, high-performance web framework
- **NetworkX**: Graph manipulation and analysis
- **OpenAI**: Character identification and relationship extraction
- **D3.js**: Interactive visualizations

## 📋 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyse-book/{book_id}/characters` | GET | Get character list with details |
| `/analyse-book/{book_id}/html-visualization` | GET | Get interactive network visualization |

## ⚡ Quick Start

1. Clone the repository
   ```bash
   git clone git@github.com:username/books-guru-api.git
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key
   ```

4. Run the server
   ```bash
   uvicorn main:app --reload
   ```

5. Visit `http://localhost:8000/docs` for interactive API documentation

## 🔄 Processing Pipeline

1. **Text Retrieval**: Fetch book content from Project Gutenberg
2. **Text Chunking**: Intelligent document chunking for better context
3. **Character Identification**: Extract character names using ML
4. **Relationship Mapping**: Determine character interactions
5. **Graph Generation**: Create character network graphs
6. **Visualization Rendering**: Generate interactive D3.js visualization

## 🔧 Known Improvements

- **Enhanced Chunking**: Implement semantic chunking for improved context
- **Caching Layer**: Add Redis caching for faster repeated analyses
- **Character Clustering**: Group related characters for better visualization
- **Sentiment Analysis**: Analyze emotional context of character interactions
- **Multiple Languages**: Support for books in various languages
- **Custom NER Models**: Fine-tuned models for character detection
- **Performance Optimization**: Async processing for large books

## 📊 Deployment

The API is deployed at: `https://books-guru.onrender.com/`

## 📝 License

MIT License

## 👨‍💻 Contributors

- [Sahil khan](https://github.com/sahilk01)
