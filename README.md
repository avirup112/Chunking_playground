# 🔪 Chunking Playground

An interactive Streamlit application for experimenting with different text chunking strategies and retrieval methods in RAG (Retrieval-Augmented Generation) pipelines. This educational tool helps developers, researchers, and students understand how different chunking and retrieval approaches affect information retrieval quality and LLM responses.

![Chunking Playground](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink&logoColor=white)

## 🎯 Features

### 📄 Document Processing
- **Multi-format Support**: Upload PDF, TXT, and DOCX files
- **Text Extraction**: Automatic text extraction with error handling
- **Document Analytics**: Character count, word count, and preview

### 🔪 Advanced Chunking Strategies

#### Basic Strategies
- **Fixed Length**: Equal-sized chunks (good for consistent processing)
- **Overlapping (Sliding Window)**: Preserves context at boundaries
- **Recursive Text Splitter**: Respects natural text boundaries

#### Advanced Strategies
- **Semantic Chunking**: Groups content by meaning using embeddings
- **Sentence-Based**: Splits by complete sentences
- **Paragraph-Based**: Maintains document structure
- **Heading-Based**: Follows document headings and sections
- **Topic-Based**: Uses K-means clustering for semantic grouping
- **Dynamic AI Chunking**: Uses OpenAI GPT for optimal boundaries

### 🔍 Comprehensive Retrieval Methods

#### Classical Methods
- **BM25 (Keyword)**: Traditional TF-IDF based keyword matching
- **FAISS (Dense)**: Fast vector similarity search
- **Chroma (Dense)**: Persistent vector database with metadata

#### Advanced Methods
- **Hybrid Retrieval**: Combines BM25 + dense search with adjustable weighting
- **MMR (Maximal Marginal Relevance)**: Balances relevance vs diversity
- **Reranking**: Two-stage retrieval with sophisticated reranking
- **Multi-Vector**: Multiple representations per chunk (original, summary, keywords)

### 🤖 LLM Integration
- **AI-Powered Answers**: Generate responses using retrieved chunks
- **Source Attribution**: Shows which chunks influenced the answer
- **Strategy Comparison**: Compare how different retrieval methods affect LLM responses
- **OpenAI Integration**: Secure API key handling

### 📊 Visualization & Analytics
- **Chunk Statistics**: Size distribution, count, and metadata
- **Retrieval Results**: Scored results with expandable details
- **Strategy Comparison**: Side-by-side performance analysis
- **Educational Insights**: Explanations of why strategies work

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd chunking_playground
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Open your browser** to `http://localhost:8501`

### Basic Usage

1. **Upload a document** (PDF, TXT, or DOCX)
2. **Select chunking strategy** from the sidebar
3. **Choose retriever type** and adjust parameters
4. **Enter a query** to test retrieval
5. **Optionally add OpenAI API key** for LLM features
6. **Enable comparison mode** to test all strategies

## 📁 Project Structure

```
chunking_playground/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration and constants
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── utils/                      # Core utilities
│   ├── file_loader.py         # Document text extraction
│   ├── chunking.py            # All chunking strategies
│   ├── retrievers.py          # All retrieval methods
│   ├── llm_integration.py     # OpenAI integration
│   └── visualization.py       # Charts and displays
│
├── components/                 # UI components
│   ├── sidebar.py             # Sidebar controls
│   └── display.py             # Main display components
│
└── data/                      # Data storage
    ├── uploads/               # Uploaded documents
    └── vectorstores/          # Local vector databases
```

### Customization
Edit `config.py` to modify:
- Default chunk sizes and parameters
- Supported file types
- Storage directories
- Strategy configurations

## 📚 Educational Use Cases

### For Students
- **Learn RAG fundamentals**: Understand chunking and retrieval concepts
- **Compare strategies**: See real-time impact of different approaches
- **Hands-on experimentation**: Test with your own documents

### For Researchers
- **Evaluate methods**: Compare retrieval quality across strategies
- **Optimize parameters**: Find best settings for your use case
- **Analyze trade-offs**: Understand speed vs accuracy implications

### For Developers
- **Prototype RAG systems**: Test before production implementation
- **Debug retrieval issues**: Visualize what's being retrieved
- **Benchmark performance**: Compare different approaches systematically

## 🎓 Key Learning Concepts

### Chunking Trade-offs
- **Size vs Context**: Larger chunks preserve context but may dilute relevance
- **Overlap vs Storage**: Overlapping chunks improve context but increase storage
- **Structure vs Flexibility**: Structured chunking (headings) vs flexible (semantic)

### Retrieval Trade-offs
- **Keyword vs Semantic**: BM25 for exact matches, embeddings for meaning
- **Speed vs Accuracy**: Simple methods are faster, complex methods more accurate
- **Relevance vs Diversity**: MMR balances finding relevant and diverse results

### Advanced Techniques
- **Hybrid Approaches**: Combining multiple methods often works best
- **Two-stage Retrieval**: Initial broad search + refined reranking
- **Multi-representation**: Different views of the same content

## 🛠️ Technical Details

### Dependencies
- **Streamlit**: Web interface and visualization
- **LangChain**: Text splitting and processing utilities
- **Sentence Transformers**: Embedding generation
- **FAISS**: Fast vector similarity search
- **ChromaDB**: Persistent vector storage
- **OpenAI**: LLM integration for advanced features
- **Scikit-learn**: Clustering and similarity metrics

### Performance Considerations
- **Memory Usage**: Large documents may require chunking before processing
- **API Costs**: Dynamic AI chunking uses OpenAI API (optional)
- **Processing Time**: Semantic methods are slower but more accurate

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- **New chunking strategies**: Implement additional text splitting methods
- **More retrievers**: Add support for other vector databases
- **Evaluation metrics**: Add quantitative retrieval quality measures
- **UI enhancements**: Improve visualization and user experience
- **Documentation**: Add more educational content and examples

### Development Setup
```bash
# Install in development mode
pip install -e .

# Run tests (if available)
pytest

# Format code
black .
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain** for excellent text processing utilities
- **Streamlit** for making interactive apps simple
- **Sentence Transformers** for high-quality embeddings
- **OpenAI** for powerful language models
- **FAISS** and **ChromaDB** for efficient vector search

## 📞 Support

- **Issues**: Report bugs or request features via GitHub Issues
- **Discussions**: Join community discussions for questions and ideas
- **Documentation**: Check the code comments for detailed explanations

## 🔮 Future Roadmap

- [ ] **Graph-based retrieval**: Implement knowledge graph approaches
- [ ] **Multi-modal support**: Add image and table processing
- [ ] **Evaluation metrics**: Add BLEU, ROUGE, and custom metrics
- [ ] **Batch processing**: Support for multiple document analysis
- [ ] **Export functionality**: Save results and configurations
- [ ] **Advanced visualizations**: Interactive chunk and similarity plots
- [ ] **Cloud deployment**: One-click deployment options

---

**Happy Chunking!** 🔪✨

*Built with ❤️ for the RAG community*