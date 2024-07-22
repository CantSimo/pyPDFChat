</a>

<!-- *The LangChain Chatbot is an AI chat interface for the open-source library LangChain. It provides conversational answers to questions about vector ingested documents.* -->

# 🚀 Installation

## Dev-Setup
Prerequisites:
- [Git](https://git-scm.com/downloads) - Free
- [Pinecone Database] (https://www.pinecone.io) - Free
- [OpenAI API Key](https://platform.openai.com/account/api-keys) - Billing Required

```

Reference [example.env] to create `.env` file
```python
OPENAI_API_KEY=
PINECONE_API_KEY=
PINECONE_ENVIRONMENT=
PINECONE_INDEX=
DOCS_TMP_PATH=
```

### Install Requirements

```python
poetry install
```

### Activate Environment
```python
poetry shell
```

### Run Startup
```python
python3 startup.py
```


# 🔧 Key Features

✅ Interactive Ingestion UI for files 

✅ Chat UI with source, temperature, vector_k, and other parameter changing abilities

# 💻 Contributing

If you would like to contribute to the pyPDFChat, please follow these steps:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Write tests for your changes
4. Implement your changes and ensure that all tests pass
5. Submit a pull request

# 🔨 License

The pyPDFChat is released under the [MIT License](https://opensource.org/licenses/MIT).

Maintained by Developers of [legalyze.ai](https://legalyze.ai)
