
# Chatwith42 

Overview

![ask42](https://github.com/codeordie-prog/fortwo/blob/master/streamlitapp/logo/stimage.jfif)

find the app at https://chatwith42@streamlit.app

Chatwith42 is an AI Assistant that leverages Retrieval Augmented Generation (RAG) to answer queries about your documents in .pdf, .txt,.csv,.jpg,.png or .jpeg formats. The assistant, named 42, is inspired by the answer to the ultimate question in "The Hitchhiker's Guide to the Galaxy" and is designed to provide accurate and helpful responses to a wide range of queries, including coding requests, math assistance, and document writing.

# Features
1. **Query documents** in .pdf, .txt, or .csv formats.
2. **Interact with GitHub python repositories.**
3. **Query web information**.
4. **Engage in general chat sessions with the AI assistant.**

### Requirements
Install the necessary dependencies using: 

   
   ```pip install streamlit langchain streamlit-chat gitpython requests lxml pillow pydantic```

# Usage

**Running the Application**
To run the application, use Streamlit:

   ```streamlit run fortytwo.py```

# Sidebar Options

Clear Message History: Clears the chat history.
Upload Files: Upload documents in .pdf, .txt, or .csv formats.
GitHub Repository: Enter the URL of a GitHub repository to query its contents.
Query Web Section: Interact with web information by entering a URL and a document name.
Chat Interface
User Query: Enter your query in the chat input field.
AI Responses: The assistant will respond with relevant information from the provided documents or general knowledge.

# Configuration

# Document Loader
The application supports loading documents from various formats:

.pdf: Uses PyPDFLoader.
.txt: Uses TextLoader.
.csv: Uses CSVLoader.

# Text Splitting
Documents are split into manageable chunks using RecursiveCharacterTextSplitter.

# Embeddings
Embeddings are generated using HuggingFaceEmbeddings with the model all-MiniLM-L6-v2.

# Vector Database
The document chunks and embeddings are stored in a vector database (DocArrayInMemorySearch) for efficient retrieval.

# Functionality

GitHub Repository Query
To query a python GitHub repository, enter the repository URL in the sidebar. The code will clone the repository, load the relevant files, and prepare them for querying.

Web Query
To query web information, enter the URL and a name for the web document. The assistant will process the webpage and make it available for querying.

# Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

# License
This project is licensed under the MIT License. See the LICENSE file for details.

# Contact
For help or issues, please visit the GitHub repository.

This README file provides a clear and structured overview of your project, its features, requirements, and usage instructions. Feel free to modify it further to suit your specific needs.
