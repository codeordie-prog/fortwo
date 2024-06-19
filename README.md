
# Ask FortyTwo

## Overview

**Ask FortyTwo** is an intelligent digital assistant that leverages the power of GPT models and Retrieval Augmented Generation (RAG) to answer user queries. It can handle various tasks, including document querying, code requests, math assistance, and writing help. Additionally, it can query a GitHub repository for specific information.

## Features

- **Chat Interface**: Interact with the AI through a chat interface.
- **Document Upload**: Upload PDF, TXT, and CSV files for querying.
- **GitHub Repository Query**: Enter a repository URL and query its contents.
- **Conversational Memory**: The AI maintains context throughout the conversation.

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management. Follow the steps below to set up and run the application.

### Prerequisites

- Python 3.7 or higher
- Poetry

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/codeordie-prog/askfortytwo.git
   cd ask-fortytwo

2. **install dependencies**
    poetry install

3. **Set Up Environment Variables**

    Create a .env file in the project root and add your OpenAI API key:


    OPENAI_API_KEY=your_openai_api_key_here

4. **Run the Application**

    poetry run streamlit run app.py


# Contributing
    We welcome contributions! Please fork the repository and create a pull request.

    Fork the repository
    Create a new branch (git checkout -b feature/your-feature)
    Commit your changes (git commit -am 'Add new feature')
    Push to the branch (git push origin feature/your-feature)
    Create a new Pull Request
    License
    This project is licensed under the MIT License. See the LICENSE file for details.

# Acknowledgments
Inspired by the need to utilize Retrieval Augmented Generation in data querying.

