# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install git and other necessary packages
RUN apt-get update && apt-get install -y git

# Set environment variables
ENV POETRY_VERSION=1.8.3
ENV GIT_PYTHON_REFRESH=quiet
ENV GIT_PYTHON_GIT_EXECUTABLE=/usr/bin/git

# Install Poetry
RUN pip install poetry==$POETRY_VERSION

# Set working directory
WORKDIR /app

# Copy only the dependency files first for better caching
COPY pyproject.toml ./

# Increase Poetry timeout and install dependencies
ENV POETRY_HTTP_TIMEOUT=600
RUN poetry install

# Copy the rest of the project files
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["poetry", "run", "streamlit", "run", "streamlitapp/fortytwo.py"]