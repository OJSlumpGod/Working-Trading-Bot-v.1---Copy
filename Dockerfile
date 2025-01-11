# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables to prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CYTHON_WARNINGS=0

# Set working directory
WORKDIR /app

# Install system dependencies including build tools and dependencies to compile TA-Lib
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    libtool \
    autoconf \
    automake && \
    rm -rf /var/lib/apt/lists/*

# Download, build, and install TA-Lib from source
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzvf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Upgrade pip, setuptools, and wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with quiet mode to reduce log verbosity
RUN pip install --quiet --prefer-binary -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port (Railway provides the port via the PORT environment variable)
ENV PORT=5000
EXPOSE $PORT

# Define environment variable for Flask application entry point
ENV FLASK_APP=app.py

# Start the application using Gunicorn, binding to all interfaces and the specified port
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]
