FROM python:3.10-slim

# Install system dependencies required for dlib, face_recognition and opencv
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements file to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir keeps the docker image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on (7860 for Hugging Face)
EXPOSE 7860

# Start the FastAPI server using Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
