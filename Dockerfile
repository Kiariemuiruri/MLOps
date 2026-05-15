# Step 1: Use an official, lightweight Python image
FROM python:3.10-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Install system dependencies required for ML packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy and install python dependencies first (allows Docker caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of project files into the container
COPY . .

# Step 6: Expose the port FastAPI will run on
EXPOSE 5000

# Step 7: Command to run your FastAPI application using, python3 or Uvicorn
CMD ["python3", "app.py"]