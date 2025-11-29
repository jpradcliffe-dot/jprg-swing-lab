# Base image with Node + Python
FROM node:18-bullseye

# Install system dependencies for OpenCV & MediaPipe
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    ffmpeg \
    libsm6 libxext6 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy package.json first (for caching)
COPY package.json package-lock.json* ./

# Install Node dependencies
RUN npm install

# Copy full project
COPY . .

# Install Python dependencies for MediaPipe processor
RUN pip3 install mediapipe opencv-python numpy

# Expose Next.js port
EXPOSE 3000

# Start both services with one command
CMD ["npm", "run", "dev"]
