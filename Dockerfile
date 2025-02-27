# Use the Ubuntu 22.04 base image
FROM ubuntu:22.04

# Update the system and install necessary dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    sudo \
    python3.9 \
    python3-distutils \
    python3-pip \
    ffmpeg

# Upgrade pip and install openai-whisper
RUN pip install --upgrade pip
RUN pip install -U openai-whisper

# Set the working directory in the container
WORKDIR /app

# Copy your transcription script into the container
COPY transcribe_consensus.py /app/

# Set the default command to run your script
CMD ["python3", "-u", "transcribe_consensus.py"]
