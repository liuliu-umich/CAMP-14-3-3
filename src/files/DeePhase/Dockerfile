# Use an official Python 3.8 runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
# The dot (.) represents the current directory on the host
COPY . /app

# Install ncbi-blast+
RUN apt-get update && apt-get install -y ncbi-blast+ && apt-get clean

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
