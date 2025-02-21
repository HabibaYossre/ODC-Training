# Use TensorFlow official image
FROM tensorflow/tensorflow:latest

# Set working directory
WORKDIR /MLOps

# Copy project files into the container
COPY train.py /MLOps/train.py

# Install dependencies 
RUN pip install numpy

# Run the training script
CMD ["python", "train.py"]
