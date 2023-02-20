# Use an official Python runtime as the parent image
FROM python:3.11.1

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install the necessary dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
# Copy the rest of the files to the container
COPY . .

# Specify the command to run when the container starts
CMD ["flask", "run", "--host=0.0.0.0"]