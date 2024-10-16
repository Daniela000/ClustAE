FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt (or other dependency files) into the image
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire src folder into the /app directory in the container
COPY src/ /app/src/

# Set the working directory
WORKDIR /app/src

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run app.py when the container launches
CMD python app.py