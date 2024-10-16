FROM python:3.10

# Set the working directory
WORKDIR /src

# Copy the current directory contents into the container at /src
COPY . ./

# Install the required dependencies
RUN pip install -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run app.py when the container launches
CMD python app.py