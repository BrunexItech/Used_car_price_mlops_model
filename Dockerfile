FROM python:3.9-slim

WORKDIR /app

#Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copy project files
COPY . .

#Create models directory and copy trained models
RUN mkdir -p models
COPY models/ ./models/

#Install the package
RUN pip install -e .

#Expose port 
EXPOSE 8000


#Command to run the API
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8010"]
