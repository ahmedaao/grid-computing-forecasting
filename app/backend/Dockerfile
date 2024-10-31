# Use of the Python 3.10.12 base image
FROM python:3.10.12

# Definition of the working directory in the container
WORKDIR /repo

# Copy the necessary files into the container
COPY app/backend/fastapi_app.py /repo/
COPY app/backend/df_sample.pkl /repo/app/backend/
COPY src/ /repo/src/
COPY models/dataset_cls.pth /repo/models/
COPY models/xgboost.pkl /repo/models/
COPY pyproject.toml poetry.lock /repo/

# Installation Poetry
RUN pip install --no-cache-dir poetry

# Install dependencies with Poetry
RUN poetry config virtualenvs.create false && poetry install --no-root --no-interaction --no-ansi

# Definition of the ROOT_DIR environment variable
ENV ROOT_DIR=/repo/
ENV PYTHONPATH="/repo/src"

# Exposure of the port on which the application will listen
EXPOSE 8000

# Command to execute the FastAPI application
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
