# Use of the Python 3.10.12 base image
FROM python:3.10.12

# Definition of the working directory in the container
WORKDIR /repo

# Copy the necessary files into the container
COPY app/frontend/streamlit_app.py /repo/
COPY app/backend/df_sample.pkl /repo/app/backend/
COPY src/ /repo/src/
COPY reports/exploratory-data-analysis.html /repo/reports/
COPY pyproject.toml poetry.lock /repo/

# Installation Poetry
RUN pip install --no-cache-dir poetry

# Install dependencies with Poetry
RUN poetry config virtualenvs.create false && poetry install --no-root --no-interaction --no-ansi

# Definition of the ROOT_DIR environment variable
ENV ROOT_DIR=/repo/
ENV PYTHONPATH="/repo/src"

# Exposure of the port on which the application will listen
EXPOSE 8501

# Command to execute the FastAPI application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
