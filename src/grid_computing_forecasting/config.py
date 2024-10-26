"""Load environment variables from .env"""
import os
from dotenv import load_dotenv


# Load environment variables from the .env file
load_dotenv()

# Access environment variables
root_dir = os.getenv('ROOT_DIR')
database_path = os.getenv('DATABASE_PATH')
python_path = os.getenv('PYTHONPATH')
