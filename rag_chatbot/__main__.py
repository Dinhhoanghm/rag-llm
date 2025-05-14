import argparse
import llama_index
from dotenv import load_dotenv
from .ui import LocalChatbotUI
from .pipeline import LocalRAGPipeline
from .logger import Logger
import os
import requests
import time
import subprocess
import sys

load_dotenv()

# CONSTANTS
LOG_FILE = "logging.log"
DATA_DIR = "data/data"
AVATAR_IMAGES = ["./assets/user.png", "./assets/bot.png"]

# PARSER
parser = argparse.ArgumentParser()
parser.add_argument(
    "--host", type=str, default="localhost",
    help="Set host to local or in docker container"
)
parser.add_argument(
    "--share", action='store_true',
    help="Share gradio app"
)
parser.add_argument(
    "--model", type=str, default="openai",  # Default to OpenAI for Kaggle
    help="Set default model type: ollama or openai"
)
args = parser.parse_args()

# LOGGER
llama_index.core.set_global_handler("simple")
logger = Logger(LOG_FILE)
logger.reset_logs()

# Check if we're running in Kaggle
def is_kaggle():
    return os.path.exists('/kaggle/input')

# Function to check if port is open
def is_port_open(port):
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        result = s.connect_ex(('localhost', port))
        s.close()
        return result == 0
    except:
        return False

# OLLAMA SERVER
ollama_available = False
if args.host != "host.docker.internal" and args.model == "ollama":
    port_number = 11434
    if not is_port_open(port_number):
        print("Ollama not running, attempting to start...")
        try:
            if is_kaggle():
                # Installing Ollama in Kaggle environment
                print("Setting up Ollama in Kaggle environment...")
                subprocess.run(["curl", "-fsSL", "https://ollama.com/install.sh", "|", "sh"], check=True)
                # Start Ollama
                subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE)
                # Wait for Ollama to start
                for i in range(10):
                    if is_port_open(port_number):
                        ollama_available = True
                        break
                    time.sleep(1)
                    print(f"Waiting for Ollama to start... ({i+1}/10)")
            else:
                # For local environment
                subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE)
                time.sleep(5)  # Wait for Ollama to start
                ollama_available = is_port_open(port_number)
        except Exception as e:
            print(f"Failed to start Ollama: {str(e)}")
            ollama_available = False
    else:
        ollama_available = True

    if not ollama_available:
        print("⚠️ Ollama is not available. Switching to OpenAI API.")
        args.model = "openai"

# PIPELINE
pipeline = LocalRAGPipeline(host=args.host, model_type=args.model)

# UI
ui = LocalChatbotUI(
    pipeline=pipeline,
    logger=logger,
    host=args.host,
    data_dir=DATA_DIR,
    avatar_images=AVATAR_IMAGES
)

ui.build().launch(
    share=True,
    server_name="0.0.0.0",
    debug=False,
    show_api=False,
    queue=True
)