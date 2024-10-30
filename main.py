# main.py
import yaml
import logging
import os
from model_loader import load_classes, load_model
from video_processor import process_video

# Create logs directory if it doesn't exist
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Initialize logging
logging.basicConfig(filename=os.path.join(log_dir, 'app.log'), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration
with open('config.yaml') as f:
    config = yaml.safe_load(f)

print(config)  # Print to verify the structure

# Load model and classes
classes = load_classes(config['paths']['classes_file'])
net = load_model(config['paths']['proto_path'], config['paths']['model_path'])

# Process the video
process_video(config['paths']['video_path'], net, classes, config['settings'])
