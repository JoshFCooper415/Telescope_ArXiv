
import subprocess
import json
import random
from datetime import datetime
from collections import defaultdict
import os
import pathlib
import sys
# First, let's list the bucket contents to find the correct path
print("Checking metadata directory...")
test_cmd = "gsutil ls gs://arxiv-dataset/metadata-v5/"
test_process = subprocess.run(test_cmd, shell=True, capture_output=True, text=True)
print("Metadata directory contents:")
print(test_process.stdout)
print("Checking file list...")
test_cmd = "gsutil cat gs://arxiv-dataset/arxiv-dataset_list-of-files.txt.gz | gunzip -"
test_process = subprocess.run(test_cmd, shell=True, capture_output=True, text=True)
print("First few lines of file list:")
print("\n".join(test_process.stdout.splitlines()[:10]))