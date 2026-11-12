"""
Run this script to create the folder structure for this project.

Author: Cemal Yilmaz - 2026
"""

import os

repo_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

folders_to_create = ["evaluation_data", "models", "models_meta", "models_replay_buffers", "monitor_logs", "schedules", "tensorboard"]

for folder in folders_to_create:
    folder_path = os.path.join(repo_parent_dir, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)