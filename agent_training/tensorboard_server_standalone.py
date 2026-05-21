"""
Standalone script to start a TensorBoard server for accessing training logs while not training.

Author: Cemal Yilmaz - 2026
"""

import os
import sys
import time

# Add parent directory to path for imports (must be before local imports)
_drl_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _drl_repo_dir not in sys.path:
    sys.path.insert(0, _drl_repo_dir)

from agent_training.trainer import repo_dir, log_path, RED_START, GREEN_START, YELLOW_START, COLOR_END
import subprocess

def start_tensorboard():
    """Start TensorBoard server in background, access with http://localhost:6006"""
    print("|")
    print(f"|---{YELLOW_START}Looking for TensorBoard logs in: {log_path}{COLOR_END}")

    # Check if the log directory exists
    if not os.path.exists(log_path):
        print(f"|-----{RED_START}Log directory does not exist: {log_path}{COLOR_END}")
        print(f"|-----{RED_START}Available directories:{COLOR_END}")
        for item in os.listdir(repo_dir):
            item_path = os.path.join(repo_dir, item)
            if os.path.isdir(item_path) and "tensorboard" in item.lower():
                print(f"|-------{RED_START}{item}{COLOR_END}")
    else:
        print(f"|-----{GREEN_START}Log directory found: {log_path}{COLOR_END}")

    try:
        print("|")
        print(f"|---{YELLOW_START}Starting TensorBoard server...{COLOR_END}")
        
        # Start TensorBoard process in background
        process = subprocess.Popen([
            "tensorboard",
            f"--logdir={log_path}",
            "--port=6006",
            "--host=localhost"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("|-----Access TensorBoard at: http://localhost:6006")
        print("|-----Press Ctrl+C to stop the server")
        
        # Wait a moment for TensorBoard to start
        time.sleep(3)

        return process
    
    except FileNotFoundError:
        print(f"|-----{RED_START}TensorBoard not found.{COLOR_END}")


def stop_tensorboard(process):
    """Stop TensorBoard server"""
    if process is not None:
        try:
            print(f"|---{YELLOW_START}Stopping TensorBoard server...{COLOR_END}")
            process.terminate()
            process.wait(timeout=5)
            print(f"|-----{GREEN_START}TensorBoard server stopped{COLOR_END}")
        except subprocess.TimeoutExpired:
            print(f"|-----{RED_START}Force killing TensorBoard server...{COLOR_END}")
            process.kill()
            process.wait()
            print(f"|-----{RED_START}TensorBoard server force stopped{COLOR_END}")
        except Exception as e:
            print(f"|-----{RED_START}Error stopping TensorBoard: {e}{COLOR_END}")


# Monitor training progress in TensorBoard
tensorboard_process = start_tensorboard()

# Stop TensorBoard server on ctrl+C
try:
    print("|")
    print(f"|---{YELLOW_START}Press Ctrl+C to stop the TensorBoard server.{COLOR_END}")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    stop_tensorboard(tensorboard_process)