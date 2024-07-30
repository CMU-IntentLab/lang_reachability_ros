import subprocess
import time
import logging
import os
import signal

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def start_node(command, node_name):
    logging.info(f"Starting {node_name} node...")
    return subprocess.Popen(command, shell=True, preexec_fn=os.setsid)

# Function to handle shutdown
def shutdown(signum, frame):
    logging.info("Shutting down nodes...")

    # Send SIGTERM to all processes in the group
    os.killpg(os.getpgid(simulator_node.pid), signal.SIGTERM)
    os.killpg(os.getpgid(command_node.pid), signal.SIGTERM)
    os.killpg(os.getpgid(navigator_node.pid), signal.SIGTERM)
    os.killpg(os.getpgid(constraint_detector_node.pid), signal.SIGTERM)
    os.killpg(os.getpgid(simulator_node.pid), signal.SIGTERM)
    os.killpg(os.getpgid(safety_node.pid), signal.SIGTERM)

    logging.info("Nodes have been shut down.")
    exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

simulator_node = start_node('python3 scripts/simulator_node.py', 'simulator')
command_node = start_node('python3 scripts/command_node.py', 'command')
navigator_node = start_node('python3 scripts/navigation_node.py', 'navigation')
constraint_detector_node = start_node('python3 scripts/constraint_detector_node.py', 'constraint_detector')
safety_node = start_node('python3 scripts/safe_controller_node.py', 'safety')
# metrics_node = start_node('python3 scripts/metrics_recorder_node.py', 'metrics')


try:
    while True:
        time.sleep(1)  # Keep the script running
except KeyboardInterrupt:
    shutdown(None, None)  # Call the shutdown function manually if interrupted