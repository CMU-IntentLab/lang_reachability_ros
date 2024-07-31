import json
import subprocess
import time
import logging
import os
import signal
import threading
import argparse
from pathlib import Path
from queue import Queue, Empty

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
dir_path = str(Path(__file__).parent.parent)
log_dir = os.path.join(dir_path, 'logs')

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup logger"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_path = os.path.join(log_dir, log_file)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

def enqueue_output(stream, queue):
    """Helper function to enqueue output from a stream"""
    for line in iter(stream.readline, b''):
        queue.put(line)
    stream.close()

def log_output(queue, logger):
    """Function to log output from a queue"""
    while True:
        try:
            line = queue.get_nowait()
        except Empty:
            continue
        else:
            logger.info(line.decode().strip())

def start_node(command):
    return subprocess.Popen(command, shell=True, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env={**os.environ, 'PYTHONUNBUFFERED': '1'})

def shutdown(node_list):
    logging.info("Shutting down nodes...")

    for (node, logger, queue) in node_list:
        os.killpg(os.getpgid(node.pid), signal.SIGTERM)

    logging.info("Nodes have been shut down.")
    exit(0)

def start_modules(module_name_list, exp_config_path):
    node_list = []
    module_to_node = {"mppi": "navigation_node",
                      "vlm": "constraint_detector_node",
                      "reachability": "safe_controller_node",
                      "simulator_node": "simulator_node",
                      "command_node": "command_node",
                      "metrics_recorder_node": "metrics_recorder_node"}

    for module in module_name_list:
        if module == "rtabmap":
            continue
        node_name = module_to_node[module]
        logger = setup_logger(node_name, f'{node_name}.log')
        logger.info(f"{node_name} logger setup complete")
        node = start_node(f'python3 scripts/{node_name}.py --exp_path {exp_config_path}')
        logger.info(f"{node_name} started with PID {node.pid}")
        queue = Queue()
        node_list.append((node, logger, queue))
        threading.Thread(target=enqueue_output, args=(node.stdout, queue), daemon=True).start()
        threading.Thread(target=enqueue_output, args=(node.stderr, queue), daemon=True).start()
        threading.Thread(target=log_output, args=(queue, logger), daemon=True).start()
    return node_list

if __name__ == '__main__':
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    exp_name = "mppi_vlm"
    exp_configs_path = os.path.join(dir_path, 'config', 'exps', f"{exp_name}.json")
    with open(exp_configs_path, 'r') as f:
        exp_configs = json.load(f)

    module_name_list = exp_configs['exp_name'].split('_')
    module_name_list = ["simulator_node", "command_node", "metrics_recorder_node"] + module_name_list
    node_list = start_modules(module_name_list, exp_configs_path)

    signal.signal(signal.SIGINT, lambda signum, frame: shutdown(node_list))
    signal.signal(signal.SIGTERM, lambda signum, frame: shutdown(node_list))

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown(node_list)
