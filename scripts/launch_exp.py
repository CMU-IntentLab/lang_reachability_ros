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

def enqueue_output(stream, queue, stream_type):
    """Helper function to enqueue output from a stream"""
    for line in iter(stream.readline, b''):
        queue.put((line, stream_type))
    stream.close()

def log_output(queue, logger):
    """Function to log output from a queue"""
    while True:
        try:
            line, stream_type = queue.get_nowait()
        except Empty:
            continue
        else:
            if stream_type == 'stdout':
                logger.info(line.decode().strip())
            elif stream_type == 'stderr':
                logger.error(line.decode().strip())


def start_node(command):
    return subprocess.Popen(command, shell=True, preexec_fn=os.setsid, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env={**os.environ, 'PYTHONUNBUFFERED': '1'})

def shutdown(node_list):
    logging.info("Shutting down nodes...")

    for (node, logger, queue) in node_list:
        logger.info(f"Shutting down node with PID: {node.pid}")
        logger.info("=" * 50)  # Add a divide line to the logs
        os.killpg(os.getpgid(node.pid), signal.SIGTERM)

    logging.info("Nodes have been shut down.")
    exit(0)

def start_modules(module_name_list, exp_config_path, topics_names_path):
    node_list = []
    module_to_node = {"mppi": "navigation_node",
                      "vlm": "constraint_detector_node",
                      "reachability": "safe_controller_node",
                      "reachability_solver": "brt_solver_node",
                      "simulator": "simulator_node",
                      "command_node": "command_node",
                      "metrics_recorder_node": "metrics_recorder_node"}

    for module in module_name_list:
        if module == "rtabmap":
            continue
        node_name = module_to_node[module]
        logger = setup_logger(node_name, f'{node_name}.log')
        logger.info(f"{node_name} logger setup complete")
        node = start_node(f'python3 scripts/{node_name}.py --exp_path {exp_config_path} --topics_path {topics_names_path}')
        logger.info(f"{node_name} started with PID {node.pid}")
        queue = Queue()
        node_list.append((node, logger, queue))
        threading.Thread(target=enqueue_output, args=(node.stdout, queue, 'stdout'), daemon=True).start()
        threading.Thread(target=enqueue_output, args=(node.stderr, queue, 'stderr'), daemon=True).start()
        threading.Thread(target=log_output, args=(queue, logger), daemon=True).start()
    return node_list

if __name__ == '__main__':
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    parser = argparse.ArgumentParser(description="Command Node")
    parser.add_argument('--exp', type=str, default=None, help='experiment description')
    args = parser.parse_args()

    exp_name = args.exp
    exp_configs_path = os.path.join(dir_path, 'config', 'hardware_exps', exp_name, f"{exp_name}.json")
    with open(exp_configs_path, 'r') as f:
        exp_configs = json.load(f)

    topics_names_path = os.path.join(dir_path, 'config', 'hardware_exps', exp_name, "topics_names.json")

    module_name_list = exp_configs['exp_name'].split('_')
    module_name_list = ["command_node", "metrics_recorder_node"] + module_name_list
    if "reachability" in module_name_list:
        module_name_list += ["reachability_solver"]

    platform = exp_configs["platform"]  # either simulator or hardware
    if platform == 'simulator':
        module_name_list = module_name_list + [platform]

    node_list = start_modules(module_name_list, exp_configs_path, topics_names_path)

    signal.signal(signal.SIGINT, lambda signum, frame: shutdown(node_list))
    signal.signal(signal.SIGTERM, lambda signum, frame: shutdown(node_list))

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown(node_list)
