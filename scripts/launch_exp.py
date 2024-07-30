import subprocess
import time
import logging
import os
import signal
import threading
from pathlib import Path
from queue import Queue, Empty

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
dir_path = str(Path(__file__).parent.parent)
log_dir = os.path.join(dir_path, 'logs')

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup logger"""
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

# Function to handle shutdown
def shutdown(signum, frame):
    logging.info("Shutting down nodes...")

    # Send SIGTERM to all processes in the group
    os.killpg(os.getpgid(test_node.pid), signal.SIGTERM)
    os.killpg(os.getpgid(command_node.pid), signal.SIGTERM)
    os.killpg(os.getpgid(simulator_node.pid), signal.SIGTERM)
    os.killpg(os.getpgid(navigation_node.pid), signal.SIGTERM)
    os.killpg(os.getpgid(constraint_detector_node.pid), signal.SIGTERM)
    os.killpg(os.getpgid(safe_controller_node.pid), signal.SIGTERM)

    logging.info("Nodes have been shut down.")
    exit(0)

if __name__ == '__main__':
    # Ensure the logs directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Setup loggers for each script
    test_logger = setup_logger('test', 'test_node.log')
    simulator_logger = setup_logger('simulator', 'simulator_node.log')
    command_logger = setup_logger('command', 'command_node.log')
    navigation_logger = setup_logger('navigation', 'navigation_node.log')
    constraint_detector_logger = setup_logger('constraint_detector_node', 'constraint_detector_node.log')
    safe_controller_logger = setup_logger('safe_controller_node', 'safe_controller_node.log')
    metrics_logger = setup_logger('metrics_node', 'metrics_node.log')

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    test_node = start_node('python3 scripts/test.py')
    command_node = start_node('python3 scripts/command_node.py')
    simulator_node = start_node('python3 scripts/simulator_node.py')
    navigation_node = start_node('python3 scripts/navigation_node.py')
    constraint_detector_node = start_node('python3 scripts/constraint_detector_node.py')
    safe_controller_node = start_node('python3 scripts/safe_controller_node.py')
    # metrics_node = start_node('python3 scripts/metrics_recorder_node.py')

    # Queues to hold the output of each node
    test_queue = Queue()
    command_queue = Queue()
    simulator_queue = Queue()
    navigation_queue = Queue()
    constraint_detector_queue = Queue()
    safe_controller_queue = Queue()
    # metrics_queue = Queue()

    # Start threads to enqueue output from each subprocess
    threading.Thread(target=enqueue_output, args=(test_node.stdout, test_queue), daemon=True).start()
    threading.Thread(target=enqueue_output, args=(test_node.stderr, test_queue), daemon=True).start()
    threading.Thread(target=enqueue_output, args=(command_node.stdout, command_queue), daemon=True).start()
    threading.Thread(target=enqueue_output, args=(command_node.stderr, command_queue), daemon=True).start()
    threading.Thread(target=enqueue_output, args=(simulator_node.stdout, simulator_queue), daemon=True).start()
    threading.Thread(target=enqueue_output, args=(simulator_node.stderr, simulator_queue), daemon=True).start()
    threading.Thread(target=enqueue_output, args=(navigation_node.stdout, navigation_queue), daemon=True).start()
    threading.Thread(target=enqueue_output, args=(navigation_node.stderr, navigation_queue), daemon=True).start()
    threading.Thread(target=enqueue_output, args=(constraint_detector_node.stdout, constraint_detector_queue), daemon=True).start()
    threading.Thread(target=enqueue_output, args=(constraint_detector_node.stderr, constraint_detector_queue), daemon=True).start()
    threading.Thread(target=enqueue_output, args=(safe_controller_node.stdout, safe_controller_queue), daemon=True).start()
    threading.Thread(target=enqueue_output, args=(safe_controller_node.stderr, safe_controller_queue), daemon=True).start()
    # threading.Thread(target=enqueue_output, args=(metrics_node.stdout, metrics_queue), daemon=True).start()
    # threading.Thread(target=enqueue_output, args=(metrics_node.stderr, metrics_queue), daemon=True).start()

    # Start threads to log output from each queue
    threading.Thread(target=log_output, args=(test_queue, test_logger), daemon=True).start()
    threading.Thread(target=log_output, args=(command_queue, command_logger), daemon=True).start()
    threading.Thread(target=log_output, args=(simulator_queue, simulator_logger), daemon=True).start()
    threading.Thread(target=log_output, args=(navigation_queue, navigation_logger), daemon=True).start()
    threading.Thread(target=log_output, args=(constraint_detector_queue, constraint_detector_logger), daemon=True).start()
    threading.Thread(target=log_output, args=(safe_controller_queue, safe_controller_logger), daemon=True).start()
    # threading.Thread(target=log_output, args=(metrics_queue, metrics_logger), daemon=True).start()

    try:
        while True:
            time.sleep(1)  # Keep the script running
    except KeyboardInterrupt:
        shutdown(None, None)  # Call the shutdown function manually if interrupted
