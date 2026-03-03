# ElasTF

Elastic Distributed Training with TensorFlow (CS 214 Big Data Systems).

ElasTF demonstrates fault-tolerant, elastic distributed deep learning on a local machine. A central **controller** monitors worker health via heartbeats, dynamically generates `TF_CONFIG` for TensorFlow's `MultiWorkerMirroredStrategy`, and a **supervisor** script automatically restarts training with fewer workers when one fails — resuming from the latest checkpoint.

## Features

- **Distributed training** using `tf.distribute.MultiWorkerMirroredStrategy` on MNIST
- **Heartbeat-based monitoring** — TCP heartbeat protocol between workers and controller with configurable timeout (default 15 s)
- **Automatic checkpointing** via `tf.keras.callbacks.ModelCheckpoint` with chief-worker write coordination
- **Elastic recovery** — when a worker dies, the supervisor kills remaining workers and restarts training with N−1 workers from the latest checkpoint
- **Dynamic `TF_CONFIG`** — the controller regenerates cluster configuration on every membership change

## Project Layout

```
ElasTF/
├── launch.sh               # Supervisor script (macOS Terminal.app)
├── requirements.txt         # Python dependencies
├── elas_tf/                 # Main Python package
│   ├── __init__.py
│   ├── controller.py        # Heartbeat monitor + TF_CONFIG writer
│   ├── worker.py            # Worker entrypoint (register, load config, train)
│   ├── training.py          # MNIST model + MultiWorkerMirroredStrategy training loop
│   ├── heartbeat.py         # TCP heartbeat client/server protocol
│   ├── checkpointing.py     # Checkpoint manager helpers
│   └── tests_functional.py  # Functional smoke tests
└── shared/                  # Runtime state (not for committing)
    ├── checkpoints/         # Model checkpoints
    └── config/              # Generated TF_CONFIG
```

## Prerequisites

- **macOS** (the supervisor script uses `open -a Terminal` to launch windows)
- **Python 3.11** (Apple Silicon / arm64)
- **TensorFlow 2.15** installed as `tensorflow-macos` (not the generic `tensorflow` package)

## Setup

1. Clone the repository and create a virtual environment:

   ```bash
   cd ElasTF
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install `tensorflow-macos` and other dependencies:

   ```bash
   pip install tensorflow-macos==2.15.0
   pip install -r requirements.txt
   ```

3. Fix SSL certificates so the MNIST dataset can be downloaded:

   ```bash
   pip install certifi
   ```

   Then add the following line to your `~/.zshrc` (or run it in your terminal before launching):

   ```bash
   export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")
   ```

   Apply it immediately with `source ~/.zshrc` or open a new terminal.

## Running

### Using `launch.sh` (recommended)

**Before running**, open `launch.sh` and update the `PROJECT_DIR` variable at the top to match your local checkout path:

```bash
PROJECT_DIR="/your/path/to/ElasTF"
```

Then launch the supervisor:

```bash
chmod +x launch.sh
./launch.sh
```

What happens:

1. The supervisor kills any leftover ElasTF processes and clears old state.
2. A **controller** window opens — it listens for heartbeats on port 6000.
3. Three **worker** windows open — each registers with the controller, waits for all workers to check in, loads the generated `TF_CONFIG`, and starts distributed training on MNIST.
4. To test elastic recovery, kill any worker process (the supervisor prints the command). The supervisor detects the failure, kills remaining workers, and restarts training with N−1 workers from the latest checkpoint.

Configuration is at the top of `launch.sh`:

| Variable        | Default | Description                         |
|-----------------|---------|-------------------------------------|
| `NUM_WORKERS`   | 3       | Initial number of workers           |
| `EPOCHS`        | 5       | Training epochs per generation      |
| `HEARTBEAT_PORT`| 6000    | Controller heartbeat listen port    |
| `STARTUP_SLEEP` | 15      | Seconds to wait for all workers to register before training |

### Running components manually

Start the controller:

```bash
source .venv/bin/activate
export CONFIG_DIR=shared/config CHECKPOINT_DIR=shared/checkpoints HEARTBEAT_PORT=6000
python3 -m elas_tf.controller
```

Start each worker in a separate terminal:

```bash
source .venv/bin/activate
export WORKER_ID=0 CONTROLLER_HOST=localhost WORKER_HOST=localhost
export HEARTBEAT_PORT=6000 CONFIG_DIR=shared/config CHECKPOINT_DIR=shared/checkpoints
export TF_PORT=30000 STARTUP_SLEEP_SECS=15 EPOCHS=5
python3 -m elas_tf.worker
```

Increment `WORKER_ID` and `TF_PORT` for each additional worker.

## Troubleshooting

### `SSL: CERTIFICATE_VERIFY_FAILED` when downloading MNIST

Python cannot verify the HTTPS certificate for `storage.googleapis.com`. This is common on macOS when using the python.org installer. Fix it by installing `certifi` and pointing Python at its CA bundle:

```bash
pip install certifi
export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")
```

Add the `export` line to `~/.zshrc` to make it permanent.

### `module 'tensorflow' has no attribute 'keras'`

This happens when the generic `tensorflow` PyPI package is installed alongside `tensorflow-macos`. The generic package is an empty namespace that shadows the real one. Fix it by removing `tensorflow` and reinstalling `tensorflow-macos`:

```bash
pip uninstall tensorflow -y
pip install --force-reinstall tensorflow-macos==2.15.0
```

### `exec: python: not found`

macOS does not provide a bare `python` command (only `python3`). The `launch.sh` script uses `python3` to avoid this. If you see this error, make sure you are using the latest version of `launch.sh`.

### `numpy.core.umath failed to import`

A broken or incompatible NumPy installation. Reinstall it:

```bash
pip install --upgrade --force-reinstall numpy
```
