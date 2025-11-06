# RL Inventory Optimizer

A reinforcement learning (DQN) agent for optimizing inventory management in supply chains. The agent learns to decide optimal order quantities to minimize holding, shortage, and ordering costs while handling stochastic demand and random supply disruptions. This project demonstrates a full ML pipeline: prototyping in PyTorch for flexible training, exporting via ONNX for interoperability, and deploying in TensorFlow for production-scale inference.

## Why This Project?

- **Real-World Impact**: Applies RL to logistics and retail scenarios, potentially reducing operational costs by 15-20% in simulations by balancing inventory levels under uncertainty.
- **Tech Stack Showcase**:

  - PyTorch for dynamic model development and training.
  - ONNX for seamless model conversion.
  - TensorFlow Serving for high-performance, scalable deployment.
  
- **Unique Aspects**: Custom environment with realistic disruptions; optimized for cost minimization without common pitfalls like resource leaks or unstable training.

## Features

- **Custom Inventory Environment**: Built with Gymnasium; simulates demand variability and supply issues.
- **DQN Agent**: Deep Q-Network with replay buffer, epsilon-greedy exploration, soft target updates, and optimizations for stability (e.g., AdamW optimizer, gradient clipping).
- **Training Optimizations**: Mixed precision (AMP) for GPU efficiency, efficient batching to minimize compute costs.
- **Model Export & Deployment**: ONNX export for framework-agnostic models; TensorFlow SavedModel with a Flask-based REST API for inference.
- **Testing & Benchmarks**: Unit tests for environment and components; benchmarks showing cost reduction and low-latency inference.
- **Production-Ready**: Logging, error handling, health checks, and Docker support for scalable deployment.

## Requirements

- Python 3.8+
- Libraries (install via `pip install -r requirements.txt`):

  ```text
  torch==2.0.1
  gymnasium==0.29.1
  numpy==1.26.4
  onnx==1.16.0
  onnx-tf==1.10.0
  tensorflow==2.15.0
  tensorflow-serving-api==2.15.0
  flask==3.0.3
  grpcio==1.64.1
  ```

- For testing: `unittest` (built-in).
- For deployment: Docker with TensorFlow Serving.

## Setup

1. Clone the repository:

   ```shell
   git clone https://github.com/edwinsyarief/rl-inventory-optimizer.git
   cd rl-inventory-optimizer
   ```

2. Install dependencies:

   ```shell
   pip install -r requirements.txt
   ```

3. (Optional) For GPU acceleration: Ensure CUDA is installed and compatible with PyTorch/TensorFlow.

## Usage

### Training the Agent

Train the DQN agent in PyTorch:

```shell
python train_agent.py --episodes 1000 --device cuda  # Use 'cpu' if no GPU
```

- This saves checkpoints and the final model (`dqn_inventory_agent.pth`).
- Monitors average rewards and costs to ensure minimization.

### Exporting the Model

Convert the trained PyTorch model to ONNX:

```shell
python export_to_onnx.py
```

- Outputs `dqn_inventory_agent.onnx`.

### Deploying the Model

1. Convert ONNX to TensorFlow SavedModel:

   ```shell
   python deploy_in_tf.py  # Automatically converts if needed
   ```

2. Start TensorFlow Serving (using Docker for production):

   ```shell
   docker build -t rl-inventory-server .
   docker run -p 8500:8500 -p 8501:8501 rl-inventory-server
   ```

3. Run the Flask inference server:

   ```shell
   python deploy_in_tf.py
   ```

   - Server runs at `http://localhost:5000`.
   - Health check: `curl http://localhost:5000/health`
   - Predict:

     ```shell
     curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"state": [50.0, 30.0]}'
     ```

     - Response example: `{"order_quantity": 20}` (optimal order based on current inventory and forecast).

### Running Tests

Verify components with unit tests:

```shell
python test_agent.py
```

- Covers environment logic, model forward pass, and buffer operations.
- All tests should pass, confirming no cost calculation errors or leaks.

### Running Benchmarks

Measure performance:

```shell
python benchmark_agent.py
```

- Simulates short training to show cost reduction.
- Tests inference latency for scalability.

## Benchmarks

Benchmarks were run on a standard CPU (results may vary on GPU):

```text
Training Benchmark: 50 episodes on cpu
Duration: 3.31s
Time per episode: 0.07s
Avg Cost First 5: 2286.50
Avg Cost Last 5: 1534.10
Cost Reduction: 32.91%
Inference Benchmark on cpu: 1000 inferences
Duration: 0.05s
Avg time per inference: 0.05ms
```

These demonstrate efficient training (cost decreases over time) and fast inference suitable for real-time supply chain decisions.

## Dockerfile for Deployment

The included `Dockerfile` containerizes TensorFlow Serving:

```text
FROM tensorflow/serving:latest

COPY saved_model /models/dqn_model/1/

ENV MODEL_NAME=dqn_model

EXPOSE 8500 8501
```

Build and run as shown in Usage.

## Future Enhancements

- Integrate real-world datasets (e.g., from Kaggle supply chain challenges).
- Add multi-agent RL for vendor-supplier coordination.
- Support cloud deployment (e.g., AWS SageMaker or Google Cloud AI Platform).
- Implement advanced RL techniques like Double DQN or Prioritized Replay.
- CI/CD pipeline with GitHub Actions for automated testing and deployment.

## License

MIT License. Feel free to use, modify, and distribute.

Star the repo if you find it useful! Open to contributions or feedback. For questions, reach out via GitHub Issues.
