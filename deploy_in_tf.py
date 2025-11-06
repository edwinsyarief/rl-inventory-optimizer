import argparse
import logging
import os
import threading

import grpc
import numpy as np
import onnx
import tensorflow as tf
from flask import Flask, jsonify, request
from onnx_tf.backend import prepare
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_onnx_to_tf(args):
    try:
        # Load ONNX model
        onnx_model = onnx.load(args.onnx_path)
        
        # Convert to TensorFlow with optimizations
        tf_rep = prepare(onnx_model, device='CPU')  # CPU for cost, or CUDA if available
        
        # Save as SavedModel
        os.makedirs(args.tf_model_dir, exist_ok=True)
        tf_rep.export_graph(args.tf_model_dir)
        logger.info(f"Model converted to TensorFlow SavedModel: {args.tf_model_dir}")
    except Exception as e:
        logger.error(f"Error during conversion: {str(e)}")
        raise

class TFInferenceServer:
    def __init__(self, model_dir, host='0.0.0.0', port=5000, serving_address='localhost:8500'):
        self.app = Flask(__name__)
        self.model_dir = model_dir
        self.host = host
        self.port = port
        self.channel = grpc.insecure_channel(serving_address)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            try:
                data = request.json
                state = np.array(data['state'], dtype=np.float32)  # [inventory, forecast]
                
                request_proto = predict_pb2.PredictRequest()
                request_proto.model_spec.name = 'dqn_model'
                request_proto.model_spec.signature_name = 'serving_default'
                request_proto.inputs['input'].CopyFrom(tf.make_tensor_proto(state, shape=[1, len(state)]))
                
                result = self.stub.Predict(request_proto, timeout=5.0)  # Short timeout to avoid hangs
                output = tf.make_ndarray(result.outputs['output'])
                action = np.argmax(output[0])
                
                return jsonify({'order_quantity': int(action)})
            except grpc.RpcError as e:
                logger.error(f"gRPC error: {e.details()}")
                return jsonify({'error': 'Service unavailable'}), 503
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/health', methods=['GET'])
        def health():
            try:
                # Simple health check to TF Serving
                request_proto = predict_pb2.PredictRequest()
                request_proto.model_spec.name = 'dqn_model'
                self.stub.Predict(request_proto, timeout=1.0)
                return jsonify({'status': 'healthy'})
            except:
                return jsonify({'status': 'unhealthy'}), 503
    
    def run(self):
        logger.info(f"Starting inference server at {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, threaded=True, debug=False)  # No debug for prod

def start_tf_serving(model_dir):
    logger.warning("Start TensorFlow Serving separately: docker run -p 8500:8500 -p 8501:8501 --mount type=bind,source=$(pwd)/{model_dir},target=/models/dqn_model -e MODEL_NAME=dqn_model -t tensorflow/serving")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy Model in TensorFlow")
    parser.add_argument("--onnx_path", type=str, default="dqn_inventory_agent.onnx", help="Path to ONNX model")
    parser.add_argument("--tf_model_dir", type=str, default="saved_model", help="Directory to save TF model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    parser.add_argument("--serving_address", type=str, default="localhost:8500", help="TF Serving gRPC address")
    args = parser.parse_args()
    
    if not os.path.exists(args.tf_model_dir):
        convert_onnx_to_tf(args)
    
    start_tf_serving(args.tf_model_dir)
    
    server = TFInferenceServer(args.tf_model_dir, args.host, args.port, args.serving_address)
    server.run()