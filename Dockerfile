# Dockerfile
FROM tensorflow/serving:latest

# Copy the SavedModel (generated locally)
COPY saved_model /models/dqn_model/1/

# Set model name
ENV MODEL_NAME=dqn_model

# Expose ports for gRPC and REST
EXPOSE 8500 8501