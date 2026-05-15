import numpy as np
import onnxruntime as ort
import os

MODEL_PATH = "models/notif_mlp.onnx"
EMBEDDING_DIM = 16


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "ONNX model not found at models/notif_mlp.onnx.\n"
            "   Train the model on Colab first and place the\n"
            "   exported file in the models/ folder."
        )

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(MODEL_PATH, sess_options=session_options)
    return session


def get_embedding(session, features):
    features_batch = features.reshape(1, 5)
    outputs = session.run(None, {"features": features_batch})
    embedding = outputs[0]
    return np.asarray(embedding, dtype=np.float32).reshape(EMBEDDING_DIM)


def normalize_features(features, scaler_params):
    if scaler_params is None:
        return features

    min_ = np.asarray(scaler_params["min_"], dtype=np.float32)
    max_ = np.asarray(scaler_params["max_"], dtype=np.float32)
    normalized = (features - min_) / (max_ - min_)
    normalized = np.clip(normalized, 0.0, 1.0)
    return np.asarray(normalized, dtype=np.float32)


if __name__ == "__main__":
    import time

    try:
        session = load_model()
        dummy_input = np.zeros(5, dtype=np.float32)
        start_time = time.time()
        embedding = get_embedding(session, dummy_input)
        end_time = time.time()

        print("Model loaded successfully")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Inference time: {(end_time - start_time) * 1000:.2f}ms")
    except FileNotFoundError as error:
        print(error)
