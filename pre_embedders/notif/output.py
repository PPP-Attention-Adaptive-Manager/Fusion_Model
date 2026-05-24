import numpy as np
from features import extract_features, compute_npi
from model import load_model, get_embedding, normalize_features

# Replace these values with the scaler parameters printed
# by Cell 8 of the Colab notebook after training
SCALER_PARAMS = {
    'min_': [0.0, 0.0, 0.0, 0.0, 0.0], 
    'max_': [3.0, 2.509182763787976e-07, 1.0, 0.021615064589633373, 1.0]
}

def compute_state(npi):
    if npi < 0.20:
        return "Flow"
    elif npi < 0.40:
        return "Neutral"
    elif npi < 0.60:
        return "Bored"
    elif npi < 0.80:
        return "Distracted"
    else:
        return "Overloaded"


def get_output(session):
    raw_features = extract_features()
    npi = compute_npi(
        float(raw_features[0]),
        float(raw_features[1]),
        float(raw_features[2]),
        float(raw_features[3]),
        float(raw_features[4]),
    )
    state = compute_state(npi)
    normalized_features = normalize_features(raw_features, SCALER_PARAMS)
    embedding = get_embedding(session, normalized_features)

    fusion_output = {
        "embedding": embedding,
        "npi": npi,
        "burstiness": float(raw_features[1]),
        "disruption_score": float(raw_features[3]),
    }
    full_output = {
        **fusion_output,
        "state": state,
        "metadata": {
            "module": "notifications",
            "embedding_dim": 16,
        },
    }
    return full_output


def get_fusion_output(session):
    output = get_output(session)
    return {
        "embedding": output["embedding"],
        "npi": output["npi"],
        "burstiness": output["burstiness"],
        "disruption_score": output["disruption_score"],
    }


if __name__ == "__main__":
    import time

    try:
        session = load_model()
        while True:
            output = get_output(session)
            print(f"module:           {output['metadata']['module']}")
            print(f"state:            {output['state']}")
            print(f"npi:              {output['npi']:.4f}")
            print(f"burstiness:       {output['burstiness']:.4f}")
            print(f"disruption_score: {output['disruption_score']:.4f}")
            print(f"embedding:        shape {output['embedding'].shape}")
            print(f"metadata:         {output['metadata']}")
            print()
            time.sleep(5)
    except FileNotFoundError as error:
        print(error)
    except KeyboardInterrupt:
        print("\nStopped.")
