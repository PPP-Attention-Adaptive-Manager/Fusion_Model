import traceback

import torch

from predictive_models.notif import ActiveModel


def main() -> None:
    try:
        model = ActiveModel(input_flat_dim=512)
        x = torch.randn(2, 512, dtype=torch.float32)

        out = model.forward(x)
        assert out.shape == (2, 12)

        model.reset_microstate()
        out = model.forward(x)
        assert out.shape == (2, 12)

        print("All checks passed")
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
