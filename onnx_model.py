from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


FEATURE_DIM = 32
HIDDEN_DIM = 64
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "tiny_text_mlp.onnx"


def build_model(model_path: Path) -> Path:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260909)
    weights_1 = rng.normal(
        loc=0.0,
        scale=0.05,
        size=(FEATURE_DIM, HIDDEN_DIM),
    ).astype(np.float32)
    bias_1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
    weights_2 = rng.normal(
        loc=0.0,
        scale=0.05,
        size=(HIDDEN_DIM, 1),
    ).astype(np.float32)
    bias_2 = np.zeros(1, dtype=np.float32)

    # Make two human-readable features influence the score predictably.
    weights_1[26, 0] = 1.0
    weights_1[27, 1] = 1.0
    weights_2[0, 0] = 0.8
    weights_2[1, 0] = -0.8

    input_info = helper.make_tensor_value_info(
        "features",
        TensorProto.FLOAT,
        [None, FEATURE_DIM],
    )
    output_info = helper.make_tensor_value_info(
        "score",
        TensorProto.FLOAT,
        [None, 1],
    )

    nodes = [
        helper.make_node("MatMul", ["features", "weights_1"], ["hidden_linear"]),
        helper.make_node("Add", ["hidden_linear", "bias_1"], ["hidden_bias"]),
        helper.make_node("Relu", ["hidden_bias"], ["hidden"]),
        helper.make_node("MatMul", ["hidden", "weights_2"], ["output_linear"]),
        helper.make_node("Add", ["output_linear", "bias_2"], ["logits"]),
        helper.make_node("Sigmoid", ["logits"], ["score"]),
    ]

    graph = helper.make_graph(
        nodes,
        "tiny_text_mlp",
        [input_info],
        [output_info],
        initializer=[
            numpy_helper.from_array(weights_1, name="weights_1"),
            numpy_helper.from_array(bias_1, name="bias_1"),
            numpy_helper.from_array(weights_2, name="weights_2"),
            numpy_helper.from_array(bias_2, name="bias_2"),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="async-llm-inference-gateway",
        opset_imports=[helper.make_opsetid("", 17)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, model_path)
    return model_path


def ensure_model(model_path: Path | None = None) -> Path:
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not path.exists():
        build_model(path)
    return path


if __name__ == "__main__":
    print(ensure_model())
