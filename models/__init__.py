"""与 Gradio 无关的模型推理接口。"""

from .inference import MODEL_NAMES, Prediction, Predictor, create_predictor

__all__ = ["MODEL_NAMES", "Prediction", "Predictor", "create_predictor"]
