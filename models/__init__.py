"""与 Gradio 无关的模型推理接口。"""

from .inference import Prediction, Predictor, create_predictor

__all__ = ["Prediction", "Predictor", "create_predictor"]
