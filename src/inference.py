from typing import Any
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

from settings.constants import ID2LABEL, PATH_MODEL, PATH_TOKENIZER


class Inference():
    def __init__(self):
        self.onnx_model_path = PATH_MODEL 
        self.tokenizer = AutoTokenizer.from_pretrained(PATH_TOKENIZER)
        pass

    def __call__(self, input_text) -> Any:
        inputs = self.tokenizer(input_text, return_tensors="np")

        session = ort.InferenceSession(self.onnx_model_path)

        ort_inputs = {session.get_inputs()[0].name: inputs["input_ids"]}

        ort_outputs = session.run(None, ort_inputs)

        logits = ort_outputs[0]
        predictions = np.argmax(logits, axis=-1)

        label_predictions = np.vectorize(ID2LABEL.get)(predictions)

        return label_predictions
