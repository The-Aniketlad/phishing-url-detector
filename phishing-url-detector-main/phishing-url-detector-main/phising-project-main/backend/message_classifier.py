import os

import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer


ID_TO_LABEL = {0: "S", 1: "T", 2: "P"}
_tokenizer = None
_model = None


def _resolve_model_dir(base_dir: str) -> str:
    configured = os.getenv("MESSAGE_MODEL_DIR")
    if configured:
        return configured
    return os.path.abspath(
        os.path.join(base_dir, "..", "..", "..", "..", "model", "saved_model")
    )


def _load_model():
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    model_dir = _resolve_model_dir(os.path.dirname(__file__))
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Message model directory not found: {model_dir}. "
            "Set MESSAGE_MODEL_DIR or copy trained model files."
        )

    _tokenizer = BertTokenizer.from_pretrained(model_dir)
    _model = BertForSequenceClassification.from_pretrained(model_dir)
    _model.eval()
    return _tokenizer, _model


def predict_message(text: str):
    tokenizer, model = _load_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

    predicted_id = outputs.logits.argmax(dim=1).item()
    label = ID_TO_LABEL[predicted_id]
    confidence = probs.max().item()
    return label, confidence
