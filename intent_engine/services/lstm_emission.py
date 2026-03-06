from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass(frozen=True)
class LSTMEmission:
    model: object
    max_len: int
    vocab_size: int

    def next_token_probability(self, history_tokens: List[int], token: int) -> float:
        seq = list(history_tokens) + [int(token)]
        if len(seq) < 2:
            return 1.0

        seq = seq[-int(self.max_len) :]

        tf = self._require_tensorflow()
        pad_sequences = tf.keras.preprocessing.sequence.pad_sequences

        padded = pad_sequences([seq], maxlen=int(self.max_len), padding="pre", truncating="pre", value=0)
        x = padded[:, :-1]
        y = padded[:, 1:]
        pred = self.model.predict(x, verbose=0)[0]
        y_row = y[0]

        probs_selected = np.take_along_axis(pred, y_row[..., None], axis=-1).squeeze(-1)
        probs_selected = np.clip(probs_selected.astype(np.float64), 1e-12, 1.0)

        last_idx = None
        for i in range(len(y_row) - 1, -1, -1):
            if int(y_row[i]) != 0:
                last_idx = i
                break
        if last_idx is None:
            return 1.0

        return float(probs_selected[last_idx])

    @staticmethod
    def _require_tensorflow():
        import importlib.util

        if importlib.util.find_spec("tensorflow") is None:
            raise RuntimeError("TensorFlow is not available in this environment.")
        import tensorflow as tf  # type: ignore

        return tf


def load_lstm(path: Path, max_len: int) -> LSTMEmission:
    tf = LSTMEmission._require_tensorflow()
    model = tf.keras.models.load_model(str(path))

    vocab_size = int(model.output_shape[-1])
    return LSTMEmission(model=model, max_len=int(max_len), vocab_size=vocab_size)

