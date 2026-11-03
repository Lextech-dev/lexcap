from abc import ABC, abstractmethod
from itertools import groupby
from typing import List, Optional, Tuple
import numpy as np


class CharsetAdapter:
    def __init__(self, target_charset) -> None:
        super().__init__()
        self.charset = target_charset
        self.lowercase_only = target_charset == target_charset.lower()
        self.uppercase_only = target_charset == target_charset.upper()

    def __call__(self, label):
        if self.lowercase_only:
            label = label.lower()
        elif self.uppercase_only:
            label = label.upper()
        return label


class BaseTokenizer(ABC):
    def __init__(self, charset: str, specials_first: tuple = (), specials_last: tuple = ()) -> None:
        self._itos = specials_first + tuple(charset + "[UNK]") + specials_last
        self._stoi = {s: i for i, s in enumerate(self._itos)}

    def __len__(self):
        return len(self._itos)

    def _tok2ids(self, tokens: str) -> List[int]:
        return [self._stoi[s] for s in tokens]

    def _ids2tok(self, token_ids: List[int], join: bool = True) -> str:
        tokens = [self._itos[i] for i in token_ids]
        return "".join(tokens) if join else tokens

    @abstractmethod
    def encode(self, labels: List[str], device=None):
        raise NotImplementedError

    @abstractmethod
    def _filter(self, probs: np.ndarray, ids: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        raise NotImplementedError

    def decode(self, token_dists: np.ndarray, raw: bool = False) -> Tuple[List[str], List[np.ndarray]]:
        batch_tokens = []
        batch_probs  = []
        for dist in token_dists:
            # numpy argmax replaces torch dist.max(-1) — no dispatcher, no autograd
            ids   = np.argmax(dist, axis=-1)
            probs = dist[np.arange(len(ids)), ids]   # gather max probs
            if not raw:
                probs, ids = self._filter(probs, ids)
            tokens = self._ids2tok(ids, not raw)
            batch_tokens.append(tokens)
            batch_probs.append(probs)
        return batch_tokens, batch_probs


class Tokenizer(BaseTokenizer):
    BOS = "[B]"
    EOS = "[E]"
    PAD = "[P]"

    def __init__(self, charset: str) -> None:
        specials_first = (self.EOS,)
        specials_last  = (self.BOS, self.PAD)
        super().__init__(charset, specials_first, specials_last)
        self.eos_id, self.bos_id, self.pad_id = [
            self._stoi[s] for s in specials_first + specials_last
        ]

    def encode(self, labels: List[str], device=None):
        # encode() is not used during inference — kept for completeness
        raise NotImplementedError("encode() not needed for inference")

    def _filter(self, probs: np.ndarray, ids: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        ids_list = ids.tolist()
        try:
            eos_idx = ids_list.index(self.eos_id)
        except ValueError:
            eos_idx = len(ids_list)
        ids_list = ids_list[:eos_idx]
        probs    = probs[:eos_idx + 1]
        return probs, ids_list


class CTCTokenizer(BaseTokenizer):
    BLANK = "[B]"

    def __init__(self, charset: str) -> None:
        super().__init__(charset, specials_first=(self.BLANK,))
        self.blank_id = self._stoi[self.BLANK]

    def encode(self, labels: List[str], device=None):
        raise NotImplementedError("encode() not needed for inference")

    def _filter(self, probs: np.ndarray, ids: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        ids_list = list(zip(*groupby(ids.tolist())))[0]
        ids_list = [x for x in ids_list if x != self.blank_id]
        return probs, ids_list
