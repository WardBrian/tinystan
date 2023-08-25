from typing import List
import stanio
import numpy as np


class StanOutput:
    def __init__(self, parameters: List[str], data: np.ndarray):
        self.raw_parameters = parameters
        self._params = stanio.parse_header(",".join(parameters))
        self._data = data

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def parameters(self) -> List[str]:
        return list(self._params.keys())

    def __getitem__(self, key: str) -> np.ndarray:
        return self._params[key].extract_reshape(self._data)
