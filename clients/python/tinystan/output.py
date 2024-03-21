from typing import Dict, List, Union

import numpy as np
import stanio


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

    def __repr__(self) -> str:
        return f"StanOutput(parameters={repr(self.raw_parameters)}, data={repr(self.data)})"

    def __str__(self) -> str:
        p = "\n\t".join(self.parameters)
        return f"StanOutput with parameters:\n\t{p}"

    # experimental, copied from cmdstanpy Pathfinder draft
    def create_inits(
        self, *, chains=4, seed=None
    ) -> Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:
        if self._data.ndim == 1:
            return {
                name: var.extract_reshape(self._data)
                for name, var in self._params.items()
            }

        data = self._data.reshape((-1, self._data.shape[-1]))
        rng = np.random.default_rng(seed)
        idxs = rng.choice(data.shape[0], size=chains, replace=False)
        if chains == 1:
            draw = data[idxs[0]]
            return {
                name: var.extract_reshape(draw) for name, var in self._params.items()
            }
        else:
            return [
                {
                    name: var.extract_reshape(data[idx])
                    for name, var in self._params.items()
                }
                for idx in idxs
            ]
