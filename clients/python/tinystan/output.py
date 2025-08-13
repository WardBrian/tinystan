from typing import Dict, List, Optional, Union

import numpy as np
import stanio


class StanOutput:
    """
    A holder for the output of a Stan run.

    The ``data`` attribute contains the raw output from Stan.

    If a specific parameter is needed, it can be extracted using the
    :meth:`~StanOutput.get` method, or by using the object as a dictionary.

    Additional attributes may be available depending on the algorithm used,
    such as ``hessian`` or ``inv_metric``.
    """

    stepsize: Optional[np.ndarray]
    inv_metric: Optional[np.ndarray]
    hessian: Optional[np.ndarray]

    def __init__(self, parameters: List[str], data: np.ndarray):
        self.raw_parameters = parameters
        self._params = stanio.parse_header(",".join(parameters))
        self._data = data
        # algorithm-specific attributes
        self.hessian = None
        self.inv_metric = None
        self.stepsize = None

    @property
    def data(self) -> np.ndarray:
        """The underlying draws from the Stan model."""
        return self._data

    @property
    def parameters(self) -> List[str]:
        """The names of the parameters in the Stan model."""
        return list(self._params.keys())

    def __getitem__(self, key: str) -> np.ndarray:
        """Extract a parameter from the Stan output."""
        return self.get(key)

    def get(self, key: str) -> np.ndarray:
        """
        Extract a parameter from the Stan output.
        Synonym for ``obj[key]``.

        Parameters
        ----------
        key : str
            name of the parameter to extract

        Returns
        -------
        np.ndarray
            The parameter values. Shape depends
            on the Stan type and algorithm used.
        """
        return self._params[key].extract_reshape(self._data)

    def __repr__(self) -> str:
        return f"StanOutput(parameters={repr(self.raw_parameters)}, data={repr(self.data)})"

    def __str__(self) -> str:
        p = "\n\t".join(self.parameters)
        return f"StanOutput with parameters:\n\t{p}"

    def create_inits(
        self, *, chains: int = 4, seed: Optional[int] = None
    ) -> Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:
        """
        Create a dictionary of parameters suitable for initializing a new Stan run.

        Parameters
        ----------
        chains : int, optional
            Number of chains needed, by default 4
        seed : Optional[int], optional
            The seed to use for the random number generator.
            If not provided, a random seed will be generated.

        Returns
        -------
        Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]
            A dictionary of parameters, or a list of dictionaries if
            chains > 1.
        """
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
        return [
            {name: var.extract_reshape(data[idx]) for name, var in self._params.items()}
            for idx in idxs
        ]
