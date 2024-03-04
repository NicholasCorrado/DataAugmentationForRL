from typing import List, Dict, Any

import numpy as np

from src.dafs.base_daf import BaseDAF


class TranslateAgent(BaseDAF):
    """
    Base data augmentation function (DAF) class.
    """
    def __init__(self, env=None, **kwargs):
        super().__init__(env, **kwargs)

    def _augment(
            self,
            obs: np.ndarray,
            next_obs: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            terminated: np.ndarray,
            truncated: np.ndarray,
            infos: List[Dict[str, Any]],
            **kwargs,
    ):
        raise NotImplementedError('DAF not implemented.')



