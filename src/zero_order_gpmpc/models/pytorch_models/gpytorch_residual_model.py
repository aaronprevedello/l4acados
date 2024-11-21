from typing import Optional
import torch
import gpytorch

from .pytorch_feature_selector import FeatureSelector
from .pytorch_residual_model import PyTorchResidualModel

import numpy as np
import torch


class GPyTorchResidualModel(PyTorchResidualModel):
    """Basic gpytorch based GP residual model class.

    Args:
        - gp_model: A conditioned and trained instance of a gpytorch.models.ExactGP.
        - feature_selector: Optional feature selector if certain state dimensions are
          are known to be irrelevant for the GP inference. If set to None, then no selection
          is performed.
    """

    def __init__(
        self,
        gp_model: gpytorch.models.ExactGP,
        feature_selector: Optional[FeatureSelector] = None,
    ):
        super().__init__(gp_model, feature_selector)
        self.gp_model = gp_model

    def _predictions_fun_sum(self, y):
        """Helper function for jacobian computation

        sums up the mean predictions along the first dimension
        (i.e. along the horizon).
        """
        self.evaluate(y, require_grad=True)
        return self.predictions.mean.sum(dim=0)

    def evaluate(self, y, require_grad=False):
        # NOTE(@naefjo): covar_root_decomposition=False forces linear_operator to use cholesky.
        # This is needed as otherwise our approach falls apart.
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_computations(
            covar_root_decomposition=False
        ):
            y_tensor = self.to_tensor(y)
            if require_grad:
                self.predictions = self.gp_model(self._feature_selector(y_tensor))
            else:
                with torch.no_grad():
                    self.predictions = self.gp_model(self._feature_selector(y_tensor))

        self.current_prediction = self.to_numpy(self.predictions.mean)

        # NOTE: legacy compatibility, to be removed.
        self.current_mean = self.current_prediction

        # NOTE(@naefjo): If we skipped posterior covariances, we keep the old covars in cache for hewing method.
        if gpytorch.settings.skip_posterior_variances.off():
            self.current_variance = self.to_numpy(self.predictions.variance)

        return self.current_prediction
