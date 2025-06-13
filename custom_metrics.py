import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import roc_curve
import numpy as np
import torch

from tools.decorators import _numpy_metric_conversion
from tools.numpy_utils import check_nans
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics import Metric
from typing import Optional, Any


@_numpy_metric_conversion
def md5(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1, drop_intermediate=False)
    if check_nans(fpr, tpr):
        return np.nan
    return 1-np.interp(0.05, fpr, tpr)

@_numpy_metric_conversion
def pe(y_true, y_pred):

    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1, drop_intermediate=False)
    if check_nans(fpr, tpr):
        return np.nan
    P = 0.5*(fpr+(1-tpr))
    return min(P[P>0])

class PE(Metric):
    def __init__(
        self,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds = preds.double()
        target = torch.clip(target, min=0, max=1)
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        x = dim_zero_cat(self.preds)
        y = dim_zero_cat(self.target)
        return pe(y, x).type(torch.float32)

class MD5(Metric):
    def __init__(
        self,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        preds = preds.double()
        target = torch.clip(target, min=0, max=1)
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        x = dim_zero_cat(self.preds)
        y = dim_zero_cat(self.target)
        return md5(y, x).type(torch.float32)
