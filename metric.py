from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from ignite.exceptions import NotComputableError
import numpy as np
import torch
from skimage import transform


class MeanAveragePosError(Metric):
    def __init__(self, in_shape, device='cpu'):
        in_shape_array = np.float32(in_shape)
        test_pts_array = np.array([in_shape_array / 2., in_shape_array / 4.,
                                   (in_shape[0] * 3. / 4., in_shape[1] / 4.),
                                   in_shape_array * 3. / 4.,
                                   (in_shape[0] / 4., in_shape[1] / 4.)]),
        self.__test_pts = torch.tensor(test_pts_array, device=device)
        self.__average_errors = []

    @reinit__is_reduced
    def reset(self):
        self.__average_errors = []

    @reinit__is_reduced
    def update(self, output):
        esti_homo, true_homo = output
        esti_homo = torch.reshape(esti_homo, (-1, 4, 2))
        true_homo = torch.reshape(true_homo, (-1, 4, 2))
        esti_warp_pts = self.__test_pts + esti_homo
        true_warp_pts = self.__test_pts + true_homo
        point_wise_error = (torch.sum((esti_warp_pts - true_warp_pts) ** 2, dim=-1)) ** 0.5
        self.__average_errors.append(point_wise_error.mean())

    @sync_all_reduce('__average_errors')
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('AverageError must have at least one example before it can be computed.')
        return torch.tensor(self.__average_errors).mean()
