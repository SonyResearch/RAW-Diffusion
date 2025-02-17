# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
from mmengine.fileio import FileClient
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist
from mmengine.visualization import Visualizer

from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample

from raw.visualization.vis_preprocessor import VIS_PREPROCESSOR
import torch
from mmdet.structures.bbox import get_box_tensor, scale_boxes
from mmengine.structures import BaseDataElement, InstanceData
import os

@HOOKS.register_module()
class RawDetVisualizationHook(Hook):
    """Detection Visualization Hook. Used to visualize validation and testing
    process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.
    2. If ``test_out_dir`` is specified, it means that the prediction results
        need to be saved to ``test_out_dir``. In order to avoid vis_backends
        also storing data, so ``vis_backends`` needs to be excluded.
    3. ``vis_backends`` takes effect if the user does not specify ``show``
        and `test_out_dir``. You can set ``vis_backends`` to WandbVisBackend or
        TensorboardVisBackend to store the prediction result in Wandb or
        Tensorboard.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        score_thr (float): The threshold to visualize the bboxes
            and masks. Defaults to 0.3.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        test_out_dir (str, optional): directory where painted images
            will be saved in testing process.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 interval_train: int = 100,
                 interval_epoch: int = 1,
                 score_thr: float = 0.3,
                 show: bool = False,
                 wait_time: float = 0.,
                 test_out_dir: Optional[str] = None,
                 file_client_args: dict = dict(backend='disk'),
                 vis_preprocessor_args = None,
        ):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.interval_train = interval_train
        self.interval_epoch = interval_epoch
        self.score_thr = score_thr
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.draw = draw
        self.test_out_dir = test_out_dir
        self._test_index = 0

        self.vis_preprocessor = VIS_PREPROCESSOR.build(vis_preprocessor_args)

    # doesnt work because predictions are not returned
    def after_train_iter(self, runner: Runner, batch_idx: int, data_batch: dict, outputs: Optional[dict]) -> None:

        if self.draw is False or batch_idx % self.interval_train != 0:
            return
        if runner.epoch % self.interval_epoch != 0 and (runner.epoch + 1) != runner.max_epochs:
            return
        
        model = runner.model
        
        with torch.no_grad():
            data_batch = model.data_preprocessor(data_batch, True)
            pred = model._run_forward(data_batch, mode='predict')

        current_iter = batch_idx // self.interval_train

        inputs = data_batch["inputs"]
        inputs = model.data_preprocessor.normalize_inv(inputs)
        self.log_one_image(runner=runner, current_iter=current_iter, data_batch=data_batch, outputs=pred, image_name="train_image", inputs=inputs)

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[DetDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]]): A batch of data samples
                that contain annotations and predictions.
        """

        if self.draw is False or batch_idx % self.interval != 0:
            return
        if (runner.epoch - 1) % self.interval_epoch != 0 and runner.epoch != runner.max_epochs:
            return

        current_iter = batch_idx // self.interval

        model = runner.model
        
        with torch.no_grad():
            data_batch = model.data_preprocessor(data_batch, False)
        inputs = data_batch["inputs"]
        inputs = inputs[:, :4]
        inputs = model.data_preprocessor.normalize_inv(inputs)

        self.log_one_image(runner=runner, current_iter=current_iter, data_batch=data_batch, outputs=outputs, image_name="val_image", inputs=inputs)


    def log_one_image(self, runner: Runner, current_iter: int, data_batch: dict,
                       outputs: Sequence[DetDataSample], image_name, inputs):

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.epoch * 1000 + current_iter

        if self.file_client is None:
            self.file_client = FileClient(**self.file_client_args)

        # Visualize only the first data
        img_path = outputs[0].img_path

        if self.vis_preprocessor is not None:
            img = self.vis_preprocessor.transform(inputs, batch_idx=0)
        else:
            img = inputs[0].cpu().detach().numpy().transpose((1, 2, 0))
        
        # because the model internally scales the bounding boxes to the original image size, we have to rescale them here
        self.rescale_bbox_inplace(data_batch, outputs)
        data_sample = outputs[0]
        
        self._visualizer.add_datasample(
            osp.basename(img_path) if self.show else image_name,
            img,
            data_sample=data_sample,
            show=self.show,
            wait_time=self.wait_time,
            pred_score_thr=self.score_thr,
            step=total_curr_iter)


    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[DetDataSample]) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        if self.test_out_dir is not None:
            self.test_out_dir = osp.join(runner.work_dir, runner.timestamp,
                                         self.test_out_dir)
            mkdir_or_exist(self.test_out_dir)

        if self.file_client is None:
            self.file_client = FileClient(**self.file_client_args)

        self.rescale_bbox_inplace(data_batch, outputs)
        inputs = data_batch["inputs"]

        for idx, data_sample in enumerate(outputs):
            self._test_index += 1

            img_path = data_sample.img_path

            if self.vis_preprocessor is not None:
                img = self.vis_preprocessor.transform(inputs, batch_idx=idx)
            else:
                img = inputs[idx].cpu().detach().numpy().transpose((1, 2, 0))

            out_file = None
            if self.test_out_dir is not None:
                out_file = osp.splitext(osp.basename(img_path))[0] + ".png"
                out_file = osp.join(self.test_out_dir, out_file)

            self._visualizer.add_datasample(
                osp.basename(img_path) if self.show else 'test_img',
                img,
                data_sample=data_sample,
                show=self.show,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                out_file=out_file,
                step=self._test_index)

    def rescale_bbox_inplace(self, data_batch, outputs):
        # rescale bboxes because they are transformed to original image shape

        bs = len(data_batch["data_samples"])
        for i in range(bs):
            data_sample = data_batch["data_samples"][i]
            img_meta = data_sample.metainfo
            output = outputs[i]
            rescale = True

            if rescale and output.pred_instances.bboxes.size(0) > 0:
                assert img_meta.get('scale_factor') is not None
                output.pred_instances.bboxes = scale_boxes(output.pred_instances.bboxes, img_meta['scale_factor'])
