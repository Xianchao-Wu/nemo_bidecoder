# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.precision import NativeMixedPrecisionPlugin
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities.types import _PATH
from torch.distributed.algorithms.ddp_comm_hooks.debugging_hooks import noop_hook
from torch.nn.parallel import DistributedDataParallel

from nemo.collections.nlp.modules.common.megatron.module import Float16Module
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from nemo.core.optim import MasterOptimizerWrapper
from nemo.utils import AppState, logging

try:
    from apex.transformer import parallel_state

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False


class NLPDDPPlugin(DDPPlugin):
    """ DDP plugin for Pytorch Lightning. Needed to customize DDP for model parallel models.

    Args:
        no_ddp_communication_hook: Disable DDP communication hook when using AMP-O2
        with FP32 gradient accumulation.
    """

    accelerator = "ddp"

    def __init__(
        self,
        parallel_devices: Optional[List[torch.device]] = None,
        num_nodes: int = 1,
        cluster_environment: ClusterEnvironment = None,
        sync_batchnorm: bool = False,
        checkpoint_io: Optional[CheckpointIO] = None,
        no_ddp_communication_hook: bool = False,
        **kwargs: Union[Any, Dict[str, Any]],
    ) -> None:
        super().__init__(parallel_devices, num_nodes, cluster_environment, checkpoint_io, sync_batchnorm, **kwargs)

        if not HAVE_APEX:
            logging.warning(
                "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )
        self.no_ddp_communication_hook = no_ddp_communication_hook

    def setup_distributed(self, global_rank: int = None, world_size: int = None) -> None:
        # call PTL init ddp
        super().setup_distributed()

        # init model parallel if needed
        app_state = AppState()

        if app_state.model_parallel_size is not None:
            self.init_model_parallel(app_state.global_rank, app_state.world_size)

    def configure_ddp(self):
        """ Override LightningModule ddp if using model parallel.
            Sets find_unused_parameters to False to use activation-checkpoint-recomputation.
        """

        app_state = AppState()

        if app_state.model_parallel_size is not None:
            logging.info(f"Configuring DDP for model parallelism.")

            # With model parallelism, multiple GPUs form a large "logical GPU"
            # this means that data parallel groups span multiple GPUs
            # and are non-trivial
            device_ids = self.determine_ddp_device_ids()
            self._model = DistributedDataParallel(
                LightningDistributedModule(self.model),
                device_ids=device_ids,
                output_device=device_ids[0],
                process_group=app_state.data_parallel_group,
                find_unused_parameters=False,
                **self._ddp_kwargs,
            )

            if self.no_ddp_communication_hook:
                # When using custom gradient accumulation and allreduce, disable
                # DDP communication hook that works on the gradient bucket.
                # Instead, use the custom gradient function and communication hook,
                # which is defined in the master optimizer wrapper.
                self._model.register_comm_hook(None, noop_hook)

        else:
            super().configure_ddp()

    def init_model_parallel(self, global_rank: int, world_size: int) -> None:
        """ Initializes Megatron-LM model parallel if using model parallelism.

        Args:
            global_rank (int): the global process index.
            world_size (int): the total number of GPUs, num_nodes * num_gpus
            is_slurm_managing_tasks (bool, optional): is the cluster managed by SLURM.
        """
        app_state = AppState()

        # we initialize megatron-lm model parallel and data parallel groups
        # after initializing DDP with PTL.
        if app_state.model_parallel_size is not None:
            if torch.distributed.is_initialized():
                parallel_state.initialize_model_parallel(app_state.model_parallel_size)
                app_state.model_parallel_group = parallel_state.get_tensor_model_parallel_group()
                app_state.data_parallel_group = parallel_state.get_data_parallel_group()
                app_state.model_parallel_rank = parallel_state.get_tensor_model_parallel_rank()
                app_state.data_parallel_rank = parallel_state.get_data_parallel_rank()
                app_state.data_parallel_size = parallel_state.get_data_parallel_world_size()
                logging.info(f'mp_rank: {app_state.model_parallel_rank}')
                logging.info(f'dp_rank: {app_state.data_parallel_rank}')

    def save_checkpoint(self, checkpoint: Dict[str, Any], filepath: _PATH) -> None:
        # PTL override to accomodate model parallel checkpoints
        filepath = self._inject_model_parallel_rank(filepath)
        return super().save_checkpoint(checkpoint, filepath)

    def load_model_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        # Release strict state dict matching when using Megatron AMP-O2 to skip matching
        # half-precision module wrapper module.
        if isinstance(self.lightning_module.model, Float16Module):
            new_state_dict = {}
            for key in checkpoint['state_dict'].keys():
                new_key = key.replace('model.', 'model.module.', 1)
                new_state_dict[new_key] = checkpoint['state_dict'][key]
            checkpoint['state_dict'] = new_state_dict

        self.lightning_module.load_state_dict(checkpoint["state_dict"])

    def remove_checkpoint(self, filepath: _PATH) -> None:
        # PTL override to accomodate model parallel checkpoints
        filepath = self._inject_model_parallel_rank(filepath)
        logging.info(f'Removing checkpoint: {filepath}')
        return super().remove_checkpoint(filepath)

    def _inject_model_parallel_rank(self, filepath):
        app_state = AppState()
        # inserts mp_rank_XX for model parallel checkpoints
        if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
            # filepath needs to be updated to include mp_rank
            dirname = os.path.dirname(filepath)
            basename = os.path.basename(filepath)
            filepath = f'{dirname}/mp_rank_{app_state.model_parallel_rank:02d}/{basename}'
            return filepath
        else:
            return filepath

    @property
    def should_rank_save_checkpoint(self) -> bool:
        # PTL override that determines if checkpoints should be saved based on rank
        # for model parallel we need data_parallel_rank==0
        app_state = AppState()
        if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:
            return app_state.data_parallel_rank == 0
        else:
            return super().should_rank_save_checkpoint

    @property
    def distributed_sampler_kwargs(self):
        app_state = AppState()
        if app_state.model_parallel_size is not None:
            # When using model parallel, data parallel groups are non-trivial and they
            # correspond to the logical GPUs. This means that the GPUs that form a
            # single logical GPU all need to get the same batch of data.
            distributed_sampler_kwargs = dict(
                num_replicas=app_state.data_parallel_size, rank=app_state.data_parallel_rank
            )
            return distributed_sampler_kwargs

        else:
            return super(NLPDDPPlugin, self).distributed_sampler_kwargs


class NLPSaveRestoreConnector(SaveRestoreConnector):
    def __init__(self) -> None:
        super().__init__()
        if not HAVE_APEX:
            logging.warning(
                "Apex was not found. Please see the NeMo README for installation instructions: https://github.com/NVIDIA/NeMo#megatron-gpt."
            )

    def save_to(self, model, save_path: str):
        app_state = AppState()
        if app_state.model_parallel_size is not None and app_state.model_parallel_size > 1:

            dir_name = os.path.dirname(save_path)

            # first we save the weights for each model parallel rank
            if app_state.data_parallel_rank == 0:
                mp_model_weights = os.path.join(
                    dir_name, f'mp_rank_{app_state.model_parallel_rank:02d}_' + self.model_weights_ckpt
                )
                self._save_state_dict_to_disk(model.state_dict(), mp_model_weights)

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            # create nemo file from folder with all mp_ranks checkpoints
            if app_state.model_parallel_rank == 0 and app_state.data_parallel_rank == 0:
                with tempfile.TemporaryDirectory() as tmpdir:

                    # move weights to the tmpdir
                    for mp_rank in range(app_state.model_parallel_size):
                        os.makedirs(os.path.join(tmpdir, f'mp_rank_{mp_rank:02d}'))
                        mp_model_weights = os.path.join(dir_name, f'mp_rank_{mp_rank:02d}_' + self.model_weights_ckpt)
                        shutil.move(
                            mp_model_weights, os.path.join(tmpdir, f'mp_rank_{mp_rank:02d}', self.model_weights_ckpt)
                        )

                    # create config and artifacts in tmpdir
                    config_yaml = os.path.join(tmpdir, self.model_config_yaml)
                    model.to_config_file(path2yaml_file=config_yaml)
                    if hasattr(model, 'artifacts') and model.artifacts is not None:
                        self._handle_artifacts(model, nemo_file_folder=tmpdir)
                        self._update_artifact_paths(model, path2yaml_file=config_yaml)

                    # create tar file
                    self._make_nemo_file_from_folder(save_path, tmpdir)

        else:
            return super().save_to(model, save_path)


class GradScaler(torch.cuda.amp.GradScaler):
    """
    Gradient sclaer for model-parallel inf check. The inf in gradients are checked across tensor-parallel
    ranks in (1) executing optimizer step and (2) gradient scaler update.

    """

    def __init__(
        self,
        init_scale=2.0 ** 16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=True,
        hysteresis=1,
    ):
        super().__init__(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )
        self.optimizer_update_skipped: Optional[bool] = None
        self.hysteresis = hysteresis
        self._hysteresis_tracker = self.hysteresis

    def _maybe_opt_step(self, optimizer, optimizer_state, *args, **kwargs):
        retval = None
        found_inf = torch.cuda.FloatTensor([sum(v.item() for v in optimizer_state["found_inf_per_device"].values())])

        # Update across all model parallel instances.
        torch.distributed.all_reduce(
            found_inf, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_model_parallel_group()
        )

        if found_inf.item() == 0:
            retval = optimizer.step(*args, **kwargs)
            self.optimizer_update_skipped = False
        else:
            self.optimizer_update_skipped = True
        return retval

    def update(self, new_scale=None):
        """
        Updates to native grad scaler update function.
        1. Check inf across model-parallel ranks.
        2. Update hysteresis tracker.
        3. Apply hysteresis to grad scale update.
        """
        if not self._enabled:
            return

        _scale, _growth_tracker = self._check_scale_growth_tracker("update")

        if new_scale is not None:
            # Accept a new user-defined scale.
            if isinstance(new_scale, float):
                self._scale.fill_(new_scale)  # type: ignore[union-attr]
            else:
                reason = "new_scale should be a float or a 1-element torch.cuda.FloatTensor with requires_grad=False."
                assert isinstance(new_scale, torch.cuda.FloatTensor), reason  # type: ignore[attr-defined]
                assert new_scale.numel() == 1, reason
                assert new_scale.requires_grad is False, reason
                self._scale.copy_(new_scale)  # type: ignore[union-attr]
        else:
            # Consume shared inf/nan data collected from optimizers to update the scale.
            # If all found_inf tensors are on the same device as self._scale, this operation is asynchronous.
            found_infs = [
                found_inf.to(device=_scale.device, non_blocking=True)
                for state in self._per_optimizer_states.values()
                for found_inf in state["found_inf_per_device"].values()
            ]

            assert len(found_infs) > 0, "No inf checks were recorded prior to update."

            found_inf_combined = found_infs[0]

            # Update across all model parallel instances.
            torch.distributed.all_reduce(
                found_inf_combined, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_model_parallel_group()
            )

            if len(found_infs) > 1:
                for i in range(1, len(found_infs)):
                    found_inf = found_infs[i]
                    # Update across all model parallel instances.
                    torch.distributed.all_reduce(
                        found_inf, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_model_parallel_group()
                    )
                    found_inf_combined += found_inf

            if found_inf_combined > 0:
                self._hysteresis_tracker -= 1
                if self._hysteresis_tracker <= 0:
                    # When hysteresis becomes zero, follow the native grad scale update rule.
                    # Increase scale and reset growth tracker
                    torch._amp_update_scale_(
                        _scale,
                        _growth_tracker,
                        found_inf_combined,
                        self._growth_factor,
                        self._backoff_factor,
                        self._growth_interval,
                    )
                else:
                    # Only reset the growth tracker when hysteresis is larger than zero
                    _growth_tracker.fill_(0.0)
            else:
                # When no inf found, follow the native grad scale update rule.
                # Increment growth_tracker, update scale when growth tracker reaches the interval, and
                # reset the hysteresis tracker.
                torch._amp_update_scale_(
                    _scale,
                    _growth_tracker,
                    found_inf_combined,
                    self._growth_factor,
                    self._backoff_factor,
                    self._growth_interval,
                )
                self._hysteresis_tracker = self.hysteresis

        # To prepare for next iteration, clear the data collected from optimizers this iteration.
        self._per_optimizer_states = defaultdict(torch.cuda.amp.grad_scaler._refresh_per_optimizer_state)

    def state_dict(self):
        """
        Add hysteresis_tracker to the native functions' state_dict
        """
        return (
            {
                "scale": self.get_scale(),
                "growth_factor": self._growth_factor,
                "backoff_factor": self._backoff_factor,
                "growth_interval": self._growth_interval,
                "_growth_tracker": self._get_growth_tracker(),
                "_hysteresis_tracker": self._hysteresis_tracker,
            }
            if self._enabled
            else {}
        )

    def load_state_dict(self, state_dict):
        """
        Load hysteresis_tracker in addition to the state dict of the native function
        """
        if not self._enabled:
            return

        if len(state_dict) == 0:
            raise RuntimeError(
                "The source state dict is empty, possibly because it was saved "
                "from a disabled instance of GradScaler."
            )

        self._init_scale = state_dict["scale"]
        if self._scale is not None:
            self._scale.fill_(state_dict["scale"])
        self._growth_factor = state_dict["growth_factor"]
        self._backoff_factor = state_dict["backoff_factor"]
        self._growth_interval = state_dict["growth_interval"]
        self._init_growth_tracker = state_dict["_growth_tracker"]
        if self._growth_tracker is not None:
            self._growth_tracker.fill_(state_dict["_growth_tracker"])
        if "_hysterisis_tracker" in state_dict:
            self._hysteresis_tracker = state_dict["_hysterisis_tracker"]
        else:
            self._hysteresis_tracker = 1


class MegatronHalfPrecisionPlugin(NativeMixedPrecisionPlugin):
    """
    Plugin for Half (FP16 and BF16) precision training.
    This plugin assumes the use of the optimizer with master parameters (fp32).
    This plugin uses half-precision at all operators in the model so need of input precision
    at each layer operator.

    Args:
        precision: Whether to use ``torch.float16`` (``16``) or ``torch.bfloat16`` (``'bf16'``).
        device: The device for ``torch.autocast``.
        scaler: An optional :class:`torch.cuda.amp.GradScaler` to use.
    """

    def __init__(
        self, precision: Union[str, int], device: str, scaler: Optional[torch.cuda.amp.GradScaler] = None
    ) -> None:
        super().__init__(precision, device, scaler)

    def optimizer_step(
        self,
        model: Union["pl.LightningModule", torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        optimizer_idx: int,
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> None:
        assert isinstance(
            optimizer, MasterOptimizerWrapper
        ), "MegatronHalfPrecisionPlugin supports only the optimizer with master parameters"

        if self.scaler is None:
            assert optimizer.fp32_grad_accumulation, "BF16 uses FP32 grad accumulation"
            if optimizer.async_master_grads_allreudce:
                # Execute the last step with asynchronous master gradients all-reduce
                with optimizer.grad_sync():
                    _ = closure()
            else:
                _ = closure()
                optimizer.allreduce_main_grads()

            self._after_closure(model, optimizer, optimizer_idx)
            return optimizer.step(**kwargs)

        if isinstance(optimizer, torch.optim.LBFGS):
            raise MisconfigurationException(
                f"Native AMP and the LBFGS optimizer are not compatible (optimizer {optimizer_idx})."
            )
        assert not optimizer.fp32_grad_accumulation, "FP16 uses FP16 grad accumulation"
        closure_result = closure()

        # TODO: Add an option for merged all-reduce

        # cast fp16 grads to fp32 and copy to main grads, which are used for unscale and param update
        optimizer.copy_model_grads_to_main_grads()
        # `unscale` after the closure is executed but before the `on_before_optimizer_step` hook.
        # unscale main (fp32) gradients
        self.scaler.unscale_(optimizer)
        self._after_closure(model, optimizer, optimizer_idx)
        skipped_backward = closure_result is None
        # in manual optimization, the closure does not return a value
        if not isinstance(model, pl.LightningModule) or not model.automatic_optimization or not skipped_backward:
            # note: the scaler will skip the `optimizer.step` if nonfinite gradients are found
            self.scaler.step(optimizer, **kwargs)
            self.scaler.update()
            model.log('grad_scale', self.scaler.get_scale())

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """ No explicit precision casting. Inputs are supposed to be manually casted """
        try:
            yield
        finally:
            pass
