# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from abc import ABC
from collections import defaultdict
from enum import Enum
from typing import Dict

import onnx
import torch

from nemo.core.classes import typecheck
from nemo.core.neural_types import AxisKind, NeuralType
from nemo.utils import logging
from nemo.utils.export_utils import replace_for_export

__all__ = ['ExportFormat', 'Exportable']


class ExportFormat(Enum):
    """Which format to use when exporting a Neural Module for deployment"""

    ONNX = (1,)
    TORCHSCRIPT = (2,)


_EXT_DICT = {
    ".pt": ExportFormat.TORCHSCRIPT,
    ".ts": ExportFormat.TORCHSCRIPT,
    ".onnx": ExportFormat.ONNX,
}


def get_input_names(self):
    if not (hasattr(self, 'input_types')):
        raise NotImplementedError('For export to work you must define input_types')
    input_names = list(self.input_types.keys())
    return input_names


def get_output_names(self):
    if not (hasattr(self, 'output_types')):
        raise NotImplementedError('For export to work you must define output_types')
    output_names = list(self.output_types.keys())
    return output_names


def get_input_dynamic_axes(self, input_names):
    dynamic_axes = defaultdict(list)
    for name in input_names:
        if name in self.input_types:
            dynamic_axes = {
                **dynamic_axes,
                **Exportable._extract_dynamic_axes(name, self.input_types[name]),
            }
    return dynamic_axes


def get_output_dynamic_axes(self, output_names):
    dynamic_axes = defaultdict(list)
    for name in output_names:
        if name in self.output_types:
            dynamic_axes = {
                **dynamic_axes,
                **Exportable._extract_dynamic_axes(name, self.output_types[name]),
            }
    return dynamic_axes


def to_onnxrt_input(input_names, input_list, input_dict):
    odict = {}
    for k, v in input_dict.items():
        odict[k] = v.cpu().numpy()
    for i, input in enumerate(input_list):
        if type(input) in (list, tuple):
            odict[input_names[i]] = tuple([ip.cpu().numpy() for ip in input])
        else:
            odict[input_names[i]] = input.cpu().numpy()
    return odict


def unpack_nested_neural_type(neural_type):
    if type(neural_type) in (list, tuple):
        return unpack_nested_neural_type(neural_type[0])
    return neural_type


class Exportable(ABC):
    """
    This Interface should be implemented by particular classes derived from nemo.core.NeuralModule or nemo.core.ModelPT.
    It gives these entities ability to be exported for deployment to formats such as ONNX.
    """

    @staticmethod
    def get_format(filename: str):
        _, ext = os.path.splitext(filename)
        try:
            return _EXT_DICT[ext]
        except KeyError:
            raise ValueError(f"Export file {filename} extension does not correspond to any export format!")

    @property
    def input_module(self):
        return self

    @property
    def output_module(self):
        return self

    def export(
        self,
        output: str,
        input_example=None,
        verbose=False,
        export_params=True,
        do_constant_folding=True,
        keep_initializers_as_inputs=False,
        onnx_opset_version=None,
        try_script: bool = False,
        set_eval: bool = True,
        check_trace: bool = False,
        use_dynamic_axes: bool = True,
        dynamic_axes=None,
        check_tolerance=0.01,
    ):
        my_args = locals()
        my_args.pop('self')

        exportables = []
        for m in self.modules():
            if isinstance(m, Exportable):
                exportables.append(m)

        qual_name = self.__module__ + '.' + self.__class__.__qualname__
        format = self.get_format(output)
        output_descr = f"{qual_name} exported to {format}"

        # Pytorch's default for None is too low, can't pass through
        if onnx_opset_version is None:
            onnx_opset_version = 13

        try:
            # Disable typechecks
            typecheck.set_typecheck_enabled(enabled=False)

            # Allow user to completely override forward method to export
            forward_method, old_forward_method = self._wrap_forward_method()

            # Set module to eval mode
            if set_eval:
                self.eval()

            if input_example is None:
                input_example = self._get_input_example()

            # Remove i/o examples from args we propagate to enclosed Exportables
            my_args.pop('output')
            my_args.pop('input_example')

            # Run (posibly overridden) prepare methods before calling forward()
            for ex in exportables:
                ex._prepare_for_export(**my_args)
            self._prepare_for_export(output=output, input_example=input_example, **my_args)

            input_list, input_dict = self._setup_input_example(input_example)
            input_names = self._process_input_names()
            output_names = self._process_output_names()
            output_example = tuple(self.forward(*input_list, **input_dict))

            with torch.jit.optimized_execution(True), torch.no_grad():
                jitted_model = None
                if try_script:
                    try:
                        jitted_model = torch.jit.script(module)
                    except Exception as e:
                        logging.error(f"jit.script() failed!\{e}")

                if format == ExportFormat.TORCHSCRIPT:
                    if jitted_model is None:
                        jitted_model = torch.jit.trace_module(
                            self,
                            {"forward": tuple(input_list) + tuple(input_dict.values())},
                            strict=False,
                            optimize=True,
                            check_trace=check_trace,
                            check_tolerance=check_tolerance,
                        )
                    if verbose:
                        logging.info(f"JIT code:\n{jitted_model.code}")
                    jitted_model.save(output)
                    assert os.path.exists(output)
                elif format == ExportFormat.ONNX:
                    if jitted_model is None:
                        jitted_model = self

                    # dynamic axis is a mapping from input/output_name => list of "dynamic" indices
                    if dynamic_axes is None and use_dynamic_axes:
                        dynamic_axes = get_input_dynamic_axes(self.input_module, input_names)
                        dynamic_axes = {**dynamic_axes, **get_output_dynamic_axes(self.output_module, output_names)}

                    torch.onnx.export(
                        jitted_model,
                        input_example,
                        output,
                        input_names=input_names,
                        output_names=output_names,
                        verbose=verbose,
                        export_params=export_params,
                        do_constant_folding=do_constant_folding,
                        keep_initializers_as_inputs=keep_initializers_as_inputs,
                        dynamic_axes=dynamic_axes,
                        opset_version=onnx_opset_version,
                    )

                    # Verify the model can be read, and is valid
                    onnx_model = onnx.load(output)
                    onnx.checker.check_model(onnx_model, full_check=True)

                    if check_trace:
                        self._verify_runtime(
                            onnx_model,
                            input_list,
                            input_dict,
                            input_names,
                            output_names,
                            output_example,
                            output,
                            check_tolerance,
                        )
                else:
                    raise ValueError(f'Encountered unknown export format {format}.')
        finally:
            typecheck.set_typecheck_enabled(enabled=True)
            if forward_method:
                type(self).forward = old_forward_method
            self._export_teardown()
        return ([output], [output_descr])

    def _verify_runtime(
        self, onnx_model, input_list, input_dict, input_names, output_names, output_example, output, check_tolerance
    ):
        try:
            import onnxruntime
        except (ImportError, ModuleNotFoundError):
            logging.warning(f"ONNX generated at {output}, not verified - please install onnxruntime.\n")
            return

        sess = onnxruntime.InferenceSession(onnx_model.SerializeToString())
        ort_out = sess.run(output_names, to_onnxrt_input(input_names, input_list, input_dict))
        all_good = True

        for i, out in enumerate(ort_out[0]):
            expected = output_example[i]
            if torch.is_tensor(expected):
                if not torch.allclose(
                    torch.from_numpy(out), expected.cpu(), rtol=check_tolerance, atol=100 * check_tolerance
                ):
                    all_good = False
                    logging.info(f"onnxruntime results mismatch! PyTorch(expected):\n{expected}\nONNXruntime:\n{out}")
        status = "SUCCESS" if all_good else "FAIL"
        logging.info(f"ONNX generated at {output} verified with onnxruntime : " + status)

    @property
    def disabled_deployment_input_names(self):
        """Implement this method to return a set of input names disabled for export"""
        return set()

    @property
    def disabled_deployment_output_names(self):
        """Implement this method to return a set of output names disabled for export"""
        return set()

    @property
    def supported_export_formats(self):
        """Implement this method to return a set of export formats supported. Default is all types."""
        return set([ExportFormat.ONNX, ExportFormat.TORCHSCRIPT])

    @staticmethod
    def _extract_dynamic_axes(name: str, ntype: NeuralType):
        """
        Implement this method to provide dynamic axes id for ONNX export.
        By default, this method will extract BATCH and TIME dimension ids from each provided input/output name argument.

        For example, if module/model accepts argument named "input_signal" with type corresponding to [Batch, Time, Dim]
        shape, then the returned result should contain "input_signal" -> [0, 1] because Batch and Time are dynamic axes
        as they can change from call to call during inference.

        Args:
            name: Name of input or output parameter
            ntype: Corresponding Neural Type

        Returns:

        """
        dynamic_axes = defaultdict(list)
        if type(ntype) in (list, tuple):
            ntype = unpack_nested_neural_type(ntype)

        if ntype.axes:
            for ind, axis in enumerate(ntype.axes):
                if axis.kind in [AxisKind.Batch, AxisKind.Time, AxisKind.Width, AxisKind.Height]:
                    dynamic_axes[name].append(ind)
        return dynamic_axes

    def _prepare_for_export(self, **kwargs):
        """
        Override this method to prepare module for export. This is in-place operation.
        Base version does common necessary module replacements (Apex etc)
        """
        replace_1D_2D = kwargs.get('replace_1D_2D', False)
        replace_for_export(self, replace_1D_2D)

    def _export_teardown(self):
        """
        Override this method for any teardown code after export.
        """
        pass

    def _wrap_forward_method(self):
        old_forward_method = None

        if hasattr(type(self), "forward_for_export"):
            forward_method = type(self).forward_for_export
            old_forward_method = type(self).forward
            type(self).forward = forward_method
        else:
            forward_method = None

        return forward_method, old_forward_method

    def _setup_input_example(self, input_example):
        input_list = list(input_example)
        input_dict = {}
        # process possible kwargs
        if isinstance(input_list[-1], dict):
            input_dict = input_list[-1]
            input_list = input_list[:-1]
        return input_list, input_dict

    def _get_input_example(self, **args):
        return self.input_module.input_example(**args)

    def _process_input_names(self):
        input_names = get_input_names(self.input_module)
        # remove unnecessary inputs for input_ports
        for name in self.disabled_deployment_input_names:
            if name in input_names:
                input_names.remove(name)
        return input_names

    def _process_output_names(self):
        output_names = get_output_names(self.output_module)
        # remove unnecessary inputs for input_ports
        for name in self.disabled_deployment_output_names:
            if name in output_names:
                output_names.remove(name)
        return output_names

    def _augment_output_filename(self, output, prepend: str):
        path, filename = os.path.split(output)
        filename = f"{prepend}-{filename}"
        return os.path.join(path, filename)
