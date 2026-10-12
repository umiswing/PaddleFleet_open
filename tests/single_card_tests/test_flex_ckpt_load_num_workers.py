# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Unit tests for the ``flex_ckpt_load_num_workers`` knob.

``num_workers`` only exists on newer paddle builds, so the trainer has to decide at
runtime whether the argument can be forwarded to ``dist.load_state_dict``. What is
pinned down here:

  1. An explicit ``num_workers`` parameter is detected.
  2. A ``**kwargs``-style signature (a wrapper installed around the real function, e.g.
     by a profiler) counts as supported, because the argument reaches the wrapped
     callable.
  3. A paddle build without the parameter is detected, so the trainer falls back to the
     serial read instead of raising ``TypeError``.
  4. A callable ``inspect.signature`` cannot introspect is treated as unsupported rather
     than letting the exception escape into the checkpoint load path.
  5. The training argument defaults to 1, i.e. the original serial read.
  6. ``Trainer._load_ema_with_reshard`` takes the forwarded kwargs as an optional
     trailing argument, so the ZCC EMA reshard read is covered by the same knob without
     breaking the existing positional call sites.
  7. ``Trainer._load_flex_checkpoint`` forwards ``num_workers`` to every
     ``dist.load_state_dict`` call (HF safetensors, master weights, optimizer state,
     EMA state, model state and the ZCC EMA reshard) only when the knob is > 1 and the
     installed paddle accepts it; otherwise it keeps the serial read.

``_load_flex_checkpoint`` is driven on a stub trainer with ``dist.load_state_dict`` and
the other collective / IO helpers mocked out, so no process group or GPU is needed.
"""

import inspect
import os
import tempfile
import types
import unittest
from dataclasses import fields
from unittest import mock

import paddle
import paddle.distributed as dist

from paddlefleet.trainer import trainer as trainer_module
from paddlefleet.trainer.trainer import Trainer, _supports_load_num_workers
from paddlefleet.trainer.training_args import TrainingArguments
from paddlefleet.utils.env import (
    EMA_STATE_DIC,
    MASTER_WEIGHT_DIC,
    MODEL_STATE_DIC,
    OPTIMIZER_STATE_DIC,
)


def _load_state_dict_with_num_workers(state_dict, path=None, num_workers=1):
    pass


def _load_state_dict_wrapper(*args, **kwargs):
    pass


def _load_state_dict_legacy(state_dict, path=None, process_group=None):
    pass


class TestSupportsLoadNumWorkers(unittest.TestCase):
    def setUp(self):
        _supports_load_num_workers.cache_clear()

    def tearDown(self):
        _supports_load_num_workers.cache_clear()

    def _probe(self, fn):
        with mock.patch.object(dist, "load_state_dict", fn):
            return _supports_load_num_workers()

    def test_explicit_parameter_is_supported(self):
        self.assertTrue(self._probe(_load_state_dict_with_num_workers))

    def test_var_keyword_signature_is_supported(self):
        self.assertTrue(self._probe(_load_state_dict_wrapper))

    def test_missing_parameter_is_not_supported(self):
        self.assertFalse(self._probe(_load_state_dict_legacy))

    def test_uninspectable_callable_is_not_supported(self):
        with mock.patch.object(
            inspect, "signature", side_effect=ValueError("no signature")
        ):
            self.assertFalse(_supports_load_num_workers())

    def test_result_is_cached(self):
        self.assertTrue(self._probe(_load_state_dict_with_num_workers))
        # The second probe must not re-inspect: the cached answer wins over the
        # new signature.
        self.assertTrue(self._probe(_load_state_dict_legacy))


class TestFlexCkptLoadNumWorkersArgument(unittest.TestCase):
    def test_defaults_to_serial_read(self):
        field = {f.name: f for f in fields(TrainingArguments)}[
            "flex_ckpt_load_num_workers"
        ]
        self.assertEqual(field.type, int)
        self.assertEqual(field.default, 1)
        self.assertIn("help", field.metadata)


class TestEmaReshardAcceptsLoadKwargs(unittest.TestCase):
    """The ZCC EMA reshard read also honours the knob.

    ``_load_ema_with_reshard`` is called positionally both from
    ``_load_flex_checkpoint`` and from the multi-card layout tests, so the forwarded
    kwargs have to be an optional trailing parameter.
    """

    def test_load_kwargs_is_optional_and_trailing(self):
        params = list(
            inspect.signature(Trainer._load_ema_with_reshard).parameters
        )
        self.assertEqual(params[-1], "load_kwargs")
        self.assertIsNone(
            inspect.signature(Trainer._load_ema_with_reshard)
            .parameters["load_kwargs"]
            .default
        )

    def _reshard_harness(self):
        harness = types.SimpleNamespace(
            args=types.SimpleNamespace(aoa_config=None, load_via_cpu=False),
            _build_ema_target=mock.MagicMock(return_value={}),
        )
        return harness

    def test_forwards_load_kwargs(self):
        harness = self._reshard_harness()
        with mock.patch.object(dist, "load_state_dict") as load:
            result = Trainer._load_ema_with_reshard(
                harness, "/ckpt/ema", "broadcast", None, {"num_workers": 4}
            )
        self.assertEqual(result, {})
        load.assert_called_once()
        self.assertEqual(load.call_args.args[1], "/ckpt/ema")
        self.assertEqual(load.call_args.kwargs["num_workers"], 4)
        self.assertEqual(load.call_args.kwargs["comm_method"], "broadcast")

    def test_positional_call_keeps_serial_read(self):
        harness = self._reshard_harness()
        with mock.patch.object(dist, "load_state_dict") as load:
            Trainer._load_ema_with_reshard(
                harness, "/ckpt/ema", "broadcast", None
            )
        load.assert_called_once()
        self.assertNotIn("num_workers", load.call_args.kwargs)


def _fp32_weight():
    return types.SimpleNamespace(
        local_tensor=types.SimpleNamespace(dtype=paddle.float32)
    )


def _stub_trainer(**arg_overrides):
    """The smallest object ``Trainer._load_flex_checkpoint`` runs on.

    Defaults take the non-HF path, load master weights, optimizer state and model
    state, and skip bf16 master-weight casting (``bf16=False``).
    """
    args = types.SimpleNamespace(
        flex_ckpt_load_num_workers=1,
        flex_ckpt_comm_method="broadcast",
        load_from_hf=False,
        sharded_model_from_ema=False,
        enable_zero_cost_checkpoint=False,
        zcc_save_ema_coef=None,
        ignore_load_lr_and_optim=False,
        aoa_config=None,
        load_via_cpu=False,
        tensorwise_offload_optimizer=False,
        bf16=False,
    )
    for key, value in arg_overrides.items():
        setattr(args, key, value)
    harness = types.SimpleNamespace(
        args=args,
        model=types.SimpleNamespace(
            config=None,
            sharded_state_dict=lambda: {"linear.weight": _fp32_weight()},
        ),
        optimizer=types.SimpleNamespace(
            sharded_state_dict=lambda _model_sd: {
                "linear.weight.w_0": _fp32_weight(),
                "linear.weight.moment1_0": _fp32_weight(),
            }
        ),
        _load_scheduler=mock.MagicMock(),
        ema_weight_reshard=mock.MagicMock(return_value=True),
        _load_ema_with_reshard=mock.MagicMock(return_value={}),
    )
    harness._is_fc_format_ema = Trainer._is_fc_format_ema.__get__(harness)
    return harness


class TestLoadFlexCheckpointForwardsNumWorkers(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ckpt = tmp.name
        for sub in (
            MODEL_STATE_DIC,
            OPTIMIZER_STATE_DIC,
            MASTER_WEIGHT_DIC,
            EMA_STATE_DIC,
        ):
            os.makedirs(os.path.join(self.ckpt, sub))
            with open(os.path.join(self.ckpt, sub, "0.metadata"), "w"):
                pass

        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("HACK_CONVERT_CKPT", None)
        os.environ.pop("HACK_CONVERT_CKPT_NOPP_TO_PP", None)

        self.load = self._patch(dist, "load_state_dict")
        self.supports = self._patch(
            trainer_module, "_supports_load_num_workers", return_value=True
        )
        self.logger = self._patch(trainer_module, "logger")
        self._patch(dist.fleet, "get_hybrid_communicate_group")
        self._patch(
            paddle,
            "load",
            return_value=types.SimpleNamespace(state_dict_metadata={}),
        )
        self._patch(trainer_module, "init_optimizer")

    def _patch(self, target, name, **kwargs):
        patcher = mock.patch.object(target, name, **kwargs)
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked

    def _loaded_paths(self):
        return [call.args[1] for call in self.load.call_args_list]

    def _assert_num_workers(self, expected):
        self.assertTrue(self.load.call_args_list)
        for call in self.load.call_args_list:
            if expected is None:
                self.assertNotIn("num_workers", call.kwargs)
            else:
                self.assertEqual(call.kwargs["num_workers"], expected)

    def test_default_keeps_serial_read(self):
        harness = _stub_trainer()
        Trainer._load_flex_checkpoint(harness, self.ckpt)

        self.assertEqual(
            self._loaded_paths(),
            [
                os.path.join(self.ckpt, MASTER_WEIGHT_DIC),
                os.path.join(self.ckpt, OPTIMIZER_STATE_DIC),
                os.path.join(self.ckpt, MODEL_STATE_DIC),
            ],
        )
        self._assert_num_workers(None)
        # A single worker never needs the capability probe.
        self.supports.assert_not_called()

    def test_none_value_is_treated_as_serial_read(self):
        harness = _stub_trainer(flex_ckpt_load_num_workers=None)
        Trainer._load_flex_checkpoint(harness, self.ckpt)
        self._assert_num_workers(None)
        self.supports.assert_not_called()

    def test_forwards_num_workers_to_every_load(self):
        harness = _stub_trainer(
            flex_ckpt_load_num_workers=4,
            enable_zero_cost_checkpoint=True,
            zcc_save_ema_coef=0.99,
        )
        Trainer._load_flex_checkpoint(harness, self.ckpt)

        self.assertEqual(len(self.load.call_args_list), 3)
        self._assert_num_workers(4)
        harness._load_scheduler.assert_called_once()
        # The ZCC EMA reshard read gets the same kwargs, positionally.
        harness._load_ema_with_reshard.assert_called_once_with(
            os.path.join(self.ckpt, EMA_STATE_DIC),
            "broadcast",
            None,
            {"num_workers": 4},
        )
        self.assertTrue(
            any(
                "num_workers=4" in str(call)
                for call in self.logger.info.call_args_list
            )
        )

    def test_aligned_ema_skips_reshard(self):
        harness = _stub_trainer(
            flex_ckpt_load_num_workers=4,
            enable_zero_cost_checkpoint=True,
            zcc_save_ema_coef=0.99,
        )
        harness.ema_weight_reshard.return_value = False
        Trainer._load_flex_checkpoint(harness, self.ckpt)
        harness._load_ema_with_reshard.assert_not_called()
        self._assert_num_workers(4)

    def test_unsupported_paddle_falls_back_to_serial_read(self):
        self.supports.return_value = False
        harness = _stub_trainer(flex_ckpt_load_num_workers=4)
        Trainer._load_flex_checkpoint(harness, self.ckpt)

        self.supports.assert_called_once()
        self._assert_num_workers(None)
        self.logger.warning.assert_called_once()
        self.assertIn(
            "flex_ckpt_load_num_workers=4",
            self.logger.warning.call_args.args[0],
        )

    def test_hf_safetensors_load(self):
        harness = _stub_trainer(
            flex_ckpt_load_num_workers=4,
            load_from_hf=True,
            ignore_load_lr_and_optim=True,
        )
        self._patch(trainer_module, "resolve_aoa_config", return_value=None)
        self._patch(
            trainer_module, "hf_checkpoint_is_quantized", return_value=False
        )
        self._patch(
            trainer_module, "build_hf_dequant_load_transform", return_value=None
        )
        Trainer._load_flex_checkpoint(harness, self.ckpt)

        self.load.assert_called_once()
        self.assertEqual(self._loaded_paths(), [self.ckpt])
        self.assertTrue(self.load.call_args.kwargs["safetensors"])
        self.assertNotIn("load_transform", self.load.call_args.kwargs)
        self._assert_num_workers(4)

    def test_sharded_model_from_fc_ema(self):
        harness = _stub_trainer(
            flex_ckpt_load_num_workers=4, sharded_model_from_ema=True
        )
        Trainer._load_flex_checkpoint(harness, self.ckpt)

        self.assertEqual(
            self._loaded_paths(), [os.path.join(self.ckpt, EMA_STATE_DIC)]
        )
        # Both the fp32 model weight and the master weight are read from EMA.
        self.assertEqual(
            sorted(self.load.call_args.args[0]),
            ["linear.weight", "linear.weight.w_0"],
        )
        self._assert_num_workers(4)


if __name__ == "__main__":
    unittest.main()
