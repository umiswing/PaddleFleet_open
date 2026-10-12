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

"""Unit tests for the ZCC EMA reshard fast check.

``Trainer.ema_weight_reshard`` decides whether every rank can read its EMA shards
straight from ``{rank}_0.distcp``. It tries, in order:

  1. ``Trainer._ema_resumable_locally_sharded_fast``: partial parse of
     ``state_dict_metadata`` plus the distcp pickle header, handling sharded tensors.
     It only ever answers "yes, load locally" (True) or "undecided" (None). It
     subsumes Paddle's unsharded-only ``check_resumable_locally_fast``, which is
     therefore not tried.
  2. The stock full ``paddle.load(metadata)`` + ``check_resumable_locally``.

The Paddle reader modules (``distcp_reader``, ``metadata_reader``) only exist on
newer Paddle builds, so they are injected through ``sys.modules`` here; no process
group or GPU collective is needed.
"""

import os
import sys
import types
import unittest
from unittest import mock

import paddle
import paddle.distributed as dist
from paddle.distributed import ShardedWeight
from paddle.distributed.flex_checkpoint.dcp import (
    load_state_dict as dcp_load_state_dict,
    metadata_manager as dcp_metadata_manager,
    utils as dcp_utils,
)
from paddle.distributed.flex_checkpoint.dcp.metadata import LocalTensorMetadata

from paddlefleet.trainer import trainer as trainer_module
from paddlefleet.trainer.trainer import Trainer

_DCP = "paddle.distributed.flex_checkpoint.dcp"
_DISTCP_READER = f"{_DCP}.distcp_reader"
_METADATA_READER = f"{_DCP}.metadata_reader"
_FAST_RESUMABLE = f"{_DCP}.fast_resumable"

CKPT = "/ckpt/checkpoint-10"
EMA_PATH = os.path.join(CKPT, "ema_state")
METADATA_PATH = os.path.join(EMA_PATH, "0.metadata")


def _sw(key, local_shape, global_shape, global_offset, flattened_range=None):
    numel = (
        flattened_range.stop - flattened_range.start
        if flattened_range is not None
        else None
    )
    return ShardedWeight(
        key=key,
        local_tensor=paddle.zeros(
            [numel] if numel is not None else list(local_shape),
            dtype="float32",
        ),
        local_shape=local_shape,
        global_shape=global_shape,
        global_offset=global_offset,
        is_flattened=flattened_range is not None,
        flattened_range=flattened_range,
    )


def _meta(local_shape, global_shape, global_offset, flattened_range=None):
    return LocalTensorMetadata(
        global_offset=global_offset,
        local_shape=local_shape,
        dtype="float32",
        global_shape=global_shape,
        is_flattened=flattened_range is not None,
        flattened_range=flattened_range,
    )


def _reader_modules(state_dict_metadata, file_shapes):
    """Fake ``distcp_reader`` / ``metadata_reader`` modules."""
    distcp_reader = types.ModuleType(_DISTCP_READER)
    if isinstance(file_shapes, Exception):
        distcp_reader.scan_tensor_shapes = mock.MagicMock(
            side_effect=file_shapes
        )
    else:
        distcp_reader.scan_tensor_shapes = mock.MagicMock(
            return_value=file_shapes
        )
    metadata_reader = types.ModuleType(_METADATA_READER)
    metadata_reader.load_state_dict_metadata = mock.MagicMock(
        return_value=state_dict_metadata
    )
    return {_DISTCP_READER: distcp_reader, _METADATA_READER: metadata_reader}


class TestShardedFastCheck(unittest.TestCase):
    """``_ema_resumable_locally_sharded_fast`` on a single process."""

    def setUp(self):
        self.harness = types.SimpleNamespace(args=types.SimpleNamespace())
        self.logger = self._patch(trainer_module, "logger")
        self.all_gather = self._patch(dist, "all_gather_object")
        self.validator = self._patch(
            trainer_module, "DistInfoCollectorValidator"
        )
        self.validator.return_value.check_same_strategy.return_value = (
            True,
            None,
        )

    def _patch(self, target, name, **kwargs):
        patcher = mock.patch.object(target, name, **kwargs)
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked

    def _run(
        self, ema_target, state_dict_metadata, file_shapes, use_dist=False
    ):
        self.modules = _reader_modules(state_dict_metadata, file_shapes)
        with mock.patch.dict(sys.modules, self.modules):
            return Trainer._ema_resumable_locally_sharded_fast(
                self.harness, METADATA_PATH, EMA_PATH, ema_target, use_dist
            )

    def _scan(self):
        return self.modules[_DISTCP_READER].scan_tensor_shapes

    # TP=2 split of a [4, 8] weight along dim 1, this rank holds the left half.
    def _tp_target(self):
        return {"w": _sw("w", (4, 4), (4, 8), (0, 0))}

    def _tp_metadata(self, right_shape=(4, 4)):
        return {
            "w": [
                _meta((4, 4), (4, 8), (0, 0)),
                _meta(right_shape, (4, 8), (0, 4)),
            ]
        }

    def test_missing_reader_modules_declines(self):
        with mock.patch.dict(
            sys.modules, {_DISTCP_READER: None, _METADATA_READER: None}
        ):
            result = Trainer._ema_resumable_locally_sharded_fast(
                self.harness, METADATA_PATH, EMA_PATH, self._tp_target(), True
            )
        self.assertIsNone(result)
        self.all_gather.assert_not_called()

    def test_partial_parse_failure_declines_before_collective(self):
        self.assertIsNone(self._run(self._tp_target(), None, {}, use_dist=True))
        self.modules[
            _METADATA_READER
        ].load_state_dict_metadata.assert_called_once_with(
            METADATA_PATH, allow_full_load=False
        )
        self._scan().assert_not_called()
        self.all_gather.assert_not_called()

    def test_unique_shard_shape_is_confirmed(self):
        # The other shard has a different shape, so the header shape alone pins
        # the file to this rank's shard and no strategy check is needed.
        result = self._run(
            self._tp_target(),
            self._tp_metadata(right_shape=(4, 3)),
            {"w": (4, 4)},
        )
        self.assertTrue(result)
        self._scan().assert_called_once_with(
            os.path.join(EMA_PATH, "0_0.distcp")
        )
        self.validator.assert_not_called()

    def test_flattened_shard_uses_1d_stored_shape(self):
        target = {"w": _sw("w", (4, 8), (4, 8), (0, 0), slice(0, 20))}
        metadata = {
            "w": [
                _meta((4, 8), (4, 8), (0, 0), (0, 20)),
                _meta((4, 8), (4, 8), (0, 0), (20, 32)),
            ]
        }
        self.assertTrue(self._run(target, metadata, {"w": (20,)}))
        self.validator.assert_not_called()

    def test_flattened_shard_with_other_range_declines(self):
        target = {"w": _sw("w", (4, 8), (4, 8), (0, 0), slice(0, 16))}
        metadata = {
            "w": [
                _meta((4, 8), (4, 8), (0, 0), (0, 20)),
                _meta((4, 8), (4, 8), (0, 0), (20, 32)),
            ]
        }
        self.assertIsNone(self._run(target, metadata, {"w": (16,)}))

    def test_single_metadata_entry_is_accepted(self):
        target = {"b": _sw("b", (8,), (8,), (0,))}
        self.assertTrue(
            self._run(target, {"b": _meta((8,), (8,), (0,))}, {"b": (8,)})
        )

    def test_tensor_without_local_shard_is_skipped(self):
        # extract_tensor_metadata returns (None, None) for a tensor this rank holds
        # no data of; such entries are not read from the file at all.
        not_local = object()
        real_extract = dcp_utils.extract_tensor_metadata
        target = {"b": _sw("b", (8,), (8,), (0,)), "absent": not_local}
        with mock.patch.object(
            dcp_utils,
            "extract_tensor_metadata",
            side_effect=lambda v: (
                (None, None) if v is not_local else real_extract(v)
            ),
        ):
            self.assertTrue(
                self._run(target, {"b": _meta((8,), (8,), (0,))}, {"b": (8,)})
            )

    def test_key_missing_from_checkpoint_declines(self):
        self.assertIsNone(self._run(self._tp_target(), {}, {"w": (4, 4)}))
        self.assertIn(
            "not in the checkpoint", self.logger.info.call_args_list[0].args[0]
        )

    def test_shard_missing_from_checkpoint_declines(self):
        # Saved with TP=1, now loading with TP=2: the [0:4, 0:4] shard never existed.
        metadata = {"w": [_meta((4, 8), (4, 8), (0, 0))]}
        self.assertIsNone(self._run(self._tp_target(), metadata, {"w": (4, 8)}))

    def test_file_shape_mismatch_declines(self):
        self.assertIsNone(
            self._run(
                self._tp_target(),
                self._tp_metadata(right_shape=(4, 3)),
                {"w": (4, 3)},
            )
        )

    def test_key_missing_from_file_declines(self):
        self.assertIsNone(
            self._run(
                self._tp_target(), self._tp_metadata(right_shape=(4, 3)), {}
            )
        )

    def test_unreadable_distcp_declines(self):
        self.assertIsNone(
            self._run(
                self._tp_target(),
                self._tp_metadata(),
                FileNotFoundError("0_0.distcp"),
            )
        )
        self.assertIn(
            "FileNotFoundError", self.logger.info.call_args_list[0].args[0]
        )

    def test_ambiguous_shape_with_same_strategy_is_confirmed(self):
        self.assertTrue(
            self._run(self._tp_target(), self._tp_metadata(), {"w": (4, 4)})
        )
        # hcg is not set on the harness, so the validator resolves it itself.
        self.validator.assert_called_once_with(self.harness.args, None)
        self.validator.return_value.check_same_strategy.assert_called_once_with(
            CKPT
        )

    def test_ambiguous_shape_with_changed_strategy_declines(self):
        self.validator.return_value.check_same_strategy.return_value = (
            False,
            "mp_degree not match: 2 vs 4",
        )
        self.assertIsNone(
            self._run(self._tp_target(), self._tp_metadata(), {"w": (4, 4)})
        )
        self.assertIn(
            "parallel strategy changed",
            self.logger.info.call_args_list[0].args[0],
        )

    def test_ambiguous_shape_without_model_meta_declines(self):
        self.validator.return_value.check_same_strategy.side_effect = (
            AssertionError("model_meta.json not exist")
        )
        self.assertIsNone(
            self._run(self._tp_target(), self._tp_metadata(), {"w": (4, 4)})
        )
        self.assertIn(
            "cannot verify parallel strategy",
            self.logger.info.call_args_list[0].args[0],
        )

    def test_uses_hcg_when_available(self):
        self.harness.hcg = object()
        self._run(self._tp_target(), self._tp_metadata(), {"w": (4, 4)})
        self.validator.assert_called_once_with(
            self.harness.args, self.harness.hcg
        )


class TestShardedFastCheckDistributed(unittest.TestCase):
    """With ``use_dist`` the verdict is agreed through one ``all_gather_object``."""

    def setUp(self):
        self.harness = types.SimpleNamespace(args=types.SimpleNamespace())
        for target, name in (
            (trainer_module, "logger"),
            (trainer_module, "DistInfoCollectorValidator"),
        ):
            patcher = mock.patch.object(target, name)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(dist, "get_rank", return_value=3)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _run(self, file_shapes, peers):
        gathered = []

        def _all_gather(results, obj):
            gathered.append(obj)
            results.extend([obj, *peers])

        target = {"w": _sw("w", (4, 4), (4, 8), (0, 0))}
        metadata = {
            "w": [_meta((4, 4), (4, 8), (0, 0)), _meta((4, 3), (4, 8), (0, 4))]
        }
        modules = _reader_modules(metadata, file_shapes)
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(
                dist, "all_gather_object", side_effect=_all_gather
            ) as all_gather,
        ):
            result = Trainer._ema_resumable_locally_sharded_fast(
                self.harness, METADATA_PATH, EMA_PATH, target, True
            )
        all_gather.assert_called_once()
        modules[_DISTCP_READER].scan_tensor_shapes.assert_called_once_with(
            os.path.join(EMA_PATH, "3_0.distcp")
        )
        return result, gathered

    def test_all_ranks_agree(self):
        result, gathered = self._run({"w": (4, 4)}, [(True, True, 0)] * 3)
        self.assertTrue(result)
        self.assertEqual(gathered, [(True, True, 0)])

    def test_one_peer_fails(self):
        result, _ = self._run(
            {"w": (4, 4)}, [(True, True, 0), (False, True, 0), (True, True, 0)]
        )
        self.assertIsNone(result)

    def test_peer_with_changed_strategy(self):
        result, _ = self._run({"w": (4, 4)}, [(True, False, 2)])
        self.assertIsNone(result)

    def test_local_failure_still_joins_collective(self):
        # A rank that already knows it cannot load locally must still take part in
        # the all_gather, otherwise the other ranks would hang.
        result, gathered = self._run({"w": (4, 3)}, [(True, True, 0)])
        self.assertIsNone(result)
        self.assertEqual(gathered, [(False, True, 0)])


class TestEmaWeightReshardFallbackChain(unittest.TestCase):
    def setUp(self):
        self.ema_target = {"w": object()}
        self.harness = types.SimpleNamespace(
            _build_ema_target=mock.MagicMock(return_value=self.ema_target),
            _ema_resumable_locally_sharded_fast=mock.MagicMock(
                return_value=None
            ),
        )
        patches = (
            (trainer_module, "logger", {}),
            (
                dcp_load_state_dict,
                "get_checkpoint_files",
                {"return_value": (["0.metadata"], ["0_0.distcp"])},
            ),
            (dcp_metadata_manager, "MetadataManager", {}),
            (dcp_utils, "check_resumable_locally", {"return_value": True}),
            (paddle, "load", {"return_value": "full-metadata"}),
            (dist, "get_world_size", {"return_value": 8}),
        )
        mocks = []
        for target, name, kwargs in patches:
            patcher = mock.patch.object(target, name, **kwargs)
            mocks.append(patcher.start())
            self.addCleanup(patcher.stop)
        (
            _,
            _,
            self.metadata_manager,
            self.full_check,
            self.paddle_load,
            _,
        ) = mocks

    def _run(self):
        # Paddle's unsharded-only check_resumable_locally_fast is a strict subset of
        # the sharded fast check, so it must never be consulted even when shipped.
        module = types.ModuleType(_FAST_RESUMABLE)
        module.check_resumable_locally_fast = mock.MagicMock()
        with mock.patch.dict(sys.modules, {_FAST_RESUMABLE: module}):
            result = Trainer.ema_weight_reshard(self.harness, EMA_PATH)
        module.check_resumable_locally_fast.assert_not_called()
        return result

    def test_fast_check_short_circuits(self):
        self.harness._ema_resumable_locally_sharded_fast.return_value = True
        self.assertFalse(self._run())
        self.harness._ema_resumable_locally_sharded_fast.assert_called_once_with(
            METADATA_PATH, EMA_PATH, self.ema_target, True
        )
        self.harness._build_ema_target.assert_called_once_with(allocate=False)
        self.paddle_load.assert_not_called()
        self.full_check.assert_not_called()

    def test_undecided_fast_check_falls_back_to_full_check(self):
        for full_resumable, need_reshard in ((True, False), (False, True)):
            with self.subTest(full_resumable=full_resumable):
                self.full_check.reset_mock()
                self.full_check.return_value = full_resumable
                self.assertEqual(self._run(), need_reshard)
                self.paddle_load.assert_called_with(METADATA_PATH)
                self.full_check.assert_called_once_with(
                    EMA_PATH,
                    self.ema_target,
                    self.metadata_manager.return_value,
                    True,
                    None,
                )
        self.metadata_manager.return_value.set_metadata_list.assert_called_with(
            ["full-metadata"]
        )

    def test_single_process(self):
        dist.get_world_size.return_value = 1
        self._run()
        self.assertFalse(
            self.harness._ema_resumable_locally_sharded_fast.call_args.args[3]
        )
        self.assertFalse(self.full_check.call_args.args[3])


if __name__ == "__main__":
    unittest.main()
