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

"""grad_health exact cross-rank norms under a real distributed context.

The single-card suite covers the metric math and the AMP de-scale; this case
exists for the path that only runs with >1 rank: ``ExactNormReducer``'s group
planning and the shard/total ``all_reduce`` in ``reduce``, plus
``_emit_exact_norms``. It drives them through a real backward on an 8-GPU
``mp=2 x sharding=4`` mesh with sequence parallel on, so the reducer gets both a
``tp`` shard group and a ``sharding`` data group covering the whole world, and
checks the reduced global norms against hand values.

Launched by ``ci/multi-card_test.sh`` via
``coverage run -m paddle.distributed.launch --gpus 0..7``.
"""

import math
import unittest
from types import SimpleNamespace

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import fleet

from paddlefleet.internal_medicine.backends.paddlefleet.grad_monitor import (
    PaddleGradHealthMonitor,
)
from paddlefleet.internal_medicine.core.training_logs import training_logs
from paddlefleet.training.initialize import initialize_fleet

WIDTH = 8
SEED_VALUE = 2.0  # uniform across ranks, so a correct reduction returns it back


class FakeBranch(nn.Layer):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor


class FakeLayer(nn.Layer):
    """Two residual branches, both hooked, plus the layer output itself."""

    def __init__(self, idx, attn_factor=0.5, mlp_factor=0.25):
        super().__init__()
        self.idx = idx
        self.self_attn = FakeBranch(attn_factor)
        self.mlp = FakeBranch(mlp_factor)

    def forward(self, x):
        x = x + self.self_attn(x)
        return x + self.mlp(x)


def _model(layers):
    # ``_detect_sp`` reads ``config.sequence_parallel``; with it on, the reducer
    # counts the tp group as a shard group instead of a replica.
    return SimpleNamespace(
        decoder=SimpleNamespace(layers=nn.LayerList(layers)),
        config=SimpleNamespace(sequence_parallel=True),
    )


def _run_backward(layers, seed):
    x = paddle.ones([2, WIDTH], dtype="float32")
    x.stop_gradient = False
    out = x
    for layer in layers:
        out = layer(out)
    (out * seed).sum().backward()


class GradHealthDistributedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 2,
            "pp_degree": 1,
            "sharding_degree": 4,
            "sep_degree": 1,
            "cp_degree": 1,
            "ep_degree": 1,
            "moe_sharding_degree": 1,
            "order": [
                "sharding",
                "moe_sharding",
                "pp",
                "sep",
                "cp",
                "dp",
                "ep",
                "mp",
            ],
        }
        initialize_fleet(strategy=strategy)

    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_reducer_engages_the_tp_shard_group(self):
        monitor = PaddleGradHealthMonitor(log_global=True)
        monitor.register_hooks(_model([FakeLayer(0)]))
        r = monitor._reducer
        self.assertTrue(
            r.enabled, "reducer did not find any distributed groups"
        )
        self.assertIn("tp", [g.name for g in r.shard_groups])
        # mp=2 (tp) x sharding=4 should cover the whole 8-rank world exactly.
        self.assertEqual(r.total_factor, dist.get_world_size())
        self.assertTrue(r.exact_total)

    def test_exact_global_norms_reduce_across_ranks(self):
        layers = [FakeLayer(0)]
        monitor = PaddleGradHealthMonitor(log_global=True)
        monitor.register_hooks(_model(layers))
        # Uniform gradient on every rank: the layer-out grad equals the seed, so a
        # correct cross-rank reduction returns rms_global == SEED_VALUE regardless
        # of which ranks the covering group spans.
        _run_backward(layers, paddle.full([2, WIDTH], SEED_VALUE))
        monitor.finalize_scaled_grad_metrics(None)
        monitor.step()

        latest = training_logs.get_latest(prefix="grad_health")
        rms_key = "grad_health/layer_0/layer_out_rms_global"
        self.assertIn(rms_key, latest)
        self.assertAlmostEqual(latest[rms_key], SEED_VALUE, places=3)

        # norm_global is only emitted when the chosen groups cover the world
        # exactly; then it is sqrt(sum of squares) over that cover.
        norm_key = "grad_health/layer_0/layer_out_norm_global"
        if monitor._reducer.exact_total:
            self.assertIn(norm_key, latest)
            cover = monitor._reducer.total_factor
            expected_norm = SEED_VALUE * math.sqrt(cover * 2 * WIDTH)
            self.assertAlmostEqual(latest[norm_key], expected_norm, places=2)


if __name__ == "__main__":
    unittest.main()
