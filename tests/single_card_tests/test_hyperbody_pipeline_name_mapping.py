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

"""CPU behaviour tests for ``HyperBodyUnifiedModel._set_pipeline_name_mapping``
and its ``_shared_layer_prefix`` helper.

The mapping is pure Python orchestration over ``_sequential_layers`` and the raw
``nn.Layer.state_dict`` keys (what ``super().state_dict()`` returns); it needs
neither an accelerator nor the expensive ``PipelineLayer.__init__`` (which
requires a Fleet hybrid-communicate group). The model is created via ``__new__``
so the ``PretrainedModel`` metaclass ``__init__`` wrapper never runs, keeping the
real methods under test; the base ``nn.Layer.state_dict`` (the raw key source) is
replaced by a controlled stand-in so the remapping algorithm is observed end to
end. Expected values are derived by hand from the production source.

MTP weight sharing registers a ``SharedLayerDesc("mtp_reuse_transformer", ...)``
for both the backbone pivot (the last backbone layer, registered FIRST) and each
MTP layer. ``PipelineLayer`` names a shared param ``shared_layers.{key}.rest``;
the fix resolves such keys to the FIRST matching ``SharedLayerDesc``'s
``name_prefix`` (the pivot/owner), mirroring ``GPTModel.get_shardlayer_prefix``.
"""

import unittest
from unittest.mock import patch

try:
    from paddle.distributed.fleet.meta_parallel import SharedLayerDesc
    from paddle.nn import Layer

    from paddlefleet.transformers.hyperbody.modeling import (
        HyperBodyUnifiedModel,
    )

    def _make_unified(sequential_layers):
        """A real ``HyperBodyUnifiedModel`` instance with ONLY the attributes the
        mapping methods read, created via ``__new__`` so the ``PretrainedModel``
        metaclass ``__init__`` wrapper (which runs ``init_weights`` and needs a
        Fleet-built sub-layer tree) never fires. Attributes are written straight
        into ``__dict__`` to bypass ``nn.Layer.__setattr__`` (which also expects
        that tree). Being a genuine instance is required so the zero-arg
        ``super().state_dict()`` inside ``_set_pipeline_name_mapping`` resolves.
        """
        enc = HyperBodyUnifiedModel.__new__(HyperBodyUnifiedModel)
        enc.__dict__["_sequential_layers"] = sequential_layers
        enc.__dict__["_single_to_pp_mapping"] = None
        enc.__dict__["_pp_to_single_mapping"] = None
        return enc

    class _DummyLayer(Layer):
        """A minimal ``nn.Layer`` subclass. ``LayerDesc`` requires ``layer_func``
        to subclass ``nn.Layer``; it is only stored (never built) here."""

    class _Param:
        """Opaque state-dict value carrier (remapping treats values as opaque)."""

        def __init__(self, tag):
            self.tag = tag

    _IMPORT_ERROR = None
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
    Layer = None
    SharedLayerDesc = None
    HyperBodyUnifiedModel = None
    _make_unified = None
    _DummyLayer = None
    _Param = None
    _IMPORT_ERROR = exc


_SKIP_REASON = (
    ""
    if _IMPORT_ERROR is None
    else f"paddle/paddlefleet import unavailable: {_IMPORT_ERROR!r}"
)


def _shared(key):
    """A genuine ``SharedLayerDesc`` carrying ``key`` as ``layer_name``."""
    return SharedLayerDesc(
        key, _DummyLayer, shared_weight_attr="transformer_layer_weights"
    )


@unittest.skipUnless(_IMPORT_ERROR is None, _SKIP_REASON)
class TestSetPipelineNameMapping(unittest.TestCase):
    def _make(self):
        """A PP=1 layout mirroring ``get_layer_desc_list`` with MTP sharing:

        0 encoder             -> "encoder"
        1 decoder embedding   -> "model"
        2 backbone L0         -> "model.layers.0"
        3 backbone L1 (pivot) -> "model.layers.1"   SharedLayerDesc owner
        4 final norm          -> "model"
        5 MTP layer           -> "model.layers.2"   SharedLayerDesc alias
        6 lm_head             -> "model.lm_head"
        """
        return _make_unified(
            [
                {"layer": object(), "name_prefix": "encoder"},
                {"layer": object(), "name_prefix": "model"},
                {"layer": object(), "name_prefix": "model.layers.0"},
                {
                    "layer": _shared("mtp_reuse_transformer"),
                    "name_prefix": "model.layers.1",
                },
                {"layer": object(), "name_prefix": "model"},
                {
                    "layer": _shared("mtp_reuse_transformer"),
                    "name_prefix": "model.layers.2",
                },
                {"layer": object(), "name_prefix": "model.lm_head"},
            ]
        )

    def test_digit_shared_and_passthrough_keys(self):
        enc = self._make()
        raw = {
            "0.image_encoder.pos_embed": _Param("a"),
            "1.embed_tokens.weight": _Param("b"),
            "2.self_attn.o_proj.weight": _Param("c"),
            "3.self_attn.o_proj.weight": _Param("d"),
            # Shared param: PipelineLayer names it ``shared_layers.{key}.rest``.
            # It must resolve to the FIRST matching SharedLayerDesc's prefix
            # (the pivot "model.layers.1"), NOT the later MTP alias.
            "shared_layers.mtp_reuse_transformer.self_attn.o_proj.weight": _Param(
                "e"
            ),
            "6.weight": _Param("f"),
            # Non-digit, non-"shared_layers" head passes through unchanged.
            "extra.weight": _Param("g"),
        }

        def fake_state_dict(self, *args, **kwargs):
            return dict(raw)

        with patch.object(Layer, "state_dict", fake_state_dict):
            returned = enc._set_pipeline_name_mapping()

        # The digit-"3" key and the shared key both resolve to the single name
        # "model.layers.1.self_attn.o_proj.weight"; single_to_pp keeps only the
        # last writer, so assert both directions explicitly rather than against a
        # collapsed single_to_pp dict.
        self.assertEqual(
            enc._pp_to_single_mapping["3.self_attn.o_proj.weight"],
            "model.layers.1.self_attn.o_proj.weight",
        )
        self.assertEqual(
            enc._pp_to_single_mapping[
                "shared_layers.mtp_reuse_transformer.self_attn.o_proj.weight"
            ],
            "model.layers.1.self_attn.o_proj.weight",
        )
        self.assertEqual(
            returned["encoder.image_encoder.pos_embed"],
            "0.image_encoder.pos_embed",
        )
        self.assertEqual(
            returned["model.embed_tokens.weight"], "1.embed_tokens.weight"
        )
        self.assertEqual(
            returned["model.layers.0.self_attn.o_proj.weight"],
            "2.self_attn.o_proj.weight",
        )
        self.assertEqual(returned["model.lm_head.weight"], "6.weight")
        self.assertEqual(returned["extra.weight"], "extra.weight")
        self.assertEqual(
            enc._pp_to_single_mapping["extra.weight"], "extra.weight"
        )

    def test_shared_layer_prefix_resolves_to_pivot(self):
        enc = self._make()
        # The pivot is registered before the MTP alias, so the first match wins.
        self.assertEqual(
            enc._shared_layer_prefix("mtp_reuse_transformer"), "model.layers.1"
        )

    def test_shared_layer_prefix_unknown_raises(self):
        enc = self._make()
        with self.assertRaises(ValueError):
            enc._shared_layer_prefix("does_not_exist")


if __name__ == "__main__":
    unittest.main()
