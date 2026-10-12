# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Unit coverage for the HyperBody encoder-decoder bridge MTP passthrough.

Pure dict round-trip tests (no GPU / no distributed init) for the fix that keeps
``mtp_startend_row_indices_all`` / ``mtp_hidden_inputs_mask_all`` alive across
the frontend -> bridge handoff:

  * ``_export_decoder_passthrough`` always writes the ``_hb_dec_*`` keys
    (``None`` when absent) so the exported dict's key set does not vary across
    micro-steps -- a varying key set breaks the PP framework's tensor-dict to
    tuple conversion;
  * ``_restore_decoder_passthrough`` renames the stashed values back onto the
    decoder dict under the ORIGINAL key names, and only writes non-``None``
    values (an absent key and a ``None`` value are equivalent for every
    consumer, and None-valued entries crash the PP tuple conversion);
  * the key names must stay in sync with the downstream consumers that read
    them by original name via ``dict_args.get(...)`` (see
    ``transformer/multi_token_prediction.py`` and ``models/gpt/gpt_embedding.py``).
"""

from paddlefleet.transformers.hyperbody.modeling import (
    _HB_DECODER_PASSTHROUGH_KEYS,
    _export_decoder_passthrough,
    _restore_decoder_passthrough,
)


class _FakeTensor:
    """Minimal tensor stand-in: only ``shape`` is touched (INFO log path)."""

    def __init__(self, shape=(2, 1, 8, 1)):
        self.shape = shape


def test_export_stashes_present_values_under_hb_dec_keys():
    """Present MTP fields are stashed under ``_hb_dec_*`` (identity preserved)."""
    row_indices = _FakeTensor((2, 1, 8, 1))
    hidden_mask = _FakeTensor((2, 1, 8))
    dict_args = {
        "mtp_startend_row_indices_all": row_indices,
        "mtp_hidden_inputs_mask_all": hidden_mask,
    }
    out = {"input_ids": _FakeTensor()}

    returned = _export_decoder_passthrough(out, dict_args)

    assert returned is out, "export must return the same out dict"
    assert out["_hb_dec_mtp_startend_row_indices_all"] is row_indices
    assert out["_hb_dec_mtp_hidden_inputs_mask_all"] is hidden_mask


def test_export_writes_none_for_absent_keys_keyset_stable():
    """Absent fields export as ``None`` -- key set is stable across micro-steps.

    Some micro-steps carry the MTP fields, others (text-only batches) do not;
    the exported dict must expose the same key set either way.
    """
    with_mtp = {
        "mtp_startend_row_indices_all": _FakeTensor(),
        "mtp_hidden_inputs_mask_all": _FakeTensor((2, 1, 8)),
    }
    without_mtp = {}

    out_a = _export_decoder_passthrough({"keep": 1}, with_mtp)
    out_b = _export_decoder_passthrough({"keep": 1}, without_mtp)

    assert set(out_a.keys()) == set(out_b.keys()), (
        "exported key set must not vary across micro-steps",
        sorted(out_a.keys()),
        sorted(out_b.keys()),
    )
    assert out_b["_hb_dec_mtp_startend_row_indices_all"] is None
    assert out_b["_hb_dec_mtp_hidden_inputs_mask_all"] is None


def test_export_does_not_touch_original_dict_args():
    """Export only reads from ``dict_args`` (never strips/overwrites its keys)."""
    row_indices = _FakeTensor()
    dict_args = {
        "mtp_startend_row_indices_all": row_indices,
        "mtp_hidden_inputs_mask_all": None,
        "labels": _FakeTensor(),
    }
    before = dict(dict_args)

    _export_decoder_passthrough({}, dict_args)

    assert dict_args == before
    # None values in the source are preserved as-is in the stash (not dropped).
    stash = _export_decoder_passthrough({}, dict_args)
    assert stash["_hb_dec_mtp_hidden_inputs_mask_all"] is None


def test_restore_round_trip_renames_back_to_original_keys():
    """Round-trip: stashed ``_hb_dec_*`` values reappear under original names."""
    row_indices = _FakeTensor((2, 1, 8, 1))
    hidden_mask = _FakeTensor((2, 1, 8))
    dict_args = _export_decoder_passthrough(
        {},
        {
            "mtp_startend_row_indices_all": row_indices,
            "mtp_hidden_inputs_mask_all": hidden_mask,
        },
    )
    decoder_args = {"input_ids": _FakeTensor(), "labels": _FakeTensor()}

    returned = _restore_decoder_passthrough(decoder_args, dict_args)

    assert returned is decoder_args, "restore must return the same dict"
    assert decoder_args["mtp_startend_row_indices_all"] is row_indices
    assert decoder_args["mtp_hidden_inputs_mask_all"] is hidden_mask
    # Pre-existing decoder keys are untouched.
    assert "input_ids" in decoder_args and "labels" in decoder_args


def test_restore_skips_none_and_absent_no_none_valued_entries():
    """``None`` / absent stash values must NOT be written to the decoder dict.

    An absent key and a ``None`` value are equivalent for every consumer
    (``dict_args.get``); leaving ``None`` out keeps the decoder dict free of
    None-valued entries (the PP tuple conversion crashes on those).
    """
    # One field None, one field absent entirely.
    dict_args = {"_hb_dec_mtp_startend_row_indices_all": None}
    decoder_args = {"input_ids": _FakeTensor()}

    _restore_decoder_passthrough(decoder_args, dict_args)

    assert "mtp_startend_row_indices_all" not in decoder_args
    assert "mtp_hidden_inputs_mask_all" not in decoder_args

    # Fully empty stash: still no None-valued entries.
    _restore_decoder_passthrough(decoder_args, {})
    assert "mtp_startend_row_indices_all" not in decoder_args
    assert "mtp_hidden_inputs_mask_all" not in decoder_args


def test_restore_leaves_stash_keys_out_of_decoder_dict():
    """The ``_hb_dec_*`` stash keys themselves must not leak into the decoder."""
    dict_args = _export_decoder_passthrough(
        {}, {"mtp_startend_row_indices_all": _FakeTensor()}
    )
    decoder_args = {}

    _restore_decoder_passthrough(decoder_args, dict_args)

    assert not any(k.startswith("_hb_dec_") for k in decoder_args), decoder_args


def test_passthrough_key_names_match_downstream_consumers():
    """Key names must stay in sync with downstream ``dict_args.get`` readers.

    ``transformer/multi_token_prediction.py`` and ``models/gpt/gpt_embedding.py``
    read these fields by original name:
        ``dict_args.get("mtp_startend_row_indices_all", None)``
        ``dict_args.get("mtp_hidden_inputs_mask_all", None)``
    A rename here would silently drop the MTP masks again (the bug this fix
    addresses), so pin the exact tuple.
    """
    assert _HB_DECODER_PASSTHROUGH_KEYS == (
        "mtp_startend_row_indices_all",
        "mtp_hidden_inputs_mask_all",
    ), _HB_DECODER_PASSTHROUGH_KEYS
    # The stash prefix used by both halves must stay consistent.
    row_indices = _FakeTensor()
    stash = _export_decoder_passthrough(
        {}, {"mtp_startend_row_indices_all": row_indices}
    )
    assert set(stash.keys()) == {
        f"_hb_dec_{k}" for k in _HB_DECODER_PASSTHROUGH_KEYS
    }, stash.keys()
