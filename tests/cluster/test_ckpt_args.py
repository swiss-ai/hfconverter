"""The stub-unpickler reader: cluster/ckpt_args.py."""

import os
import zipfile

import pytest
from gate_helpers import ForeignClass, write_checkpoint

from ckpt_args import is_opaque, read_checkpoint_args, resolve_model_optim_rng


class TestResolve:
    def test_root_with_tracker_resolves_to_the_named_iter(self, tmp_path):
        root = tmp_path / "c"
        root.mkdir()
        write_checkpoint(root, {"num_layers": 10}, iteration=1192)
        assert "iter_0001192" in resolve_model_optim_rng(root)

    def test_iter_dir_works_directly(self, tmp_path):
        root = tmp_path / "c"
        root.mkdir()
        write_checkpoint(root, {"num_layers": 10}, iteration=730)
        assert resolve_model_optim_rng(os.path.join(str(root), "iter_0000730"))

    def test_tracker_is_zero_padded_to_seven(self, tmp_path):
        """`7` must resolve to iter_0000007, not iter_7."""
        root = tmp_path / "c"
        root.mkdir()
        write_checkpoint(root, {"num_layers": 10}, iteration=7)
        assert "iter_0000007" in resolve_model_optim_rng(root)

    def test_torch_dist_checkpoint_is_rejected(self, tmp_path):
        """A torch_dist dir has metadata.json + *.distcp and NO mp_rank_* -- the message
        must say so rather than leaving the user staring at a bare listdir error."""
        d = tmp_path / "td" / "iter_0001192"
        d.mkdir(parents=True)
        (d / "metadata.json").write_text("{}")
        with pytest.raises(ValueError, match="no mp_rank_"):
            resolve_model_optim_rng(str(d))

    def test_tracker_naming_a_missing_iter_is_reported(self, tmp_path):
        root = tmp_path / "c"
        root.mkdir()
        write_checkpoint(root, {"num_layers": 10}, iteration=100)
        with open(str(root / "latest_checkpointed_iteration.txt"), "w") as handle:
            handle.write("999\n")
        with pytest.raises(ValueError, match="does not exist"):
            resolve_model_optim_rng(str(root))

    def test_missing_dir_is_reported(self, tmp_path):
        with pytest.raises(ValueError, match="not a directory"):
            resolve_model_optim_rng(str(tmp_path / "nope"))


class TestRead:
    def test_reads_the_stored_args(self, tmp_path):
        root = tmp_path / "c"
        root.mkdir()
        write_checkpoint(root, {"num_layers": 10, "hidden_size": 768, "swiglu": True})
        args = read_checkpoint_args(root)
        assert args["num_layers"] == 10
        assert args["hidden_size"] == 768
        assert args["swiglu"] is True

    def test_tensor_storages_are_never_loaded(self, tmp_path):
        """persistent_load returns None, so the storages decode to None instead of being
        read off disk. This is what makes the gate cheap on a multi-hundred-MB file."""
        root = tmp_path / "c"
        root.mkdir()
        write_checkpoint(root, {"num_layers": 10}, with_storage=True)
        args = read_checkpoint_args(root)  # would raise if a storage had to be materialized
        assert args["num_layers"] == 10

    def test_foreign_classes_decode_to_opaque_stubs(self, tmp_path):
        """A megatron enum / torch dtype cannot be reconstructed without its module. It must
        become an inert placeholder -- NOT abort the read of the other 780 keys."""
        root = tmp_path / "c"
        root.mkdir()
        write_checkpoint(root, {"num_layers": 10, "attention_backend": ForeignClass(5)})
        args = read_checkpoint_args(root)
        assert args["num_layers"] == 10, "one foreign value must not poison the whole read"
        assert is_opaque(args["attention_backend"])

    def test_plain_values_are_not_opaque(self, tmp_path):
        root = tmp_path / "c"
        root.mkdir()
        write_checkpoint(root, {"num_layers": 10, "normalization": "RMSNorm", "swiglu": True})
        args = read_checkpoint_args(root)
        assert not any(is_opaque(args[k]) for k in ("num_layers", "normalization", "swiglu"))

    def test_a_file_without_args_is_reported(self, tmp_path):
        rank = tmp_path / "c" / "iter_0000001" / "mp_rank_00_000"
        rank.mkdir(parents=True)
        import io
        import pickle

        buf = io.BytesIO()
        pickle.dump({"model": {}}, buf, protocol=2)
        with zipfile.ZipFile(str(rank / "model_optim_rng.pt"), "w") as zf:
            zf.writestr("archive/data.pkl", buf.getvalue())
        with pytest.raises(ValueError, match="no 'args' key"):
            read_checkpoint_args(str(tmp_path / "c" / "iter_0000001"))

    def test_a_non_torch_file_is_reported(self, tmp_path):
        rank = tmp_path / "c" / "iter_0000001" / "mp_rank_00_000"
        rank.mkdir(parents=True)
        with zipfile.ZipFile(str(rank / "model_optim_rng.pt"), "w") as zf:
            zf.writestr("something_else.txt", "not a checkpoint")
        with pytest.raises(ValueError, match="no data.pkl"):
            read_checkpoint_args(str(tmp_path / "c" / "iter_0000001"))
