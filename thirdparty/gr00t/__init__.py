"""Isaac-GR00T N1.7 inference path, vendored from NVIDIA/Isaac-GR00T.

See VENDORED_FROM.md in this directory for the source commit, the file list,
and every local modification. Re-vendor with scripts/vendor_gr00t.sh.

Upstream's ``gr00t/__init__.py`` is deliberately NOT vendored: it only holds
``from_pretrained`` monkeypatches gated behind pytest / ``GROOT_*`` env flags,
none of which the inference path needs.
"""
