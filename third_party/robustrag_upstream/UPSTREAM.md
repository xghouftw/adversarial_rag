# RobustRAG upstream snapshot

This directory contains only the Python source needed by this project's
reproduction and extension experiments. The files are copied from:

- repository: https://github.com/inspire-group/RobustRAG
- commit: `9bc35b2fa5fa7d1088383b2789ec19a512d316b1`
- upstream commit date: 2024-09-15

`README.upstream.md` and `requirements.upstream.txt` preserve the corresponding
upstream files. The datasets are intentionally excluded from git and remain in
the checksum-verified shared data directory described in `ONBOARDING.md`.

The upstream repository did not contain a license file at this pinned commit.
This snapshot is retained for academic reproducibility and should not be
treated as relicensed project code.

The importable Python source is unchanged; a final newline was added to the
few upstream text files that lacked one. The committed-file checksums in
`reproduction/robustrag_setup.py` account for that formatting normalization.

Do not edit these files in place. `reproduction/robustrag_setup.py` verifies
their SHA-256 checksums, creates an ephemeral runner copy, and applies the
documented compatibility-only import deferrals to that copy.
