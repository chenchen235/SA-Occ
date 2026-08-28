# SA-Occ+ minimal-update branch

This branch starts from the unmodified upstream SA-Occ commit
`6d837ef082b1d1b3871e32e81ad8b06ac0d4fb65`.

## Scope

The first functional update ports only the runtime path required by the final
SA-Occ+ experiment:

1. a standalone `sa-occ-r50-aux.py` configuration;
2. satellite input loading, satellite dataset metadata, and the `gen_sat.py` /
   `make_scenes_infos.py` data-construction utilities required by that
   configuration;
3. the SA-Occ detector changes needed to consume satellite features;
4. the dual-view satellite transformer and its dependencies.

The following are deliberately excluded from the first update because they do
not change model inference: validation-frame dumps, feature-gallery scripts,
debug probes, historical ablations, checkpoints, and generated outputs.

## Iteration order

1. Port the minimum runtime implementation and compile the CUDA extension.
2. Verify configuration construction and checkpoint-key compatibility.
3. Run a small validation smoke test, then full validation.
4. Add optional reproducibility/visualisation utilities in separate commits.

## Repository layout

- `SA-Occ-upstream/`: untouched shallow clone of the official repository.
- `SA-Occ-plus-update/`: clean Git worktree on branch `saocc-plus-update`.
- `SA-Occ/`: the original working directory; its existing local changes are
  intentionally not part of this branch.
