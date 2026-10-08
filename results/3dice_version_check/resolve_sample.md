# Re-solve of a stratified sample under the pinned 3D-ICE (2026-10-08)

Question: do the shipped datasets equal what the pinned build (upstream 4953952, WSL ~/3d-ice-4.0, now the enforced default) produces? Method: first and last sorted file of each live dataset re-solved with `scripts/resolve_from_metadata.py` (no `--snap`; stored offsets used as-is) into a scratch directory; stored `temp` compared with the new `temp`. Rise = max(stored T) - ambient.

| dataset | n | max abs dT (K) | mean abs dT (K) | max dT / rise |
|---|---|---|---|---|
| 3d-ice-geometry7-material-sweep | 2 | 0 | 0 | 0 |
| 3d-ice-geometry7-pilot | 2 | 0 | 0 | 0 |
| 3d-ice-interface-multi | 2 | 0 | 0 | 0 |
| 3d-ice-interface-pilot-hi | 2 | 0 | 0 | 0 |
| 3d-ice-interface-pilot-median | 2 | 0 | 0 | 0 |
| 3d-ice-interface-pilot | 2 | 0 | 0 | 0 |
| 3d-ice-layout-geometry1 | 2 | 0 | 0 | 0 |
| 3d-ice-layout-geometry2a | 2 | 0 | 0 | 0 |
| 3d-ice-layout-geometry3 | 2 | 0 | 0 | 0 |
| 3d-ice-layout-geometry4 | 2 | 0 | 0 | 0 |
| 3d-ice-layout-geometry5 | 2 | 0 | 0 | 0 |
| 3d-ice-layout-geometry6 | 2 | 0 | 0 | 0 |
| 3d-ice-layout-geometry7 | 2 | 0 | 0 | 0 |
| 3d-ice-moving-geometry1 | 2 | 0 | 0 | 0 |
| 3d-ice-moving-geometry4-snap | 2 | 0 | 0 | 0 |

**30 files in 15 datasets: stored and re-solved temperature fields are bit-identical (max abs dT = 0 K, same node count, same coordinates).** The only solver change between the old build 39ed2a7 and the pinned build is `openblas_set_num_threads(1)`, consistent with this.

Provenance caveat: `resolve_from_metadata.py` does not write `ice_commit` into the output metadata, and the stored files have none either. The pinned commit 4953952a1ef6d38807ff307212a6f15e5b2ef935 was confirmed from `verify_ice_executable(DEFAULT_ICE_EXECUTABLE)`, which every solve passed through (it raises on an unpinned build).

## Not compared
- **data/3d-ice/geometry2a, geometry5, geometry6**: nonzero tsv_frac: resolve_from_metadata raises (cannot carry tsv_frac to new grid); not run (geometry5/6 attempted, raised)
- **3d-ice-leakage-pilot, 3d-ice-throttle-pilot**: feedback scenarios need their generator; resolve_from_metadata refuses
- **3d-ice-microchannel-pilot**: geometry 'geometry6_microchannel' unknown to get_geometry_by_name; not supported
- **3d-ice-moving-geometry4 (unsnapped), _archive_pre_snap_20260930/3d-ice-layout-geometry{4,5,6}**: iceforge guard (S001/S003 errors, off-grid edges with layout) refuses to solve; only bypass is ICEFORGE_GUARD=0, which the auto-mode permission classifier denied; needs user approval. Not run.
- **data/3d-ice/geometry{1,3,4} fixed-placement**: re-solved but INVALID comparison: files use pattern uniform/checkerboard/split_chiplet_b_hot (sub-block power maps), resolve_from_metadata rebuilds from block powers only, so power field differs (stored vs rebuilt power differs by up to 1e9 W/m3); differences of 1.6-13 K are a wrong rebuild, not a solver difference

Files compared are listed in `resolve_sample.json`.
