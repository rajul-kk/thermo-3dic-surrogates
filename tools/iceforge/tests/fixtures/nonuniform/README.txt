Original fixtures for 3D-ICE 4.0 non-uniform grids (non-uniform true). 10x10 mm chip, die of two floorplan
elements A (4000 um wide, 6 W) and B (6000 um wide, 4 W), a thin silicon layer under it, top heat sink.
  conforming/      every cell of die and layer is 500 x 500 um, so all meshes line up
  nonconforming/   same physics; die element B has 600 x 1000 um cells, A 500 x 500, the layer 500 x 500
recorded/ holds the files 3D-ICE 4.0 itself wrote for these inputs (tmap.txt = TOPD, tmapb.txt = BOT, and the
xyaxis_<instance>.txt cell geometry); tests copy them next to the .stk so no solver is needed.
Measured with `iceforge diff` (reference r = 4/8, tolerance 2 %): conforming AGREE (max |dT| 0.10 K),
nonconforming DISAGREE with up to 0.87 K at cells that do not line up with the other mesh.
