3dice_examples/   copies of 3D-ICE 4.0 example files (GPL-3.0-or-later, see the README there)
repro/            the die-edge artefact repro (notes/3dice_repro/edge_artifact_3dice.sh) as relative-path
                  models: aligned / misaligned / nolayout / snapped  (die 6000x6000 on a 10000 chip, 250 um cells)
repo_generated/   a hand-written copy of the style of stack.stk emitted by src/simulators/ice_simulator.py

LICENCES: iceforge itself is MIT (../../LICENSE). The two fixture directories derived from 3D-ICE source or example
files, 3dice_examples/ and nonuniform/repro_topsink_null/, are GPL-3.0-or-later and carry LICENSE-GPL-3.0 and
README-LICENSE.txt. The other fixtures are original to this repo (MIT); they only use the 3D-ICE input syntax.
