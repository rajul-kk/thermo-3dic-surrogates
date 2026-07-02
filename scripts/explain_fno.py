"""
FNO/WHNO spectral mode-importance XAI.

Compares the learned spectral weight distributions of two trained
checkpoints (typically FNO3d vs WHNO3d on the SAME geometry) to test
whether the Walsh-Hadamard basis concentrates more importance in
finer/high-sequency z-bands than Fourier does at low-frequency truncation
-- the mechanism-level evidence for the Gibbs-ringing argument, checked
against real trained weights rather than the synthetic step-function demo.

Usage
-----
python scripts/explain_fno.py \
    --model-a checkpoints/fno/geometry1_best.pt --label-a FNO \
    --model-b checkpoints/whno/geometry1_best.pt --label-b WHNO \
    --geometry geometry1 --model-a-type fno --model-b-type whno \
    --output results/fno_whno_mode_importance
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.geometry_builders import get_geometry_by_name
from src.fno.model import build_fno, build_cno_fno
from src.fno.whno import build_whno
from src.fno.interpret import (
    model_mode_importance, marginal_importance, plot_mode_importance_comparison,
)

_log = logging.getLogger(__name__)

_BUILDERS = {'fno': build_fno, 'cno-fno': build_cno_fno, 'whno': build_whno}


def _load_model(ckpt_path: Path, model_type: str, grid_shape, modes, channels, blocks):
    if model_type == 'cno-fno':
        model = build_cno_fno(grid_shape=grid_shape, ch=channels, n_fno_blocks=blocks)
    else:
        model = _BUILDERS[model_type](grid_shape=grid_shape, modes=modes, hidden_ch=channels, n_blocks=blocks)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # Accept either checkpoint key convention: FNOTrainer._save() uses
    # 'model_state', finetune_therm_fm.py's own saves use 'model'.
    sd = ckpt.get('model') or ckpt.get('model_state') or ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        _log.warning("Missing keys: %s", missing[:5])
    if unexpected:
        _log.warning("Unexpected keys: %s", unexpected[:5])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="FNO/WHNO spectral mode-importance XAI")
    parser.add_argument('--model-a', required=True, type=Path)
    parser.add_argument('--model-a-type', choices=['fno', 'cno-fno', 'whno'], default='fno')
    parser.add_argument('--label-a', default='Model A')
    parser.add_argument('--model-b', required=True, type=Path)
    parser.add_argument('--model-b-type', choices=['fno', 'cno-fno', 'whno'], default='whno')
    parser.add_argument('--label-b', default='Model B')
    parser.add_argument('--geometry', required=True)
    parser.add_argument('--modes', type=int, nargs=3, default=[16, 16, 12])
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--blocks', type=int, default=4)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%H:%M:%S')
    args.output.mkdir(parents=True, exist_ok=True)

    geometry = get_geometry_by_name(args.geometry)
    grid_shape = geometry.mesh_resolution

    model_a = _load_model(args.model_a, args.model_a_type, grid_shape, tuple(args.modes), args.channels, args.blocks)
    model_b = _load_model(args.model_b, args.model_b_type, grid_shape, tuple(args.modes), args.channels, args.blocks)

    imp_a = model_mode_importance(model_a)
    imp_b = model_mode_importance(model_b)
    n_blocks = min(len(imp_a), len(imp_b))
    _log.info("Comparing %d spectral blocks (%s: %d blocks, %s: %d blocks)",
              n_blocks, args.label_a, len(imp_a), args.label_b, len(imp_b))

    for block_idx in range(n_blocks):
        marg_a = marginal_importance(imp_a[block_idx], axis=2)   # z-axis
        marg_b = marginal_importance(imp_b[block_idx], axis=2)
        out_path = args.output / f'z_mode_importance_block{block_idx}.png'
        plot_mode_importance_comparison(
            {args.label_a: marg_a, args.label_b: marg_b},
            out_path,
            title=f'Block {block_idx}: z-axis spectral mode importance ({args.geometry})',
        )
        _log.info("Block %d: saved %s", block_idx, out_path)

    _log.info("Done. Compare the decay curves: if %s retains more energy in "
              "high-index (fine) z-bands than %s, that's evidence it's using "
              "its basis to represent the material-interface discontinuities "
              "that Fourier truncation smooths over.", args.label_b, args.label_a)


if __name__ == '__main__':
    main()
