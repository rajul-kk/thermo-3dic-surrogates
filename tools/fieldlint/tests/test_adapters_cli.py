import json

import numpy as np
import pytest

from fieldlint import AdapterError, Options, load_config, open_dataset, run
from fieldlint.adapters import norm_layout
from fieldlint.cli import main

h5py = pytest.importorskip('h5py')
sio = pytest.importorskip('scipy.io')


def _codes(rep, status='flagged'):
    return {r.code for r in rep.results if r.status == status}


def test_layout_letters():
    assert norm_layout('nchw') == 'NCYX' and norm_layout('NLHW') == 'NZYX'
    with pytest.raises(AdapterError):
        norm_layout('NHH')
    with pytest.raises(AdapterError):
        norm_layout('NQW')


def test_npz_stack_mapping(tmp_path, valid):
    src, u = valid
    np.savez(tmp_path / 'd.npz', temp=u, power=src, kmap=np.ones_like(u))
    cfg = {'u': {'key': 'temp', 'layout': 'NHW'}, 'source': {'key': 'power', 'layout': 'NHW'},
           'k': {'key': 'kmap', 'layout': 'NHW'}, 'units': 'K', 'ambient': 300.0}
    ds = open_dataset(tmp_path / 'd.npz', cfg)
    assert len(ds) == len(u)
    s = ds.get(3)
    assert np.array_equal(s.u, u[3]) and np.array_equal(s.source, src[3]) and s.k.shape == u[3].shape and s.ambient == 300.0
    rep = run(ds, rules=['F001', 'F002', 'F003', 'F005'])
    assert rep.exit_code == 0


def test_layout_letters_reorder_axes(tmp_path, valid):
    src, u = valid
    np.savez(tmp_path / 'd.npz', temp=np.moveaxis(u, 0, -1), power=np.moveaxis(src, 0, -1))   # (H, W, N)
    ds = open_dataset(tmp_path / 'd.npz', {'u': {'key': 'temp', 'layout': 'HWN'}, 'source': {'key': 'power', 'layout': 'HWN'}})
    assert np.array_equal(ds.get(5).u, u[5])
    # declaring the wrong H/W order is exactly what the orientation rule detects
    ds = open_dataset(tmp_path / 'd.npz', {'u': {'key': 'temp', 'layout': 'HWN'}, 'source': {'key': 'power', 'layout': 'WHN'}})
    assert 'F003' in _codes(run(ds, rules=['F003']))


def test_npy_stack_and_channel_layout(tmp_path, valid):
    src, u = valid
    np.save(tmp_path / 'u.npy', u)
    np.save(tmp_path / 'x.npy', np.stack([src, np.ones_like(src)], axis=1))                     # (N, C, H, W)
    cfg = {'u': {'file': 'u.npy', 'layout': 'NHW'}, 'source': {'file': 'x.npy', 'layout': 'NCHW', 'channel': 0}}
    ds = open_dataset(tmp_path, cfg)
    assert np.array_equal(ds.get(0).source, src[0])
    assert run(ds, rules=['F002', 'F003']).exit_code == 0
    with pytest.raises(AdapterError, match='channel'):
        open_dataset(tmp_path, {'u': cfg['u'], 'source': {'file': 'x.npy', 'layout': 'NCHW'}})


def test_mat_v5_and_v73_and_h5(tmp_path, valid):
    src, u = valid
    cfg = {'u': {'key': 'T', 'layout': 'NHW'}, 'source': {'key': 'P', 'layout': 'NHW'}}
    sio.savemat(tmp_path / 'v5.mat', {'T': u, 'P': src})
    with h5py.File(tmp_path / 'v73.mat', 'w') as f:                                             # HDF5-based, like MATLAB v7.3
        f['T'], f['P'] = u, src
    with h5py.File(tmp_path / 'd.h5', 'w') as f:
        g = f.create_group('fields')
        g['T'], g['P'] = u, src
    for name, c in (('v5.mat', cfg), ('v73.mat', cfg),
                    ('d.h5', {'u': {'key': 'fields/T', 'layout': 'NHW'}, 'source': {'key': 'fields/P', 'layout': 'NHW'}})):
        ds = open_dataset(tmp_path / name, c)
        assert np.array_equal(ds.get(7).u, u[7]), name
        assert run(ds, rules=['F002']).exit_code == 0, name


def test_ictherm_style_separate_files_with_optional_k_and_split(tmp_path, valid):
    src, u = valid
    n = len(u)
    x = np.zeros((n, 4, 1) + u.shape[1:], np.float32)
    x[:, 0, 0], x[:, 3, 0] = src, 5.0
    with h5py.File(tmp_path / 'input.mat', 'w') as f:
        f['data'] = x
    with h5py.File(tmp_path / 'output.mat', 'w') as f:
        f['data'] = u[:, None].astype(np.float32)
    ds = open_dataset(tmp_path, load_config('ictherm'))
    s = ds.get(0)
    assert s.u.shape == (24, 24) and s.k is not None and float(s.k.mean()) == 5.0
    splits = [ds.get(i).split for i in range(n)]
    tv = int(n * 0.8)
    assert splits.count('train') == int(tv * 0.9) and splits.count('test') == n - tv       # the benchmark's own index split
    # three-channel scope: k is optional and silently absent
    ds.close()
    with h5py.File(tmp_path / 'input.mat', 'w') as f:
        f['data'] = x[:, :3]
    ds3 = open_dataset(tmp_path, load_config('ictherm'))
    assert ds3.get(0).k is None
    ds3.close()


def test_ictherm_preset_detects_transposed_inputs(tmp_path, valid):
    src, u = valid
    x = np.zeros((len(u), 4, 1) + u.shape[1:], np.float32)
    x[:, 0, 0] = src.swapaxes(-1, -2)                                                          # stored transposed vs temperature
    x[:, 3, 0] = 1.0
    with h5py.File(tmp_path / 'input.mat', 'w') as f:
        f['data'] = x
    with h5py.File(tmp_path / 'output.mat', 'w') as f:
        f['data'] = u[:, None].astype(np.float32)
    cfg = load_config('ictherm')
    assert {'F002', 'F003'} <= _codes(run(open_dataset(tmp_path, cfg)))
    cfg['source']['transpose'] = True                                                          # the in-plane fix
    assert not ({'F002', 'F003'} & _codes(run(open_dataset(tmp_path, cfg))))


def test_per_file_directory_and_yaml_json_config(tmp_path, valid):
    src, u = valid
    d = tmp_path / 'ds'
    d.mkdir()
    for i in range(6):
        np.savez(d / f'case_{"train" if i < 4 else "test"}_{i}.npz', T=u[i], q=src[i])
    cfg = {'u': {'key': 'T', 'layout': 'HW'}, 'source': {'key': 'q', 'layout': 'HW'}, 'splits': 'from_filename'}
    ds = open_dataset(d, cfg)
    assert len(ds) == 6 and all(ds.get(i).split == ('train' if '_train_' in ds.get(i).id else 'test') for i in range(6))
    assert sorted(ds.get(i).split for i in range(6)) == ['test'] * 2 + ['train'] * 4
    (tmp_path / 'c.json').write_text(json.dumps(cfg))
    assert open_dataset(d, load_config(str(tmp_path / 'c.json'))).get(1).id.startswith('case_')
    pytest.importorskip('yaml')
    (tmp_path / 'c.yaml').write_text("u: {key: T, layout: HW}\nsource: {key: q, layout: HW}\n")
    assert len(open_dataset(d, load_config(str(tmp_path / 'c.yaml')))) == 6


def test_adapter_errors(tmp_path, valid):
    src, u = valid
    np.savez(tmp_path / 'd.npz', temp=u)
    with pytest.raises(AdapterError, match='not in'):
        open_dataset(tmp_path / 'd.npz', {'u': {'key': 'nope', 'layout': 'NHW'}})
    with pytest.raises(AdapterError, match='layout'):
        open_dataset(tmp_path / 'd.npz', {'u': {'key': 'temp', 'layout': 'NZHW'}})
    with pytest.raises(AdapterError, match='no such path'):
        open_dataset(tmp_path / 'missing.npz', {'u': 'temp'})
    with pytest.raises(AdapterError, match='presets'):
        load_config('nonexistent-preset')


def _write_3dice(path, u, p, scattered=True, name='geometryX_test_001', full=True):
    nz, ny, nx = u.shape
    zz, yy, xx = np.meshgrid(np.arange(nz) * 100.0 + 50, np.arange(ny) * 250.0 + 125, np.arange(nx) * 250.0 + 125, indexing='ij')
    coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], 1)
    temp, power = u.ravel().copy(), p.ravel().copy()
    order = np.random.default_rng(0).permutation(len(coords)) if scattered else np.arange(len(coords))
    if not full:
        order = order[:-3]
    np.savez(path / f'{name}.npz', coords=coords[order], temp=temp[order], power=power[order],
             metadata=np.array([{'t_ambient_kelvin': 318.15, 'geometry': 'x'}], dtype=object))


def test_3dice_adapter_scatters_to_grid_and_lints_clean(tmp_path, valid):
    src, u = valid
    yy, xx = np.mgrid[:12, :20]
    blob = 30 * np.exp(-((yy - 5) ** 2 + (xx - 6) ** 2) / 8.0)
    u3 = np.stack([319 + 0.2 * blob, 319 + blob, 319 + 0.1 * blob, 319 + 0.0 * blob])         # layer 1 is the active layer
    p3 = np.zeros_like(u3)
    p3[1, 4:7, 5:8] = 1.0
    _write_3dice(tmp_path, u3, p3)
    ds = open_dataset(tmp_path, load_config('3dice'))
    s = ds.get(0)
    assert s.u.shape == (4, 12, 20) and np.array_equal(s.u, u3) and np.array_equal(s.source, p3)   # coords -> grid, any order
    assert s.ambient == 318.15 and s.split == 'test' and s.spacing == (100.0, 250.0, 250.0)
    assert run(ds, rules=['F001', 'F002', 'F005']).exit_code == 0


def test_3dice_non_tensor_grid_reported_by_f001(tmp_path, valid):
    u3, p3 = np.full((3, 5, 5), 320.0), np.zeros((3, 5, 5))
    p3[1, 2, 2] = 1
    u3[1, 2, 2] = 330
    _write_3dice(tmp_path, u3, p3, full=False)
    rep = run(open_dataset(tmp_path, load_config('3dice')), rules=['F001'])
    assert rep.exit_code == 2 and 'tensor grid' in rep.results[0].findings[0].message


# ── CLI ──────────────────────────────────────────────────────────────────────────
def test_cli_text_json_rules_and_exit_codes(tmp_path, valid, capsys):
    src, u = valid
    np.savez(tmp_path / 'good.npz', temp=u, power=src)
    np.savez(tmp_path / 'bad.npz', temp=u, power=src.swapaxes(-1, -2))
    flags = ['--u', 'temp', '--source', 'power', '--layout', 'NHW']
    assert main([str(tmp_path / 'good.npz')] + flags + ['--units', 'K']) == 0
    out = capsys.readouterr().out
    assert 'F003 orientation' in out and 'exit code 0' in out
    assert main([str(tmp_path / 'bad.npz')] + flags + ['--rules', 'F003']) == 2
    out = capsys.readouterr().out
    assert 'F003' in out and 'F002' not in out and 'FLAGGED' in out
    assert main([str(tmp_path / 'bad.npz')] + flags + ['--json', '--skip', 'F006']) == 2
    rep = json.loads(capsys.readouterr().out)
    assert rep['exit_code'] == 2 and rep['worst_severity'] == 'error' and 'F006' not in {r['code'] for r in rep['rules']}
    f3 = next(r for r in rep['rules'] if r['code'] == 'F003')
    assert f3['metrics']['mean_corr']['transpose'] > f3['metrics']['mean_corr']['stored']
    out_file = tmp_path / 'r.json'
    assert main([str(tmp_path / 'good.npz')] + flags + ['--json', str(out_file), '--rules', 'F002']) == 0
    assert json.loads(out_file.read_text())['exit_code'] == 0


def test_cli_transpose_inputs_flag_and_head(tmp_path, valid, capsys):
    src, u = valid
    np.savez(tmp_path / 'bad.npz', temp=u, power=src.swapaxes(-1, -2))
    flags = ['--u', 'temp', '--source', 'power', '--layout', 'NHW', '--rules', 'F002,F003']
    assert main([str(tmp_path / 'bad.npz')] + flags) == 2
    assert main([str(tmp_path / 'bad.npz')] + flags + ['--transpose-inputs', '--head', '20']) == 0
    assert '(20 samples)' in capsys.readouterr().out


def test_cli_usage_errors_and_list_rules(tmp_path, capsys):
    assert main([str(tmp_path / 'missing.npz'), '--u', 'x', '--layout', 'NHW']) == 3
    assert 'no such path' in capsys.readouterr().err
    np.savez(tmp_path / 'd.npz', t=np.zeros((3, 4, 4)))
    assert main([str(tmp_path / 'd.npz'), '--u', 't', '--layout', 'NHW', '--rules', 'F999']) == 3
    assert main(['--list-rules']) == 0
    out = capsys.readouterr().out
    for c in ('F001', 'F002', 'F003', 'F004', 'F005', 'F006', 'F007'):
        assert c in out
    assert 'Does NOT apply' in out


def test_darcy_preset_reads_flattened_fields_and_beta_from_filename(tmp_path, darcy_like):
    k, u = darcy_like
    n = u.shape[-1]
    np.savez(tmp_path / 'darcy_beta0.5_n24.npz', X=k.reshape(len(k), -1), Y=u.reshape(len(u), -1))
    cfg = load_config('darcy')
    cfg['u']['reshape'] = cfg['k']['reshape'] = [n, n]
    ds = open_dataset(tmp_path / 'darcy_beta0.5_n24.npz', cfg)
    s = ds.get(2)
    assert np.array_equal(s.u, u[2]) and np.array_equal(s.k, k[2]) and np.all(s.source == 0.5)
    rep = run(ds, rules=['F001', 'F008', 'F009'])
    assert rep.exit_code == 0 or {r.code: r.status for r in rep.results}['F001'] == 'ok'
    assert {r.code: r.status for r in rep.results}['F008'] in ('ok', 'skipped')
    ds.close()


def test_time_dependent_flag_and_affine_in_config(tmp_path, darcy_like):
    k, u = darcy_like
    n = u.shape[-1]
    np.savez(tmp_path / 'darcy_f1.npz', X=(k.reshape(len(k), -1) - 1.0) / 9.0, Y=u.reshape(len(u), -1))
    cfg = {'format': 'npz', 'time_dependent': True, 'source_uniform': 1.0,
           'u': {'key': 'Y', 'layout': 'NHW', 'reshape': [n, n]},
           'k': {'key': 'X', 'layout': 'NHW', 'reshape': [n, n], 'affine': [9.0, 1.0]}}
    ds = open_dataset(tmp_path / 'darcy_f1.npz', cfg)
    assert ds.time_dependent and ds.head(5).time_dependent
    assert np.allclose(ds.get(1).k, k[1])
    r = {x.code: x for x in run(ds, rules=['F002', 'F003', 'F008', 'F009']).results}
    assert all(x.status == 'skipped' and 'time-dependent' in x.summary for x in r.values())
    ds.close()
