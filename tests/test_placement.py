"""
Tests for src/core/placement.py — per-scenario chiplet placement (docs/report.md §9.14-9.15).

The behaviours worth pinning down are the ones that were wrong on the first attempt:
stacked footprints must move as one chiplet, blocks must stay registered with the die they
sit on, and nothing may leave the package footprint.
"""
import numpy as np
import pytest

from src.core.geometry_builders import get_geometry_by_name
from src.core.placement import (associate_blocks_to_dies, lateral_chiplets, place_chiplets,
                                placement_summary, random_block_placement, random_placement)


def _inside(item, geom):
    return (item.x >= -1e-6 and item.y >= -1e-6
            and item.x + item.width <= geom.die_width + 1e-6
            and item.y + item.height <= geom.die_length + 1e-6)


class TestAssociation:
    def test_blocks_are_assigned_to_the_die_that_contains_them(self):
        geom = get_geometry_by_name('geometry4')
        groups = associate_blocks_to_dies(geom)
        assert set(groups) == {'chiplet_a', 'chiplet_b'}
        assert all(n.startswith('chipA_') for n in groups['chiplet_a'])
        assert all(n.startswith('chipB_') for n in groups['chiplet_b'])

    def test_geometry_without_footprints_groups_everything_under_empty_key(self):
        geom = get_geometry_by_name('geometry1')
        groups = associate_blocks_to_dies(geom)
        assert set(groups) == {''}
        assert len(groups['']) == len(geom.power_blocks)

    def test_association_is_by_containment_not_by_name(self):
        """Renaming a block must not change which die it belongs to."""
        geom = get_geometry_by_name('geometry4')
        before = associate_blocks_to_dies(geom)
        owner_of_first = next(d for d, names in before.items()
                              if geom.power_blocks[0].name in names)
        geom.power_blocks[0].name = 'totally_unrelated_name'
        after = associate_blocks_to_dies(geom)
        assert 'totally_unrelated_name' in after[owner_of_first]


class TestLateralGrouping:
    def test_stacked_footprints_on_different_layers_are_one_chiplet(self):
        """geometry6's hbm1_die1/hbm1_die2 share a rectangle: one physical stack."""
        geom = get_geometry_by_name('geometry6')
        chiplets = lateral_chiplets(geom)
        assert len(chiplets) < len(geom.die_footprints), (
            'stacked footprints were not grouped; this is the bug that made rejection '
            'sampling fail 200/200 times on geometry6')
        multi = [names for _, names in chiplets if len(names) > 1]
        assert multi, 'expected at least one multi-layer stack in geometry6'


class TestPlacement:
    def test_die_and_its_blocks_translate_together(self):
        geom = get_geometry_by_name('geometry4')
        groups = associate_blocks_to_dies(geom)
        moved = place_chiplets(geom, {'chiplet_a': (500.0, -250.0)})

        orig_b = {b.name: (b.x, b.y) for b in geom.power_blocks}
        for b in moved.power_blocks:
            ox, oy = orig_b[b.name]
            if b.name in groups['chiplet_a']:
                assert (b.x - ox, b.y - oy) == (500.0, -250.0)
            else:
                assert (b.x, b.y) == (ox, oy), 'a chiplet_b block moved with chiplet_a'

        dp = {d.name: d for d in moved.die_footprints}
        orig_dp = {d.name: d for d in geom.die_footprints}
        assert dp['chiplet_a'].x - orig_dp['chiplet_a'].x == 500.0
        assert dp['chiplet_b'].x == orig_dp['chiplet_b'].x

    def test_original_geometry_is_not_mutated(self):
        geom = get_geometry_by_name('geometry4')
        before = [(b.x, b.y) for b in geom.power_blocks]
        place_chiplets(geom, {'chiplet_a': (500.0, 0.0)})
        assert [(b.x, b.y) for b in geom.power_blocks] == before

    def test_leaving_the_footprint_is_rejected(self):
        geom = get_geometry_by_name('geometry4')
        with pytest.raises(ValueError, match='leaves the footprint'):
            place_chiplets(geom, {'chiplet_a': (99999.0, 0.0)})

    def test_unknown_key_is_rejected(self):
        geom = get_geometry_by_name('geometry4')
        with pytest.raises(ValueError, match='unknown die footprint or block'):
            place_chiplets(geom, {'no_such_die': (10.0, 10.0)})

    def test_single_block_can_be_moved_alone(self):
        """Independent-block mode: a key naming one block moves only that block."""
        geom = get_geometry_by_name('geometry1')
        target = geom.power_blocks[0].name
        moved = place_chiplets(geom, {target: (250.0, 250.0)})
        orig = {b.name: (b.x, b.y) for b in geom.power_blocks}
        for b in moved.power_blocks:
            ox, oy = orig[b.name]
            expected = (ox + 250.0, oy + 250.0) if b.name == target else (ox, oy)
            assert (b.x, b.y) == expected


class TestRandomPlacement:
    @pytest.mark.parametrize('name', ['geometry1', 'geometry4', 'geometry5', 'geometry6'])
    def test_sampled_placements_are_valid_and_in_bounds(self, name):
        geom = get_geometry_by_name(name)
        rng = np.random.default_rng(0)
        for _ in range(8):
            offsets = random_placement(geom, rng, max_shift_um=1500.0)
            moved = place_chiplets(geom, offsets)          # validates internally
            for item in list(moved.power_blocks) + list(moved.die_footprints or []):
                assert _inside(item, moved), f'{item.name} left the footprint'

    @pytest.mark.parametrize('name', ['geometry4', 'geometry5', 'geometry6'])
    def test_placements_actually_move_something(self, name):
        """A silent fallback to the nominal layout would defeat the whole point."""
        geom = get_geometry_by_name(name)
        rng = np.random.default_rng(1)
        moved_count = sum(
            any(dx or dy for dx, dy in random_placement(geom, rng, 1500.0).values())
            for _ in range(8))
        assert moved_count >= 6, f'{name}: only {moved_count}/8 placements moved'

    def test_stacked_layers_of_one_chiplet_share_an_offset(self):
        geom = get_geometry_by_name('geometry6')
        rng = np.random.default_rng(2)
        offsets = random_placement(geom, rng, max_shift_um=1000.0)
        for _, names in lateral_chiplets(geom):
            got = {offsets[n] for n in names if n in offsets}
            assert len(got) <= 1, f'stack {names} was sheared apart: {got}'

    def test_independent_block_placement_gives_each_block_its_own_position(self):
        geom = get_geometry_by_name('geometry1')
        rng = np.random.default_rng(3)
        offsets = random_block_placement(geom, rng, margin_um=300.0)
        assert set(offsets) == {b.name for b in geom.power_blocks}
        assert len({tuple(v) for v in offsets.values()}) > 1, 'blocks moved as a rigid group'
        moved = place_chiplets(geom, offsets)
        for b in moved.power_blocks:
            assert _inside(b, moved)

    def test_independent_blocks_do_not_overlap(self):
        geom = get_geometry_by_name('geometry1')
        rng = np.random.default_rng(4)
        for _ in range(5):
            moved = place_chiplets(geom, random_block_placement(geom, rng, margin_um=300.0))
            bs = moved.power_blocks
            for i in range(len(bs)):
                for j in range(i + 1, len(bs)):
                    a, b = bs[i], bs[j]
                    separated = (a.x + a.width <= b.x or b.x + b.width <= a.x or
                                 a.y + a.height <= b.y or b.y + b.height <= a.y)
                    assert separated, f'{a.name} overlaps {b.name}'


def test_placement_summary_is_readable():
    assert placement_summary(None) == 'nominal'
    assert 'chiplet_a' in placement_summary({'chiplet_a': (10.0, -20.0)})
