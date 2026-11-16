"""Numerical checks for the window-transfer rebuttal analysis (CPU only)."""
import copy
from pathlib import Path
import sys
import unittest
import shlex
import json

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import window_transfer_controls as controls
from make_window_replication_jobs import original_jobs, selected_windows

ROOT = Path(__file__).resolve().parents[2]


class TransferChecks(unittest.TestCase):
    def test_fixed_effects_match_explicit_dummy_regression(self):
        rng = np.random.default_rng(17)
        source = np.repeat(np.arange(5), 17)
        target = np.tile(np.arange(17), 5)
        keep = rng.random(len(source)) > .25
        source, target = source[keep], target[keep]
        matrix = rng.normal(size=(len(source), 4))
        matrix += source[:, None] - 3 * target[:, None]
        design = np.column_stack([np.ones(len(source)), np.eye(5)[source, 1:],
                                  np.eye(17)[target, 1:]])
        expected = matrix - design @ np.linalg.lstsq(design, matrix, rcond=None)[0]
        actual = controls.absorb_groups(matrix, [source, target])
        np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_repeated_seeds_do_not_increase_window_count(self):
        rows = [dict(run=f"seed{s}", width=.5, start=3., target_r=r,
                     loss_penalty=s + r) for s in (0, 1, 2) for r in (1., 2.)]
        averaged = controls.aggregate_seeds(rows)
        self.assertEqual(len(averaged), 2)
        self.assertEqual(len({r['run'] for r in averaged}), 1)
        np.testing.assert_allclose(controls.values(averaged, 'loss_penalty'), [2, 3])
        self.assertTrue(all(r['n_seeds'] == 3 for r in averaged))
        with self.assertRaises(ValueError):
            controls.aggregate_seeds(rows + [rows[0]])

    def test_frozen_reference_matches_existing_results(self):
        grouped, grid = controls.transition.load_runs(ROOT / "runs_win_fix_ce")
        rows = controls.read_csv(ROOT / "figures/window_transition_similarity.csv")
        controls.validate_old_rows(rows, grouped, grid)
        changed = copy.deepcopy(rows)
        changed[0]['loss_penalty'] += .01
        with self.assertRaises(ValueError):
            controls.validate_old_rows(changed, grouped, grid)

    def test_replication_matches_original_windows_and_chaotic_inventory(self):
        jobs = original_jobs()
        selected = selected_windows()
        self.assertEqual(len(selected), 16)
        summary = controls.read_csv(ROOT / "figures/window_transition_window_summary.csv")
        positive = {controls.window_key(r) for r in summary if r['chaos_fraction'] > 0}
        picked = {(w,s) for w,s,g in selected if g == 'chaotic'}
        self.assertEqual(picked, positive)
        for width, start, group in selected:
            self.assertIn((width, start), jobs)
            self.assertEqual(jobs[(width,start)]['--max_steps'], '160000')
            self.assertEqual(jobs[(width,start)]['--n_train_traj'], '8000')
            if group == 'regular':
                self.assertNotIn((width,start), positive)

    def test_subset_run_with_external_reference_reproduces_metrics(self):
        grouped, grid = controls.transition.load_runs(ROOT / "runs_win_fix_ce")
        original = controls.read_csv(ROOT / "figures/window_transition_similarity.csv")
        run = next(r for r in grouped[.125] if r['start'] == 3.75)
        selected = [r for r in original if r['run'] == run['name']]
        # Use the saved target Lyapunov estimates, preserving that independent
        # calculation exactly. Missing entries are in-support and never emitted.
        lam = np.zeros(len(grid))
        for row in selected:
            lam[np.argmin(abs(grid-row['target_r']))] = row['target_lambda']
        with np.load(ROOT / "figures/window_transition_process_cache.npz") as c:
            generated = controls.transition.make_pair_rows(
                {.125: [run]}, grid, c['process_counts'], c['processes'], c['process_js'],
                c['context_process_counts'], c['context_processes'], c['support_distance'],
                c['support_distance_euclidean'], lam, 64, reference_grouped=grouped)
        self.assertEqual(len(generated), len(selected))
        for a, b in zip(generated, selected):
            for key in a:
                if key != 'run':
                    self.assertAlmostEqual(a[key], b[key], places=10, msg=key)

    def test_common_targets_are_identical_and_outside_every_selected_window(self):
        reference, grid = controls.transition.load_runs(ROOT / 'runs_win_fix_ce')
        pairs = controls.aggregate_seeds(controls.read_csv(
            ROOT / 'figures/window_transition_similarity.csv'))
        common = controls.common_target_summary(pairs, reference, grid)
        self.assertEqual(len(common), 16)
        self.assertEqual(len(set(r['n_targets'] for r in common)), 1)
        for row in common:
            self.assertLessEqual(row['target_upper_exclusive'], row['start'])

    def test_generated_jobs_preserve_saved_hyperparameters(self):
        jobs = original_jobs()
        commands = (ROOT / 'jobs_rb_window_replicate.txt').read_text().splitlines()
        self.assertEqual(len(commands), 32)
        identities = set()
        for command in commands:
            tokens = shlex.split(command)
            flags = dict(zip(tokens[::2], tokens[1::2]))
            key=(float(flags['--width']),float(flags['--start']))
            params=json.loads((ROOT / jobs[key]['--out_dir'] / 'params.json').read_text())
            for flag, val in flags.items():
                k=flag[2:]
                if k in ('seed','out_dir'):
                    continue
                expected=params[k]
                if k in ('n_bins_in','n_bins_out'):
                    expected=expected or params['n_bins']
                actual=val if isinstance(expected,str) else float(val)
                self.assertEqual(actual, expected, msg=k)
            identity=(*key,int(flags['--seed']))
            self.assertNotIn(identity,identities)
            identities.add(identity)


if __name__ == "__main__":
    unittest.main()
