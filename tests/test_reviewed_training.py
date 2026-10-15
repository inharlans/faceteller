"""CPU tests for the reviewed relabel training candidate v2 (no GPU, no service)."""
import io
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.release import retrain_reviewed as rr
from facework import release_brows, release_context

HAS_DATA = (ROOT / 'data/release/training_registry.json').exists() and (ROOT / 'data/relabel-20261015/measurements.json').exists()


class SoftTargets(unittest.TestCase):
    def test_phi_soft_sums_and_limits(self):
        cuts = [-1.0, 1.0]
        for value in (-5, -1, 0, 1, 5):
            p = rr.phi_soft(value, cuts, 0.3)
            self.assertAlmostEqual(sum(p), 1.0, places=12); self.assertTrue(all(x >= 0 for x in p))
        self.assertGreater(rr.phi_soft(5, cuts, .3)[2], .999); self.assertGreater(rr.phi_soft(0, cuts, .3)[1], .99)
        at_cut = rr.phi_soft(1.0, cuts, .3); self.assertAlmostEqual(at_cut[1], at_cut[2], places=6)
        two = rr.phi_soft(0.2, [0.0], 0.1); self.assertEqual(len(two), 2); self.assertAlmostEqual(two[1], rr.phi(2.0), places=12)

    def test_sigma_is_single_capture_not_divided_by_repeats(self):
        a = rr.soft_entry(1.05, 3, [-1, 1], 0.5); b = rr.soft_entry(1.05, 8, [-1, 1], 0.5)
        self.assertEqual(a['soft'], b['soft']); self.assertEqual(a['sigma'], 0.5)
        self.assertLess(max(a['soft']), .9); self.assertEqual(a['hard'], 2)  # near boundary kept, soft

    def test_hard_class_matches_original_side_right(self):
        self.assertEqual(rr.hard_class(1.0, [-1.0, 1.0]), 2); self.assertEqual(rr.hard_class(-1.0, [-1.0, 1.0]), 1)
        self.assertEqual(rr.hard_class(-1.5, [-1.0, 1.0]), 0)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError): rr.phi_soft(float('nan'), [0], 1)
        with self.assertRaises(ValueError): rr.phi_soft(0, [0], 0)

    def test_range_sd(self):
        self.assertAlmostEqual(rr._range_sd({'value': 2.0, 'n': 3, 'relative_range': .1}), .2 / 1.693)
        self.assertIsNone(rr._range_sd({'value': 2.0, 'n': 1, 'relative_range': .1}))
        self.assertIsNone(rr._range_sd({'value': None, 'n': 3, 'relative_range': .1}))


class Calibration(unittest.TestCase):
    def test_rule(self):
        self.assertFalse(rr.calibration_rule([0, 1, 30], [0, 21, 226])['fit_temperature'])
        self.assertFalse(rr.calibration_rule([32, 2, 0], [248, 22, 0])['fit_temperature'])
        rule = rr.calibration_rule([5, 5, 6], [10, 0, 10])
        self.assertTrue(rule['fit_temperature']); self.assertEqual(rule['allowed_classes'], [0, 2])
        self.assertEqual(rr.calibration_rule([1, 26, 5], [23, 177, 66])['allowed_classes'], [1, 2])

    def _head(self, truth, supported=True, seed=0, train_support=(10, 10, 10)):
        rng = np.random.default_rng(seed); y = np.asarray(truth)
        logits = rng.normal(0, 1, (len(y), 3)); logits[np.arange(len(y)), np.maximum(y, 0)] += 4
        p = np.exp(logits); p /= p.sum(1, keepdims=True)
        d = {'classes': ['a', 'b', 'c'], 'supported': supported}
        return rr.calibrate_head(p, y, d, list(train_support), validation_ids=[f'v{i}' for i in range(len(y))], test_ids=['t1'])

    def test_degenerate_support_not_sharpened_and_threshold_is_float(self):
        out = self._head([2] * 30 + [1])
        self.assertFalse(out['calibrated']); self.assertIsNone(out['temperature']); self.assertEqual(out['allowed_confirmation_classes'], [])
        self.assertEqual(out['status'], 'insufficient_calibration_support')
        self.assertIsInstance(out['threshold'], float); self.assertGreaterEqual(out['threshold'], .5)
        self.assertEqual(out['threshold_selection']['status'], 'disabled_uncalibrated')

    def test_adequate_support_uses_existing_empirical_threshold_policy(self):
        from facework.release_thresholds import POLICY
        out = self._head([0] * 6 + [1] * 7 + [2] * 2)
        self.assertTrue(out['calibrated']); self.assertTrue(out['optimizer_success']); self.assertGreater(out['temperature'], 0)
        self.assertEqual(out['allowed_confirmation_classes'], ['a', 'b']); self.assertEqual(out['fit']['support'], [6, 7, 0])
        self.assertEqual(out['threshold_selection']['policy_version'], POLICY['version'])
        self.assertIn(out['threshold'], POLICY['candidates']); self.assertEqual(out['threshold_selection']['status'], 'selected_on_calibration')
        for c in out['threshold_selection']['candidates']:  # class c never counted as accepted
            self.assertLessEqual(c['accepted'], 13)

    def test_untrained_class_not_allowed(self):
        out = self._head([0] * 6 + [1] * 7, train_support=(10, 0, 10))
        self.assertTrue(out['calibrated']); self.assertEqual(out['allowed_confirmation_classes'], ['a'])

    def test_unsupported_head_never_calibrated(self):
        out = self._head([0] * 10 + [1] * 10, supported=False)
        self.assertFalse(out['calibrated']); self.assertEqual(out['status'], 'unsupported_head'); self.assertIsInstance(out['threshold'], float)

    def test_rejects_test_overlap(self):
        d = {'classes': ['a', 'b', 'c'], 'supported': True}
        with self.assertRaises(ValueError):
            rr.calibrate_head(np.full((10, 3), 1 / 3), [0] * 5 + [1] * 5, d, [5, 5, 5], validation_ids=[f'v{i}' for i in range(10)], test_ids=['v0'])

    def test_policy_counts_only_allowed_classes(self):
        out = rr.evaluate_policy([[.9, .1], [.2, .8], [.95, .05], [.6, .4]], [0, 1, 1, -1], .75, [0])
        self.assertEqual((out['n'], out['accepted'], out['wrong_confirmed']), (3, 2, 1))
        self.assertAlmostEqual(out['declared_cost_mean'], (1 + .25) / 3)


class ContextInput(unittest.TestCase):
    def test_contract(self):
        from facework.release_preprocess import INPUT_SIZES
        self.assertEqual(release_context.CONTEXT_VERSION, 'part-face-context-v1')
        self.assertEqual(set(release_context.CONTEXT_PARTS), {'eye', 'brow', 'nose', 'mouth', 'glabella', 'philtrum'})
        for part in release_context.CONTEXT_PARTS:
            h, w = INPUT_SIZES[part]; self.assertEqual(release_context.model_input_size(part), [h, 2 * w])
        for part in ('face', 'ear', 'forehead', 'under_eye', 'cheekbone', 'mole'):
            self.assertEqual(release_context.model_input_size(part), list(INPUT_SIZES[part]))
            with self.assertRaises(ValueError): release_context.compose_context(np.zeros((*INPUT_SIZES[part], 3), np.uint8), np.zeros((256, 224, 3), np.uint8), part)

    def test_tiles_letterbox_and_mirror(self):
        from facework.release_preprocess import letterbox
        from PIL import Image
        rng = np.random.default_rng(1)
        local = rng.integers(0, 255, (160, 256, 3), dtype=np.uint8); face = rng.integers(0, 255, (256, 224, 3), dtype=np.uint8)
        tile = release_context.compose_context(local, face, 'eye')
        self.assertEqual(tile.shape, (160, 512, 3)); self.assertEqual(tile.dtype, np.uint8)
        self.assertTrue((tile[:, :256] == local).all())
        expected = np.asarray(letterbox(Image.fromarray(face), (160, 256)))
        self.assertTrue((tile[:, 256:] == expected).all())
        self.assertTrue((tile[:, 256:300] == 127).all())  # isotropic letterbox padding (face 140 wide in 256)
        mirrored = release_context.compose_context(local, face, 'eye', mirror_face=True)
        self.assertTrue((mirrored[:, 256:] == expected[:, ::-1]).all()); self.assertTrue((mirrored[:, :256] == local).all())
        with self.assertRaises(ValueError): release_context.compose_context(local[:100], face, 'eye')
        with self.assertRaises(ValueError): release_context.compose_context(local.astype(np.float32), face, 'eye')

    def test_normalization_invariance(self):
        """Per-pixel ImageNet normalisation commutes with tiling: no cross-tile statistics."""
        import torch
        rng = np.random.default_rng(2)
        local = rng.integers(0, 255, (224, 320, 3), dtype=np.uint8); face = rng.integers(0, 255, (256, 224, 3), dtype=np.uint8)
        tile = torch.from_numpy(release_context.compose_context(local, face, 'brow')).permute(2, 0, 1).float() / 255
        mean = torch.tensor([.485, .456, .406])[:, None, None]; std = torch.tensor([.229, .224, .225])[:, None, None]
        whole = (tile - mean) / std
        left = (torch.from_numpy(local).permute(2, 0, 1).float() / 255 - mean) / std
        self.assertTrue(torch.allclose(whole[..., :320], left))

    def test_identical_tiles_receive_identical_augmentation(self):
        import torch
        for part in ('eye', 'nose', 'mouth', 'brow'):
            tile = torch.rand(4, 3, 32, 40); x = torch.cat([tile, tile], -1)
            out = rr.augment_input(x, torch.Generator().manual_seed(7), part, 40)
            self.assertTrue(torch.equal(out[..., :40], out[..., 40:]), part)
        # generator advances exactly as one tile's augmentation plus the shared flip draw
        g1 = torch.Generator().manual_seed(3); g2 = torch.Generator().manual_seed(3)
        from scripts.model_selection_train import augment
        rr.augment_input(torch.rand(2, 3, 32, 80), g1, 'eye', 40); augment(torch.rand(2, 3, 32, 40), g2, 'context')
        self.assertTrue(torch.equal(g1.get_state(), g2.get_state()))
        with self.assertRaises(ValueError): rr.augment_input(torch.rand(2, 3, 32, 70), torch.Generator(), 'eye', 40)

    def test_different_tiles_get_the_same_affine(self):
        import torch
        from scripts.model_selection_train import augment
        a = torch.rand(3, 3, 32, 40); b = torch.rand(3, 3, 32, 40)
        out = rr.augment_input(torch.cat([a, b], -1), torch.Generator().manual_seed(11), 'eye', 40)
        self.assertTrue(torch.equal(out[..., :40], augment(a, torch.Generator().manual_seed(11), 'context')))
        self.assertTrue(torch.equal(out[..., 40:], augment(b, torch.Generator().manual_seed(11), 'context')))
        # a single bright point moves to the same place in both tiles
        dot = torch.zeros(1, 3, 32, 40); dot[..., 20, 12] = 1
        out = rr.augment_input(torch.cat([dot, dot * .5 + .25], -1), torch.Generator().manual_seed(5), 'nose', 40)
        la = out[0, 0, :, :40]; lb = out[0, 0, :, 40:]
        self.assertEqual(int(la.argmax()), int(lb.argmax()))

    def test_augment_keeps_tiles_independent(self):
        import torch
        x = torch.rand(2, 3, 32, 80); y = x.clone(); y[..., 40:] = torch.rand(2, 3, 32, 40)
        a = rr.augment_input(x, torch.Generator().manual_seed(1), 'eye', 40)
        b = rr.augment_input(y, torch.Generator().manual_seed(1), 'eye', 40)
        self.assertTrue(torch.allclose(a[..., :40], b[..., :40])); self.assertFalse(torch.allclose(a[..., 40:], b[..., 40:]))


class TargetsAndTruth(unittest.TestCase):
    def table(self):
        side = lambda h: {'soft': [0., 0., 1.] if h == 2 else [1., 0., 0.], 'hard': h, 'tie': False}
        return {'definitions': {'eye_size': {'per_side': True, 'classes': ['s', 'm', 'l']},
                                'eyelid': {'per_side': False, 'classes': ['a', 'b', 'c']}},
                'subjects': {'A': {'split': 'test', 'targets': {
                    'eye_size': {'subject': {'soft': [0, .4, .6], 'hard': 1, 'tie': False}, 'sides': {'right': side(2), 'left': side(0)}},
                    'eyelid': {'subject': {'soft': [.5, .5, 0], 'hard': 0, 'tie': True}, 'sides': {}}}}}}

    def test_per_side_crop_targets_and_subject_truth(self):
        t = self.table()
        self.assertEqual(rr.crop_targets(t, 'A', ['eye_size', 'eyelid'], 'eye_right')[0], [0., 0., 1.])
        self.assertEqual(rr.crop_targets(t, 'A', ['eye_size', 'eyelid'], 'eye_left')[0], [1., 0., 0.])
        self.assertEqual(rr.crop_targets(t, 'A', ['eye_size', 'eyelid'], 'eye_left')[1], [.5, .5, 0])
        # subject truth = subject value (medium), not either side; tie excluded from hard evaluation
        self.assertEqual(rr.subject_truth(t, 'A', ['eye_size', 'eyelid']), [1, -1])

    def test_predict_aggregates_by_subject(self):
        import torch
        class Fixed(torch.nn.Module):
            def forward(self, x): return None, [torch.stack([x[:, 0, 0, 0], 1 - x[:, 0, 0, 0], torch.zeros(len(x))], 1) * 10]
        images = torch.zeros(3, 3, 8, 8, dtype=torch.uint8); images[0] = 255
        data = {'images': images, 'owners': torch.tensor([0, 0, 1]), 'subject_truth': torch.tensor([[1], [-1], [2]]), 'ids': ['A', 'B', 'C']}
        out = rr.predict([Fixed()], data, ['eye_size'], {'eye_size': ['s', 'm', 'l']}, 'cpu')['eye_size']
        self.assertEqual(out['truth'], [1, -1, -1]); self.assertEqual(out['available'], [True, True, False])
        self.assertAlmostEqual(sum(out['prob'][0]), 1, places=5)  # averaged over both crops of A
        self.assertAlmostEqual(out['prob'][0][0], out['prob'][0][1], places=3)

    def test_exposure_support_counts_people_per_side_class(self):
        owners = [0, 0, 1, 1, 2]
        votes = np.zeros((5, 1, 3)); votes[0, 0] = [0, 0, 1]; votes[1, 0] = [1, 0, 0]  # person 0: sides differ
        votes[2, 0] = [0, .6, .4]; votes[3, 0] = [0, .6, .4]
        people, crops = rr.exposure_support(owners, votes, ['k'], {'k': ['a', 'b', 'c']})
        self.assertEqual(people['k'], [1, 1, 1]); self.assertEqual(crops['k'], [1, 2, 1])  # person 2 has no target

    def test_soft_loss_ignores_missing_targets(self):
        import torch
        logits = [torch.zeros(2, 3, requires_grad=True)]
        votes = torch.tensor([[[0., 0., 1.]], [[0., 0., 0.]]])
        loss = rr.soft_loss(logits, votes, [torch.ones(3)])
        self.assertAlmostEqual(float(loss.detach()), math.log(3), places=5); loss.backward()
        self.assertTrue((logits[0].grad[1] == 0).all())
        self.assertIsNone(rr.soft_loss(logits, torch.zeros(2, 1, 3), [torch.ones(3)]))


class Identity(unittest.TestCase):
    def test_missing_or_changed_audit_fails(self):
        if not all((ROOT / rel).exists() for rel in rr.AUDITS): self.skipTest('audit artifacts absent')
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for rel in rr.AUDITS:
                (tmp / rel).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / rel, tmp / rel)
            rr.check_audits(tmp)
            first = next(iter(rr.AUDITS)); (tmp / first).unlink()
            with self.assertRaisesRegex(RuntimeError, 'missing'): rr.check_audits(tmp)
            shutil.copy2(ROOT / first, tmp / first); (tmp / first).write_bytes((tmp / first).read_bytes() + b' ')
            with self.assertRaisesRegex(RuntimeError, 'identity changed'): rr.check_audits(tmp)

    def test_never_writes_active_bundle(self):
        with self.assertRaisesRegex(RuntimeError, 'active bundle'): rr.dump(ROOT / 'models/release/x.json', {})
        with self.assertRaisesRegex(RuntimeError, 'active bundle'): rr.dump(ROOT / 'data/release/x.json', {})
        self.assertFalse((ROOT / 'models/release/x.json').exists())

    def test_protocol_detects_changed_targets(self):
        if not all((ROOT / rel).exists() for rel in rr.AUDITS): self.skipTest('audit artifacts absent')
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp)
            rr.dump(candidate / 'targets.json', {'x': 1})
            rr.dump(candidate / 'protocol.json', {'identities': rr.identities(), 'targets_sha256': rr.sha(candidate / 'targets.json')})
            rr.verify_protocol(candidate)
            rr.dump(candidate / 'targets.json', {'x': 2})
            with self.assertRaisesRegex(RuntimeError, 'targets.json changed'): rr.verify_protocol(candidate)


class Brows(unittest.TestCase):
    def synthetic(self, tail_drop=0, medial_thickness=8, fragment=None, size=600):
        from facework.morphometrics import BROW
        p = np.zeros((478, 2)); rng = np.random.default_rng(0); p[:] = [300, 300] + rng.normal(0, 1, (478, 2))
        p[33], p[133], p[362], p[263] = [200, 300], [260, 300], [340, 300], [400, 300]  # outer/inner, subject sides
        mask = np.zeros((size, size), np.uint8)
        for side, (x0, x1) in {'right': (180, 270), 'left': (330, 420)}.items():
            lateral_left = side == 'right'
            for x in range(x0, x1):
                frac = (x1 - x) / (x1 - x0) if lateral_left else (x - x0) / (x1 - x0)  # 1 at lateral end
                y = int(round(260 + tail_drop * frac)); half = int(round(4 + (medial_thickness - 4) * (1 - frac)))
                mask[y - half:y + 4, x] = 2
            for k, i in enumerate(BROW[side]['upper'] + BROW[side]['lower']): p[i] = [x0 + (x1 - x0) * (k % 5) / 4, 260]
        if fragment is not None: mask[fragment[1]:fragment[1] + 4, fragment[0]:fragment[0] + 6] = 2
        return mask, p

    def test_tail_sign_length_and_original_path(self):
        level_mask, points = self.synthetic(0, medial_thickness=4); drop_mask, _ = self.synthetic(12, medial_thickness=4)
        level = release_brows.measure_brows(level_mask, points); drop = release_brows.measure_brows(drop_mask, points)
        self.assertEqual(set(level['sides']), {'right', 'left'}); self.assertEqual(level['version'], 'visible-brow-mask-v3')
        self.assertAlmostEqual(level['sides']['right']['tail_angle_deg'], 0, delta=.5)
        self.assertLess(drop['sides']['right']['tail_angle_deg'], -3); self.assertLess(drop['sides']['left']['tail_angle_deg'], -3)
        self.assertAlmostEqual(level['sides']['right']['horizontal_eye_ratio'], 90 / 60, delta=.05)
        self.assertFalse(level['sides']['right']['visible_span_shorter_than_eye'])
        path = np.asarray(drop['sides']['right']['path_original']); self.assertGreater(path[0, 0], path[-1, 0])  # medial -> lateral
        ys = np.round(path[:, 1]).astype(int); xs = np.round(path[:, 0]).astype(int)
        self.assertGreater((drop_mask[ys, xs] > 0).mean(), .8)

    def test_thicker_medial_body_does_not_shorten(self):
        _, points = self.synthetic()
        thin = release_brows.measure_brows(self.synthetic(medial_thickness=4)[0], points)['sides']['right']
        thick = release_brows.measure_brows(self.synthetic(medial_thickness=14)[0], points)['sides']['right']
        self.assertAlmostEqual(thick['horizontal_eye_ratio'], thin['horizontal_eye_ratio'], delta=.02)
        self.assertAlmostEqual(thick['length_eye_ratio'] / thin['length_eye_ratio'], 1, delta=.05)

    def test_small_detached_fragment_ignored_large_fragment_rejected(self):
        _, points = self.synthetic()
        base = release_brows.measure_brows(self.synthetic()[0], points)['sides']['right']
        small = release_brows.measure_brows(self.synthetic(fragment=(165, 262))[0], points)['sides']['right']  # 24 px detached
        self.assertAlmostEqual(small['length_eye_ratio'], base['length_eye_ratio'], places=6)
        mask, _ = self.synthetic(); mask[240:256, 150:176] = 2  # large separate component inside ROI
        out = release_brows.measure_brows(mask, points)
        self.assertEqual(out['rejected'].get('right'), 'FRAGMENTED_MASK'); self.assertNotIn('right', out['sides'])

    def test_roi_and_image_truncation_rejected(self):
        mask, points = self.synthetic(); mask[255:262, 100:180] = 2  # hair continuing far outside the guide ROI
        self.assertEqual(release_brows.measure_brows(mask, points)['rejected'].get('right'), 'ROI_TRUNCATED')
        mask, points = self.synthetic(); points = points.copy(); points[:, 0] -= 180; mask = np.roll(mask, -180, axis=1); mask[:, -180:] = 0
        out = release_brows.measure_brows(mask, points)
        self.assertEqual(out['rejected'].get('right'), 'IMAGE_BOUNDARY')

    def test_frame_quality_gate_matches_kface_rule(self):
        from scripts.kface_labels import quality_ok
        rng = np.random.default_rng(3)
        for _ in range(500):
            pose = rng.uniform([-30, -15, -10], [20, 15, 10]); blend = rng.uniform(0, .7, 5)
            self.assertEqual(release_brows.frontal_quality_ok(pose, blend), bool(quality_ok(pose, blend, {})))
        mask, points = self.synthetic()
        out = release_brows.measure_brows(mask, points, pose=[-21.3, 0, 0], blend=[0, 0, 0, 0, 0])
        self.assertEqual(out['frame_rejected'], 'FRAME_POSE_OR_EXPRESSION'); self.assertEqual(out['sides'], {})

    def test_missing_hair_is_not_imputed(self):
        mask, points = self.synthetic(); mask[:] = 0
        out = release_brows.measure_brows(mask, points); self.assertEqual(out['sides'], {}); self.assertEqual(out['frame_rejected'], 'NO_VISIBLE_BROW_HAIR')

    def band(self, rows_tail, rows_body, body_fraction=.5):
        """Right brow: straight horizontal band; thicken only the medial body. Endpoints and centreline fixed."""
        mask, points = self.synthetic(medial_thickness=4); mask[:] = 0
        x0, x1 = 180, 270; split = int(x0 + (x1 - x0) * (1 - body_fraction))
        mask[260 - rows_tail // 2:260 + rows_tail - rows_tail // 2, x0:split] = 3
        mask[260 - rows_body // 2:260 + rows_body - rows_body // 2, split:x1] = 3
        mask[258:262, 330:420] = 2
        return mask, points

    def test_thin_tail_does_not_change_extent(self):
        """Review BROW-V2-01: one-pixel tail + thick body vs complete three-row band (same endpoints)."""
        tail, points = self.band(1, 21); full, _ = self.band(3, 3); one, _ = self.band(1, 1)
        a = release_brows.measure_brows(tail, points)['sides']['right']; b = release_brows.measure_brows(full, points)['sides']['right']
        c = release_brows.measure_brows(one, points)['sides']['right']  # one-row brow is accepted, not dropped
        self.assertAlmostEqual(a['horizontal_eye_ratio'], b['horizontal_eye_ratio'], delta=.02)
        self.assertAlmostEqual(a['length_eye_ratio'] / b['length_eye_ratio'], 1, delta=.03)
        self.assertAlmostEqual(c['horizontal_eye_ratio'], b['horizontal_eye_ratio'], delta=.02)
        self.assertGreater(a['thin_tail_columns']['low_x'] + a['thin_tail_columns']['high_x'], 10)
        self.assertEqual(a['path_probe_points'], 24 * release_brows.PATH_DENSE + 1)

    def test_bilateral_subject_requires_both_sides_stable(self):
        """Review BROW-V2-02: opposite angle changes cancel in the bilateral mean; subject must be withheld."""
        f = lambda angle: {'length_eye_ratio': 1.0, 'horizontal_eye_ratio': 1.0, 'tail_angle_deg': angle, 'visible_span_shorter_than_eye': False}
        frames = [{'left': f(-10), 'right': f(10)}, {'left': f(0), 'right': f(0)}, {'left': f(10), 'right': f(-10)}]
        limits = {k: {m: 1.0 for m in release_brows.METRICS} for k in ('subject', 'side')}
        out = release_brows.aggregate(frames, single_capture_sd=limits)
        self.assertEqual(out['subject']['tail_angle_deg']['sd'], 0); self.assertIsNone(out['subject']['tail_angle_deg']['median'])
        self.assertEqual(out['subject']['tail_angle_deg']['contributing_side_failed'], ['right', 'left'])
        self.assertIsNotNone(out['subject']['length_eye_ratio']['median'])
        one_side = [{'left': f(1), 'right': f(1)}, {'left': f(1.2), 'right': f(8)}, {'left': f(.9), 'right': f(-6)}]
        out = release_brows.aggregate(one_side, single_capture_sd=limits)
        self.assertIsNone(out['subject']['tail_angle_deg']['median']); self.assertIsNotNone(out['sides']['left']['tail_angle_deg']['median'])
        single = release_brows.aggregate([{'left': f(1), 'right': f(1)}])  # helper contract: n=1 never published
        self.assertIsNone(single['subject']['tail_angle_deg']['median']); self.assertTrue(single['subject']['tail_angle_deg']['insufficient_repeats'])

    def test_overlay_payload_uses_original_path(self):
        mask, points = self.synthetic(12, medial_thickness=4)
        result = release_brows.measure_brows(mask, points)
        payload = release_brows.overlay_payload(result, 'length_eye_ratio')
        self.assertEqual(payload['version'], 'visible-brow-mask-v3'); self.assertEqual(payload['source'], 'visible_brow_mask')
        self.assertEqual(payload['sides']['right']['path'], result['sides']['right']['path_original'])
        self.assertEqual(set(payload['sides']['right']), {'path', 'length_eye_ratio'})
        with self.assertRaises(ValueError): release_brows.overlay_payload(release_brows.measure_brows(mask, points, include_path=False))

    def test_aggregate_and_instability(self):
        f = lambda v: {'length_eye_ratio': v, 'horizontal_eye_ratio': v, 'tail_angle_deg': -v, 'visible_span_shorter_than_eye': v < 1}
        out = release_brows.aggregate([{'right': f(1.2), 'left': f(1.4)}, {'right': f(0.9), 'left': f(1.3)}, {'right': f(0.9)}])
        self.assertEqual(out['subject']['length_eye_ratio']['n'], 2); self.assertAlmostEqual(out['subject']['length_eye_ratio']['median'], (1.3 + 1.1) / 2)
        self.assertEqual(out['sides']['right']['length_eye_ratio']['n'], 3)
        self.assertEqual(out['visible_span_shorter_than_eye_frames'], {'true': 2, 'observed': 5})
        frames = [{'right': f(1.0), 'left': f(1.0)}, {'right': f(1.0), 'left': f(1.0)}, {'right': f(2.0), 'left': f(2.0)}]
        limits = {'subject': {'length_eye_ratio': .05}, 'side': {'length_eye_ratio': .05}}
        unstable = release_brows.aggregate(frames, single_capture_sd=limits)['subject']['length_eye_ratio']
        self.assertTrue(unstable['unstable']); self.assertIsNone(unstable['median'])

    @unittest.skipUnless(HAS_DATA, 'private local data absent')
    def test_main_qa_failure_cases_are_gated(self):
        cases = {('19083031', 'L2', 'left'): 'accepted_without_fragment', ('19100431', 'L1', 'right'): 'FRAGMENTED_MASK',
                 ('19100821', 'L1', 'right'): 'ROI_TRUNCATED', ('19100821', 'L2', 'right'): 'ROI_TRUNCATED',
                 ('19071022', 'L3', None): 'FRAME_POSE_OR_EXPRESSION'}
        for (sid, light, side), expected in cases.items():
            frame = [f for f in rr.brow_frames(sid) if f['light'] == light][0]
            if side is None: self.assertEqual(frame['frame_rejected'], expected)
            elif expected.startswith('accepted'):
                self.assertIn(side, frame['sides']); self.assertLess(frame['sides'][side]['length_eye_ratio'], 1.3)
            else: self.assertEqual(frame['rejected'].get(side), expected)
        # Rule on real masks: a bilateral subject value is published only if neither side failed.
        limits = {'subject': {m: 0.05 for m in release_brows.METRICS}, 'side': {m: 0.04 for m in release_brows.METRICS}}
        for sid in ('19083031', '19100821', '19062421'):
            frames = [f['sides'] for f in rr.brow_frames(sid) if not f['frame_rejected']]
            out = release_brows.aggregate(frames, single_capture_sd=limits)
            for m in release_brows.METRICS:
                s = out['subject'][m]
                failed = [side for side in ('right', 'left') if out['sides'][side][m]['median'] is None]
                self.assertEqual(s['contributing_side_failed'], failed)
                self.assertEqual(s['median'] is None, bool(failed or s['unstable'] or s['insufficient_repeats']))


@unittest.skipUnless(HAS_DATA, 'private local data absent')
class RealTargets(unittest.TestCase):
    def test_subset_targets_parity_and_semantics(self):
        ids = ['19062421', '19062431', '19100821']
        t = rr.build_targets(subjects=ids)
        d = t['definitions']; self.assertEqual(len(d), 41)
        original = json.loads((ROOT / 'data/release/landmark_labels.json').read_text(encoding='utf-8'))['features']
        self.assertEqual(d['eye_tilt']['cuts'], original['eye_tilt']['cuts'])
        self.assertEqual(d['eye_tilt']['sigma_subject'], original['eye_tilt']['median_within_subject_sd'])
        self.assertEqual(d['eye_tilt']['physical_centre'], 0.0); self.assertEqual(d['lip_balance']['physical_centre'], 1.0)
        self.assertEqual(d['brow_length']['mask_definition'], 'visible-brow-mask-v3'); self.assertEqual(d['brow_tail']['physical_centre'], 0.0)
        self.assertEqual(d['ear_length']['source'], 'extended_soft'); self.assertEqual(d['pigmented_marks']['source'], 'extended_hard')
        for key in rr.UNSUPPORTED: self.assertFalse(d[key]['supported'])
        e = t['subjects']['19062421']['targets']['eye_size']
        self.assertEqual(set(e['sides']), {'right', 'left'}); self.assertEqual(e['subject']['sigma'], d['eye_size']['sigma_subject'])
        self.assertIn('visible_span_shorter_than_eye', t['subjects']['19062421']['targets']['brow_length'])
        self.assertIsNone(t['subjects']['19100821']['targets']['brow_tail']['sides']['right'])  # gated side never imputed
        ear = t['subjects']['19062421']['targets']['ear_length']
        self.assertEqual(set(ear['sides']), {'C4', 'C10'})

    def test_registry_contract_unchanged(self):
        registry = json.loads((ROOT / 'data/release/training_registry.json').read_text(encoding='utf-8'))
        self.assertEqual(sum(len(p['features']) for p in registry['parts'].values()), 41)
        inputs = rr.protocol_inputs(registry)
        for part, spec in registry['parts'].items():
            self.assertEqual(inputs[part]['input_size'], spec['input_size']); self.assertEqual(inputs[part]['crops'], spec['crops'])
        self.assertEqual(inputs['eye']['model_input_size'], [160, 512]); self.assertNotIn('context_version', inputs['face'])


class TinyTrainingStep(unittest.TestCase):
    def test_resnet18_forward_backward_on_context_tile(self):
        import torch
        from facework.kface_training import PartNet
        model = PartNet('resnet18', 0, [3, 2], pretrained=False)
        x = rr.augment_input(torch.rand(2, 3, 160, 512), torch.Generator().manual_seed(0), 'eye', 256)
        _, out = model(x)
        votes = torch.tensor([[[.1, .2, .7], [1., 0, 0]], [[0, 0, 0], [0, 1., 0]]])
        loss = rr.soft_loss(out, votes, [torch.ones(3), torch.ones(2)]); loss.backward()
        self.assertTrue(math.isfinite(float(loss.detach())))


if __name__ == '__main__':
    unittest.main()
