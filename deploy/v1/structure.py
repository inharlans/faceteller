"""Judging rules and passage selection of the reference structure (v10: plan C, confidence floor 0.6). Pure numpy; every
number comes from the JSON files in the config directory.

  CNN features          calibrated probability p^(1/T) renormalised; confidence = max; held below the floor (0.6);
                        an unjudgeable head above 0.5 = not judged
  numeric F07, F12      boundary score of sign*value against two cuts with sigma*1.45; held below the floor
  measurement features  F01 눈 크기, F09 코 길이, F10 코 높이, F17 턱 폭, and since 2026-11-17 F06 눈썹-눈 간격, F13 입술 두께,
                        F15 인중 길이 (cuts5_measures.json): same boundary score, held below the same floor
                        (0.6; it was 0.755 = within one sigma of a cut until 2026-11-17)
  VGGT formulas         F20 광대 돌출, F21 입 돌출: formula score (depth_models.json) against the team's cut
                        (cuts5_measures.json); not held
  selection (plan C)    candidates by G (geometric mean of the confidences of the features a passage uses), then
                        round-robin by region, unfavourable <= 2, total 5
Copied from experiments/feature17-20261109 (e2e_compare.judge_value / tempered / atom, e2e_v13.select_c), logic unchanged."""
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from facework.pilot.scoring import boundary_scores


class Config:
    def __init__(self, folder):
        folder = Path(folder); load = lambda n: json.loads((folder / n).read_text(encoding='utf8'))
        self.features = load('features.json'); self.cnn = load('cnn_calibration.json'); self.numeric = load('numeric_cuts.json'); self.nose = load('nose_measures.json'); self.eye = load('eye_measures.json')
        self.jaw = load('jaw_measures.json'); self.depth = load('depth_models.json'); self.cuts5 = load('cuts5_measures.json'); self.rules = load('selection_rules.json'); self.reps = load('literature_representatives.json')['items']
        self.check = load('check_models.json') if (folder / 'check_models.json').exists() else None
        self.phrases = load('region_phrases.json') if (folder / 'region_phrases.json').exists() else None
        f = self.features['features']; self.order = sorted(f); self.name = {k: v['name'] for k, v in f.items()}; self.region = {k: v['region'] for k, v in f.items()}; self.classes = {k: v['classes'] for k, v in f.items()}
        self.method = {k: v['method'] for k, v in f.items()}; self.cnn_parts = self.features['cnn_parts']; self.floor = float(self.rules['confidence_floor']['value']); self.hold = float(self.nose['hold_below_boundary_score'])
        self.k_sigma = float(self.nose['sigma_multiplier']); lim = self.rules['limits']; self.total = int(lim['total']); self.bad_max = int(lim['unfavourable_max'])
        self.cnn_features = [k for k in self.order if self.method[k] == 'CNN']; self.measured = [k for k in self.order if self.method[k] in ('정면 수치', 'VGGT 수치')]; self.formula = [k for k in self.order if self.method[k] == 'VGGT 맞춤식']
        self.measure_spec = {'F01': self.eye['features']['F01'], 'F09': self.nose['features']['F09'], 'F10': self.nose['features']['F10'], 'F17': self.jaw['features']['F17'], **{k: self.cuts5['features'][k] for k in ('F06', 'F13', 'F15')}}
        self.formula_spec = {k: self.cuts5['features'][k] for k in self.formula}  # cut and sigma of the VGGT formula scores (team rank cut)
        assert set(self.measured) == set(self.measure_spec) and set(self.formula) == set(self.cuts5['not_held'])

    def threshold(self, k): return None if k in self.formula else self.hold if k in self.measured else self.floor


def judge_value(v, cuts, sigma):
    """(class index, boundary score) with the Gaussian membership of the design document (5.5); one cut = two classes."""
    if len(cuts) == 1:
        p = float(ndtr((v - cuts[0]) / sigma)); i = int(v >= cuts[0]); return i, (p if i else 1 - p)
    b = boundary_scores(v, cuts[0], cuts[1], sigma); return ['low', 'medium', 'high'].index(b['label']), float(b['score'])


def tempered(P, t):
    Q = np.power(np.clip(np.asarray(P, float), 1e-9, 1), 1 / t); return Q / Q.sum(-1, keepdims=True)


def region_phrases(J, cfg):
    """One sentence per region from the confirmed judgements (config/region_phrases.json); regions without an entry are left out."""
    if cfg.phrases is None: return []
    out = []
    for region, feats in cfg.phrases['regions'].items():
        have = [f for f in feats if f in J]; hit = cfg.phrases['combinations'].get(region, {}).get('|'.join(f'{f}:{cfg.classes[f][J[f][0]]}' for f in have))
        if hit: out.append({'region': region, 'text': hit['text'], 'tone': hit['tone'], 'id': hit['id'], 'features': have, 'groups': hit['groups']})
    return out


def cut_measure(label, unit, value, cuts, classes, used=True):
    """Display record of a measured value: what it is, its unit, the cuts and the classes they separate."""
    return {'label': label, 'value': None if value is None else float(value), 'unit': unit, 'used_for_judgement': used, 'cuts': None if cuts is None else [float(c) for c in cuts], 'cut_classes': list(classes) if cuts is not None else None}


def calibrated(prob, unj, classes, temperature):
    K = len(classes); q = tempered(np.asarray(prob[:K]) / np.sum(prob[:K]), temperature)
    return {'class': classes[int(q.argmax())], 'index': int(q.argmax()), 'confidence': float(q.max()), 'probabilities': [round(float(x), 3) for x in q], 'unjudgeable': unj, 'classes': list(classes)}


def atom(text):
    """'F17 턱 폭:좁음' -> ('F17', '좁음')"""
    return text.split(' ')[0], text.split(':', 1)[1]


def matches(J, cfg):
    """Every representative whose conditions hold with the judgements J {feature: (class index, confidence)}, best G first."""
    out = []
    for r in cfg.reps:
        used = []; ok = True
        for cond in r['requirement']:
            hit = None
            for alt in cond.split(' 또는 '):
                atoms = [atom(a) for a in alt.split('+')]
                if all(f in J and cfg.classes[f][J[f][0]] == c for f, c in atoms): hit = atoms; break
            if hit is None: ok = False; break
            used += hit
        if not ok: continue
        fs = sorted({f for f, _ in used}); G = float(np.exp(np.mean([np.log(max(J[f][1], 1e-12)) for f in fs]))); out.append((-G, r['representative'], fs, used, r['valence'], r))
    return sorted(out, key=lambda c: c[:2])


def select_c(cand, cfg):
    """Plan C: in each round take the best remaining passage of every region not yet taken in that round."""
    sel, bad, left = [], 0, list(cand)
    while len(sel) < cfg.total:
        taken = set(); picked = False
        for c in list(left):
            g, pid, fs, used, val, r = c; regions = {cfg.region[f] for f in fs}
            if regions & taken or (val == '나쁨' and bad >= cfg.bad_max): continue
            sel.append(c); left.remove(c); taken |= regions; bad += val == '나쁨'; picked = True
            if len(sel) == cfg.total: break
        if not picked: break
    return sel


def passage_row(c, J, held, cfg):
    g, pid, fs, used, val, r = c
    return {'representative': pid, 'requirement': list(r['requirement']), 'valence': val, 'G': -g, 'regions': sorted({cfg.region[f] for f in fs}),
            'features': {f: {'class': cl, 'confidence': J[f][1], 'held': bool(held.get(f))} for f, cl in used}, 'translation': r.get('translation'), 'source': r.get('source'),
            **{k: r.get(k) for k in ('original_text', 'commentary', 'source_detail', 'annotation_status', 'translation_kind', 'split_of')},
            # what the books say for this condition: one line per kind of statement, with every book and the merged passages
            'lines': r.get('lines') or [], 'entry_count': r.get('entry_count') or 0, 'hidden_unfavourable': r.get('hidden_unfavourable') or 0, 'omitted_entries': r.get('omitted_entries') or 0}


def held_reason(k, cls, blocked_ids, cfg):
    using = [r['representative'] for r in cfg.reps if any(atom(a) == (k, cls) for cond in r['requirement'] for alt in cond.split(' 또는 ') for a in alt.split('+'))]
    return '이 칸을 조건으로 쓰는 조항이 없음' if not using else '보류 때문에 나오지 못한 조항이 있음' if blocked_ids & set(using) else '함께 필요한 다른 조건이 맞지 않음'
