"""The closing remark of the report (research_report.summary): three sentences written by a local model through Ollama.

  first   what kind of person, ending "…상입니다."
  second  the favourable readings, joined into one flow
  third   what to mind ("…면 그 복이 오래 이어집니다." or "다만, …는 것이 좋습니다."); a closing wish when there is nothing to mind

The model writes the first two sentences only, from the favourable words. The third sentence is put together here from one
fixed wording per caution (config/summary_prompt.json care_clauses), so a caution cannot be turned around by the model.

No face part is named. The model is not shown the whole report: this module picks at most good_max favourable meanings and
care_max cautions from the readings of the confirmed judgements (config/region_phrases.json, taking turns between the
regions) and sends only those with the rules and worked examples of config/summary_prompt.json.
A remark is always returned: when the model cannot be reached, is switched off or answers with something unusable, a plain
remark is put together from the same picked words (source = "template").
Only the standard library is used; the model runs in its own process (ollama serve)."""
import json, re, time, urllib.request
from pathlib import Path

PART = re.compile(r'눈|코|입술|입꼬리|입[이과에을 ,]|턱|광대|미간|인중|이마|얼굴|생김')
batchim = lambda w: (ord(w[-1]) - 0xAC00) % 28 != 0 if '가' <= w[-1] <= '힣' else False
wa = lambda w: w + ('과' if batchim(w) else '와'); eul = lambda w: w + ('을' if batchim(w) else '를'); eun = lambda w: w + ('은' if batchim(w) else '는')


def pick(J, cfg, good_max, care_max, extra=None):
    """Favourable meanings and cautions of the confirmed judgements J, one region after another so that no region fills the list."""
    ph = cfg.phrases; good, care = [], []
    if ph is None: return good, care
    per = []
    for region, feats in ph['regions'].items():
        have = {f'{f}:{cfg.classes[f][J[f][0]]}' for f in feats if f in J}; rd = sorted((r for r in ph['readings'] if set(r['condition']) <= have), key=lambda r: -len(r['condition']))
        per.append(([m for r in rd for m in r['meanings']], [m for r in rd for m in (r['cautions'] or ([(extra or {}).get('|'.join(r['condition']))] if r['tendencies'] else [])) if m]))
    for out, idx, top in ((good, 0, good_max), (care, 1, care_max)):
        queues = [list(dict.fromkeys(p[idx])) for p in per]
        while len(out) < top and any(queues):
            for q in queues:
                while q and q[0] in out: q.pop(0)
                if q and len(out) < top: out.append(q.pop(0))
    return good, care


def care_sentence(care, c, turn=0):
    """The last sentence, from the fixed wording of each caution; the two endings take turns."""
    f = c['care_forms']
    if not care: return f['none']
    cl = [c['care_clauses'].get(x) for x in care]
    if None in cl: return f['unknown'].format(words=eun(care[0] if len(care) == 1 else wa(care[0]) + ' ' + care[1]))
    head = ''.join(x + '고 ' for x in cl[:-1]); last = cl[-1]; k = (ord(last[-1]) - 0xAC00) % 28  # stem endings: -는 / -(으)면 (no 으 after a vowel or ㄹ)
    return f['advice'].format(a=head + last + '는') if turn % 2 else f['condition'].format(a=head + last + ('면' if k in (0, 8) else '으면'))


def template_head(good):
    """First part from the picked words alone (no model)."""
    if not good: return '스스로 복을 지어 가는 상입니다.'
    first = f"{eul(good[0])} 지닌 상입니다." if len(good) == 1 else f"{wa(good[0])} {eul(good[1])} 함께 지닌 상입니다."; rest = good[2:5]
    return first + ('' if not rest else f" {eul(rest[0]) if len(rest) == 1 else wa(rest[0]) + ' ' + eul(', '.join(rest[1:]))} 더불어 누립니다.")


def template(good, care):
    """Plain remark from the picked words alone (no model, old form; kept for callers without the config)."""
    if not good: first = '삼갈 것을 삼가며 스스로 복을 지어 가는 상입니다.'; second = ''
    else:
        first = f"{eul(good[0])} 지닌 상입니다." if len(good) == 1 else f"{wa(good[0])} {eul(good[1])} 함께 지닌 상입니다."
        rest = good[2:5]; second = '' if not rest else f"{eul(rest[0]) if len(rest) == 1 else wa(rest[0]) + ' ' + eul(', '.join(rest[1:]))} 더불어 누립니다."
    third = f"다만, {eun(care[0] if len(care) == 1 else wa(care[0]) + ' ' + care[1])} 조심하는 것이 좋습니다." if care else '지금의 좋은 점을 지키면 그 복이 오래 이어집니다.'
    return ' '.join(x for x in (first, second, third) if x)


class Summariser:
    def __init__(self, config_dir, url, model, timeout=60.0):
        p = Path(config_dir) / 'summary_prompt.json'; self.cfg = json.loads(p.read_text(encoding='utf8')) if p.exists() else None
        self.url = (url or '').rstrip('/'); self.model = model or ''; self.timeout = float(timeout); self.enabled = bool(self.cfg and self.url and self.model)

    def _messages(self, good, care):
        c = self.cfg; ask = lambda g: c['user_template'].format(good=', '.join(g) or '없음'); msgs = [{'role': 'system', 'content': c['rules']}]
        for e in c['examples']: msgs += [{'role': 'user', 'content': ask(e['good'])}, {'role': 'assistant', 'content': e['text']}]
        return msgs + [{'role': 'user', 'content': ask(good)}]

    def usable(self, text):
        """Shape check of the model's answer; a failing answer is replaced by the template (it is not asked again)."""
        c = self.cfg['accept']; s = [x for x in re.split(r'(?<=[.!?])\s+', text) if x]
        return (c['sentences_min'] <= len(s) <= c['sentences_max'] and c['chars_min'] <= len(text) <= c['chars_max'] and s[0].endswith('상입니다.') and all(x.endswith('니다.') for x in s) and not PART.search(text) and not re.search(r'것입니다|겠습니다|다만|조심|삼가', text)
                and not re.search(r'[A-Za-z一-鿿"“”\n]', text))

    def warm(self):
        """Load the model into memory (the server calls this at start-up); False when it cannot be reached."""
        if not self.enabled: return False
        try: self._call([{'role': 'user', 'content': '준비'}], 1); return True
        except Exception: return False

    def _call(self, messages, num_predict):
        o = self.cfg['options']; body = json.dumps({'model': self.model, 'messages': messages, 'stream': False, 'keep_alive': o.get('keep_alive', '30m'),
                                                    'options': {'temperature': o['temperature'], 'seed': o['seed'], 'num_predict': num_predict, 'num_ctx': o['num_ctx']}}).encode()
        r = json.loads(urllib.request.urlopen(urllib.request.Request(self.url + '/api/chat', body, {'Content-Type': 'application/json'}), timeout=self.timeout).read())
        return ' '.join(r['message']['content'].split())

    def make(self, J, cfg):
        """-> (remark, info); info says where the remark came from, which words were given and how long the model took."""
        gm, cm = (self.cfg['good_max'], self.cfg['care_max']) if self.cfg else (6, 2); good, care = pick(J, cfg, gm, cm, (self.cfg or {}).get('extra_cautions')); info = {'good': good, 'care': care, 'model': self.model or None, 'source': 'template', 'seconds': 0.0}
        if not good and not care: return None, {**info, 'source': 'none'}  # nothing confirmed that the literature speaks about
        if self.cfg is None: return template(good, care), info
        tail = care_sentence(care, self.cfg, turn=sum(map(len, good + care)))  # written here, never by the model
        if self.enabled and good:
            t0 = time.perf_counter()
            try:
                text = self._call(self._messages(good, care), self.cfg['options']['num_predict']); info['seconds'] = round(time.perf_counter() - t0, 2)
                if self.usable(text): return f'{text} {tail}', {**info, 'source': 'model'}
                info['rejected'] = text
            except Exception as e: info['error'] = str(e)[:200]; info['seconds'] = round(time.perf_counter() - t0, 2)
        return f'{template_head(good)} {tail}', info
