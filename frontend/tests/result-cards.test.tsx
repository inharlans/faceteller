import {beforeAll, describe, expect, it} from 'vitest';
import {renderToStaticMarkup} from 'react-dom/server';
import {Printable, Results, diagnosticPayload, heldSummary, holdKind, passageTitle} from '../src/Results';
import {Camera, Upload} from '../src/Inputs';
import {validateResult} from '../src/api';
import {percent, points, type AnalysisResult, type Item, type Passage} from '../src/types';

// Authored synthetic text only; no real results or editorial content.
const item = (id: string, region: string, label: string): Item => ({item_id: id, region, status: 'CONFIRMED', kind: 'classification', label_id: `${id}:x`, label_name: label, confidence: .8, measurement: null, basis: '합성 근거', hold_reasons: []});
const passage = (extra: Partial<Passage> = {}): Passage => ({passage_id: 'p1', rank: 1, G: .81, original_text: '眉如新月', translation: '눈썹이 초승달 같으면 이름이 널리 알려진다.', commentary: '합성 해설 COMMENTARY_ONLY', source: {title: '합성 문헌', chapter: '눈썹', locator: '제3조', url: 'https://example.org/a', annotation_status: 'ai_annotated_unreviewed'}, evidence_item_ids: ['brow_shape'], ...extra});
const result = (extra: Partial<AnalysisResult> = {}): AnalysisResult => ({status: 'COMPLETE', items: [item('brow_shape', 'brow', '완만한 곡선')], passages: [passage()], ...extra});
const reading = {title: '초승달 눈썹', meaning: '옛 문헌은 이런 눈썹을 이름이 널리 알려질 상으로 보았습니다.', valence: 'favorable' as const, valence_label: '길한 기록'};
const render = (r: AnalysisResult) => renderToStaticMarkup(<Results result={r} onPreview={() => {}} onReset={() => {}}/>);
const before = (html: string, a: string, b: string) => html.indexOf(a) >= 0 && html.indexOf(a) < html.indexOf(b);

beforeAll(() => {(globalThis as unknown as {window: object}).window = {};});

describe('result card reading order', () => {
  it('leads with the type title, tone tag, matched features and plain meaning; original text stays in the disclosure', () => {
    const html = render(result({passages: [passage({reading})]}));
    expect(html).toContain('<h3 class="card-title">초승달 눈썹</h3>');
    expect(html).toContain('data-valence="favorable">길한 기록</span>');
    expect(html).toMatch(/<li><span>눈썹 모양<\/span><b>완만한 곡선<\/b><\/li>/);
    expect(before(html, '초승달 눈썹</h3>', reading.meaning)).toBe(true);
    const disclosure = html.slice(html.indexOf('<details class="source-box">'));
    for (const text of ['眉如新月', '눈썹이 초승달 같으면', 'COMMENTARY_ONLY', '합성 문헌 · 눈썹 · 제3조', '정답 확률이 아닙니다']) expect(disclosure).toContain(text);
    expect(before(html, reading.meaning, '<details class="source-box">')).toBe(true);
  });

  it('without a server reading it shows the matched features as title and the translation as meaning, never an invented type', () => {
    const html = render(result());
    expect(html).toContain('<h3 class="card-title">눈썹 모양 완만한 곡선</h3>');
    expect(html).not.toContain('class="valence"');
    expect(html).toContain('<p class="meaning">눈썹이 초승달 같으면 이름이 널리 알려진다.</p>');
    expect(passageTitle(passage(), [])).toBe('문헌 기록');
  });

  it('shows one unified 관상 풀이 list and never a separate type card, even when legacy named_types arrive', () => {
    const named = {id: 't1', name: '봉황눈', name_hanja: '鳳眼', description: '눈꼬리가 길게 올라감', meaning: '옛 문헌은 귀하게 될 상으로 보았습니다.', valence: 'mixed' as const, evidence_item_ids: ['brow_shape'], source_url: 'javascript:alert(1)', original_text: '鳳眼'};
    const html = render(result({named_types: [named], passages: [passage({reading: {...reading, title: '일자미 · 일자 눈썹'}})]}));
    expect((html.match(/<h2[^>]*>/g) || []).length).toBe(1);
    expect(html).toContain('<h2 id="passages-title" class="section-title">관상 풀이</h2>');
    for (const gone of ['해당하는 관상 유형', '문헌에 남은 기록', 'type-card', '봉황눈', '鳳眼', 'javascript:']) expect(html).not.toContain(gone);
    expect((html.match(/class="passage-card"/g) || []).length).toBe(1);
    expect(html).toContain('<h3 class="card-title">일자미 · 일자 눈썹</h3>');
  });

  it('says each thing once: an identical translation or commentary is not repeated in the disclosure', () => {
    const same = '눈썹이 일자로 곧으면 성품이 곧다고 보았습니다.';
    const html = render(result({passages: [passage({reading: {...reading, meaning: same}, translation: same, commentary: ` ${same} `})]}));
    expect(html.split(same).length - 1).toBe(1);
    const distinct = render(result({passages: [passage({reading})]}));
    expect(distinct).toContain('<span class="label">옮김</span>눈썹이 초승달 같으면');
    const noReading = render(result({passages: [passage({commentary: '눈썹이 초승달 같으면 이름이 널리 알려진다.'})]}));
    expect(noReading.split('눈썹이 초승달 같으면 이름이 널리 알려진다.').length - 1).toBe(1);
  });

  it('does not show held items as matched features', () => {
    const held = {...item('brow_tail', 'brow', '올라감'), status: 'HELD' as const, hold_reasons: ['LOW_CONFIDENCE']};
    const html = render(result({items: [item('brow_shape', 'brow', '완만한 곡선'), held], passages: [passage({evidence_item_ids: ['brow_shape', 'brow_tail']})]}));
    expect(html.slice(html.indexOf('class="passage-card"'))).not.toContain('<b>올라감</b>');
  });
});

describe('response enrichment validation', () => {
  it('keeps the passage but drops a malformed reading and invalid named types', () => {
    const raw = {status: 'COMPLETE', items: [], passages: [{...passage(), reading: {title: '', meaning: 'x', valence: 'great'}}],
      named_types: [{id: 'ok', name: '유형', evidence_item_ids: []}, {id: 'bad', name: '유형', evidence_item_ids: [1]}, 'junk']};
    const checked = validateResult(JSON.parse(JSON.stringify(raw)));
    expect(checked.passages[0].reading).toBeUndefined();
    expect(checked.named_types?.map(t => t.id)).toEqual(['ok']);
    expect(validateResult({...raw, passages: [{...passage(), reading}]}).passages[0].reading).toEqual(reading);
  });
});

describe('score wording', () => {
  it('never rounds a value below 1 up to a perfect score', () => {
    expect(points(.9996)).toBe('99.9점 이상');
    expect(percent(.99999)).toBe('99.9% 이상');
    expect(points(1)).toBe('100.0점');
    expect(points(.864)).toBe('86.4점');
    expect(points(null)).toBe('산출 없음');
  });
});

describe('A4 sheet', () => {
  it('keeps original, translation and source with the reading title, and leaves commentary out', () => {
    const html = renderToStaticMarkup(<Printable result={result({passages: [passage({reading})], named_types: [{id: 't', name: '봉황눈', name_hanja: '鳳眼', valence: 'favorable', evidence_item_ids: []}]})} preview={false} onBack={() => {}}/>);
    for (const text of ['관상 풀이', '1. 초승달 눈썹 · 길한 기록', '眉如新月', '눈썹이 초승달 같으면', '합성 문헌 · 눈썹 · 제3조']) expect(html).toContain(text);
    for (const gone of ['봉황눈', '관상 유형', '문헌에 남은 기록']) expect(html).not.toContain(gone);
    expect(html).not.toContain('COMMENTARY_ONLY');
    expect(html).not.toContain(reading.meaning);
  });
});


describe('held items', () => {
  const held = (...reasons: string[]): Item => ({...item('x', 'brow', ''), status: 'HELD', label_name: null, hold_reasons: reasons});
  it('only suggests a retake for photo problems, never for unsupported items', () => {
    expect(holdKind(held('HAIRLINE_NOT_SUFFICIENTLY_VISIBLE'))).toBe('photo');
    expect(holdKind(held('LOW_OR_INVALID_CONFIDENCE'))).toBe('uncertain');
    expect(holdKind(held('UNKNOWN_FUTURE_CODE'))).toBe('uncertain');
    expect(holdKind(held('CONFIDENCE_NOT_CALIBRATED', 'FACE_TOO_SMALL'))).toBe('unsupported');
    const lines = heldSummary([held('CONFIDENCE_NOT_CALIBRATED'), held('CONFIDENCE_NOT_CALIBRATED')]);
    expect(lines).toEqual(['2개는 아직 판정할 자료가 부족한 항목이라 다시 찍어도 결과가 같습니다.']);
    expect(heldSummary([held('FACE_TOO_SMALL')])[0]).toContain('다시 찍으면');
  });
});

describe('diagnostic file', () => {
  it('holds scores and identities but no photos or passage text', () => {
    const r = {...result({passages: [passage({reading})]}), runtime: {deployment_sha256: 'abc'}} as AnalysisResult;
    const payload = diagnosticPayload(r, new Date('2026-10-15T00:00:00Z'));
    const text = JSON.stringify(payload);
    expect(payload).toMatchObject({contains_photos: false, saved_at: '2026-10-15T00:00:00.000Z', runtime: {deployment_sha256: 'abc'}});
    expect(payload.passages[0]).toMatchObject({passage_id: 'p1', G: .81, reading_title: '초승달 눈썹', valence: 'favorable'});
    expect(text).not.toContain('眉如新月');
    expect(text).not.toMatch(/data:image|blob:/);
  });
});

describe('camera and upload direction labels', () => {
  it('mirrors only the camera preview and names directions as the user sees them', () => {
    const camera = renderToStaticMarkup(<Camera onReady={() => {}} onError={() => {}} onBack={() => {}}/>);
    expect(camera).toContain('class="mirrored"');
    expect(camera).toContain('거울처럼 보이는 미리보기');
    expect(camera).toContain('정면 촬영');
    expect(camera).toMatch(/2 화면 오른쪽<\/li><li[^>]*>3 화면 왼쪽/);
    expect(camera).not.toMatch(/좌측|우측/);
    const upload = renderToStaticMarkup(<Upload onReady={() => {}} onBack={() => {}} onCamera={() => {}}/>);
    expect(upload).toContain('화면 왼쪽을 향한 사진');
    expect(upload).toContain('화면 오른쪽을 향한 사진');
    expect(upload).not.toContain('mirrored');
  });
});
