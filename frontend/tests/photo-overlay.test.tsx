import {beforeAll, describe, expect, it} from 'vitest';
import {renderToStaticMarkup} from 'react-dom/server';
import {DEFAULT_OVERLAY_FEATURES, PhotoOverlay, featureVerdict, measureText, presetHidden, ratioMatches} from '../src/PhotoOverlay';
import {overlay19} from './fixtures';
import {Printable, Results, diagnosticPayload, type PhotoView} from '../src/Results';
import {validOverlay} from '../src/api';
import type {AnalysisResult, FaceOverlay, Item} from '../src/types';

// Synthetic coordinates only; no real photo or landmark data.
const overlay: FaceOverlay = {width: 640, height: 480, features: [
  {item_id: 'brow_length', label: '눈썹 길이', source: 'landmark', paths: [[[200, 150], [300, 140]], [[340, 140], [440, 150]]], value: .4567, unit: 'ratio'},
  {item_id: 'eye_tilt', label: '눈꼬리 기울기', source: 'landmark', paths: [[[220, 200], [290, 190]]], value: 3.2, unit: 'deg'},
  {item_id: 'forehead_height', label: '이마 높이', source: 'segmentation', paths: [[[320, 60], [320, 140]]]},
]};
const item = (id: string, status: Item['status'], label: string | null): Item => ({item_id: id, region: 'brow', status, kind: 'classification', label_id: null, label_name: label, confidence: .7, measurement: null, basis: '합성', hold_reasons: status === 'HELD' ? ['LOW_CONFIDENCE'] : []});
const items = [item('brow_length', 'CONFIRMED', '긴 편'), item('eye_tilt', 'HELD', '올라간 편')];
const result: AnalysisResult = {status: 'PARTIAL', items, passages: [], face_overlay: overlay};
const view = (extra: Partial<PhotoView> = {}): PhotoView => ({available: true, show: false, url: null, hidden: [], setShow: () => {}, toggle: () => {}, preset: () => {}, ...extra});
const overlayHtml = (extra: Partial<Parameters<typeof PhotoOverlay>[0]> = {}) => renderToStaticMarkup(<PhotoOverlay url="blob:test" overlay={overlay} items={items} hidden={[]} {...extra}/>);

beforeAll(() => {(globalThis as unknown as {window: object}).window = {};});

describe('overlay contract validation', () => {
  it('keeps measured paths inside the photo and drops malformed features or frames', () => {
    const raw = {width: 640, height: 480, features: [overlay.features[0],
      {item_id: 'outside', label: 'x', source: 'landmark', paths: [[[0, 0], [641, 10]]]},
      {item_id: 'nan', label: 'x', source: 'landmark', paths: [[[0, 0], [Number.NaN, 1]]]},
      {item_id: 'single', label: 'x', source: 'landmark', paths: [[[1, 1]]]},
      {item_id: 'model', label: 'x', source: 'cnn_attention', paths: [[[1, 1], [2, 2]]]},
      {item_id: 'value', label: 'x', source: 'segmentation', paths: [[[1, 1], [2, 2]]], value: 'big'}]};
    expect(validOverlay(raw)?.features.map(f => f.item_id)).toEqual(['brow_length']);
    expect(validOverlay({width: 0, height: 480, features: []})).toBeNull();
    expect(validOverlay({width: 640, height: 480})).toBeNull();
    const huge = {width: 10, height: 10, features: [{item_id: 'a', label: 'a', source: 'landmark', paths: [Array.from({length: 20001}, () => [1, 1])]}]};
    expect(validOverlay(huge)?.features).toEqual([]);
  });
});

describe('photo overlay rendering', () => {
  it('draws the exact source pixel coordinates in a viewBox of the photo size', () => {
    const html = overlayHtml();
    expect(html).toContain('viewBox="0 0 640 480"');
    expect(html).toContain('preserveAspectRatio="none"');
    expect(html).toContain('aspect-ratio:640 / 480');
    expect(html).toContain('points="200,150 300,140"');
    expect(html).toContain('points="340,140 440,150"');
    expect((html.match(/class="line"/g) || []).length).toBe(4);
    expect(html).toContain('정면 사진 위 측정선 3개');
  });
  it('hides deselected features and never gives a held item a label', () => {
    const html = overlayHtml({hidden: ['forehead_height'], onToggle: () => {}});
    expect(html).not.toContain('points="320,60 320,140"');
    expect(html).toContain('정면 사진 위 측정선 2개');
    expect(featureVerdict(overlay.features[0], items)).toBe('긴 편');
    expect(featureVerdict(overlay.features[1], items)).toBe('판정 보류');
    expect(featureVerdict(overlay.features[2], items)).toBe('');
    expect(html).not.toContain('올라간 편');
    expect(measureText(overlay.features[0])).toBe('비율 0.457');
    expect(measureText(overlay.features[1])).toBe('3.2°');
    expect(measureText(overlay.features[2])).toBe('');
  });
  it('shows the photo without invented lines when no overlay is available', () => {
    for (const missing of [null, undefined, {width: 640, height: 480, features: []}]) {
      const html = renderToStaticMarkup(<PhotoOverlay url="blob:test" overlay={missing} items={items} hidden={[]}/>);
      expect(html).toContain('<img');
      expect(html).not.toContain('<svg');
      expect(html).toContain('표시할 측정선이 없습니다');
    }
    expect(ratioMatches(overlay, 1280, 960)).toBe(true);
    expect(ratioMatches(overlay, 480, 640)).toBe(false);
  });
});

describe('opt-in photo on results and A4', () => {
  const page = (photo?: PhotoView) => renderToStaticMarkup(<Results result={result} photo={photo} onPreview={() => {}} onReset={() => {}}/>);
  it('is off by default and absent without an actual frontal photo', () => {
    expect(page()).not.toContain('사진과 측정선 표시');
    const off = page(view());
    expect(off).toContain('사진과 측정선 표시');
    expect(off).not.toMatch(/<input type="checkbox" checked/);
    expect(off).not.toContain('<img');
    expect(page(view({show: true, url: 'blob:x'}))).toContain('src="blob:x"');
  });
  it('puts the photo on the A4 sheet only when the user turned it on', () => {
    const sheet = (photo?: PhotoView) => renderToStaticMarkup(<Printable result={result} photo={photo} preview={false} onBack={() => {}}/>);
    expect(sheet(view())).not.toContain('<img');
    expect(sheet(view({show: true, url: null}))).not.toContain('<img');
    const on = sheet(view({show: true, url: 'blob:x', hidden: ['eye_tilt']}));
    expect(on).toContain('정면 사진과 측정선');
    expect(on).toContain('src="blob:x"');
    expect(on).not.toContain('points="220,200 290,190"');
    expect(on).not.toContain('type="checkbox"');
  });
  it('keeps the diagnostic file free of photos and overlay coordinates', () => {
    const text = JSON.stringify(diagnosticPayload(result, new Date(0)));
    expect(text).not.toContain('face_overlay');
    expect(text).not.toContain('paths');
    expect(text).not.toMatch(/blob:|data:image/);
  });
});

describe('realistic 19-feature overlay', () => {
  const real = overlay19(), all = real.features.map(f => f.item_id);
  const render = (hidden: string[], print = false) => renderToStaticMarkup(<PhotoOverlay url="blob:x" overlay={real} items={[]} hidden={hidden} onToggle={() => {}} onPreset={() => {}} print={print}/>);
  it('uses the backend key list and starts with only the four default lines', () => {
    expect(all).toHaveLength(19);
    expect(new Set(all).size).toBe(19);
    expect(validOverlay(JSON.parse(JSON.stringify(real)))?.features).toHaveLength(19);
    const hidden = presetHidden(real);
    expect(all.filter(id => !hidden.includes(id))).toEqual(['eye_tilt', 'brow_length', 'nose_width', 'mouth_width']);
    expect(DEFAULT_OVERLAY_FEATURES.every(id => all.includes(id))).toBe(true);
    expect(presetHidden(real, 'all')).toEqual([]);
    expect(presetHidden(real, 'none')).toEqual(all);
    expect(presetHidden({width: 1, height: 1, features: [real.features[0], real.features[1]]})).toEqual([]);
  });
  it('numbers and colours only the visible lines and keeps the other 15 one toggle away', () => {
    const html = render(presetHidden(real));
    expect((html.match(/class="overlay-feature/g) || []).length).toBe(4);
    expect((html.match(/class="swatch"/g) || []).length).toBe(4);
    expect(html).toContain('다른 측정선 15개');
    expect(html).toContain('전체 19개');
    expect(html).toContain('기본 4개');
    expect(html).toContain('3.42°');
    expect(html).not.toContain('degrees');
    expect(renderToStaticMarkup(<PhotoOverlay url="blob:x" overlay={real} items={[]} hidden={[]}/>)).toContain('차이 지수 0.0412');
    expect(html).not.toContain('log_ratio');
    const everything = render([]);
    expect((everything.match(/class="overlay-feature/g) || []).length).toBe(19);
    expect(everything).not.toContain('다른 측정선');
  });
  it('prints only the selected lines without the collapsed list', () => {
    const html = render(presetHidden(real), true);
    expect((html.match(/<li /g) || []).length).toBe(4);
    expect(html).not.toContain('다른 측정선');
    expect(html).not.toContain('type="checkbox"');
  });
});
