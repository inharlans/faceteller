#!/usr/bin/env node
/* Standalone QA server; never contacts the integrated analysis service. Only
 * solid-color PNGs are uploaded. Responses use synthetic fixtures plus one
 * recorded real-inference JSON result, clearly labeled as HTTP-stub replay.
 * Run: node scripts/pilot/ui-check.cjs
 * Optional: PILOT_CHROMIUM_EXECUTABLE=/path/to/chrome
 */
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');
const { createHash } = require('node:crypto');
const { chromium } = require('playwright');
const { PNG } = require('pngjs');
const root = path.resolve(__dirname, '../..');
const out = path.join(root, 'outputs/pilot-20261014');
fs.mkdirSync(out, { recursive: true });
const screenshotDir = process.platform === 'win32' ? path.join(require('node:os').tmpdir(), 'faceteller-pilot-20261014') : '/tmp/faceteller-pilot-20261014';
fs.mkdirSync(screenshotDir, { recursive: true });
const report = {
  schema_version: 1, date: '2026-10-14', implementation: 'vanilla JavaScript functional experiment; React/TypeScript/esbuild absent',
  browser_plugin: 'Browser plugin not available', inference_executed: false, real_webcam_tested: false,
  real_api_tested: false, data: 'Synthetic fixtures plus recorded real-inference JSON replay; HTTP stub only; uploads are solid PNG; no face photos or raw coordinates',
  flow: '/pilot → triplet upload → NDJSON progress → result/held → A4 print preview → actual Chromium PDF',
  checks: [], screenshots: [], pdfs: [], console_errors: [], requests: []
};
let scenario = 'normal';
let browser, server, recordedResult;
const notice = '문헌 조회 결과이며 개인의 성격·지능·건강·운명을 판정하거나 예측하지 않습니다.';
const chinese = '此段文字專供版面測試使用，並非古籍原文，也不是對任何人的判斷。文字應完整保留，來源與段落編號不得省略。閱讀時請檢查換行、標點以及最後一個字。';
const korean = '이 문장은 인쇄 기능을 확인하기 위해 직접 만든 검증용 문장입니다. 실제 문헌의 번역이나 사람에 대한 판정이 아닙니다. 긴 한국어 문장과 중국어 원문 칸이 줄바꿈되어도 첫 글자부터 마지막 글자까지 모두 보존되는지 확인합니다. 출처와 조번호, 판정 근거도 함께 확인합니다.';
function fixture(multiplier = 1) {
  return {
    status: 'PARTIAL', literature_status: 'FOUND', notice,
    items: [
      { item_id: 'fixture-eye', region: '눈', status: 'CONFIRMED', label_id: 'f1', label_name: '검증용 형태', confidence: .82, measurement: null, basis: '합성 응답의 표시 규격만 검사함', hold_reasons: [] },
      { item_id: 'fixture-depth', region: '코', status: 'CONFIRMED', label_id: 'f2', label_name: '검증용 상대 돌출', confidence: null, measurement: { indicator: 'relative_projection', value: .125, unit: 'ratio' }, basis: '단위 없는 비율 표시 fixture', hold_reasons: [] },
      { item_id: 'fixture-held', region: '귀', status: 'HELD', label_id: null, label_name: null, confidence: null, measurement: null, basis: '촬영 방향 확인 불가 fixture', hold_reasons: ['방향 불충족 · 검증용 보류'] }
    ],
    passages: Array.from({ length: 5 }, (_, i) => ({ passage_id: `fixture-${i + 1}`, original_text: `${i + 1}號測試開始。${chinese.repeat(multiplier * 2)}${i + 1}號測試結束。`, translation: `${i + 1}번 직역 칸 시작. ${korean.repeat(multiplier)} ${i + 1}번 직역 칸 끝.`, commentary: `PRINT_EXCLUDED_COMMENTARY_${i + 1}: 상세 해설은 화면에서만 보입니다.`, source: { literature_title: '자체 합성 인쇄 검증 자료 (문헌 아님)', chapter: `검증편 ${i + 1}`, passage_no: `FIXTURE-${i + 1}` }, evidence_item_ids: ['fixture-eye', 'fixture-depth'] }))
  };
}
const normal = fixture();
const oversized = fixture(8);
const solid = new PNG({ width: 40, height: 40 });
for (let i = 0; i < solid.data.length; i += 4) { solid.data[i] = 50; solid.data[i + 1] = 110; solid.data[i + 2] = 145; solid.data[i + 3] = 255; }
const solidPNG = PNG.sync.write(solid);
const uploadFile = () => ({ name: 'synthetic-solid.png', mimeType: 'image/png', buffer: solidPNG });
const normalize = text => text.replace(/\s/gu, '');
const saveReport = () => fs.writeFileSync(path.join(out, 'ui_results.json'), JSON.stringify(report, null, 2) + '\n');
async function check(name, fn) {
  const start = Date.now();
  try { const details = await fn(); report.checks.push({ name, status: 'PASS', elapsed_ms: Date.now() - start, ...(details ? { details } : {}) }); }
  catch (error) { report.checks.push({ name, status: 'FAIL', error: error.message }); throw error; }
}
async function main() {
  const moduleSource = fs.readFileSync(path.join(root, 'web/pilot.js'), 'utf8');
  const { createCaptureState, readNDJSON, formatMeasurementScore } = await import(`data:text/javascript;base64,${Buffer.from(moduleSource).toString('base64')}`);
  await check('pure camera: continuous hold, reset, stale generations, 3 captures, one submit', () => {
    const state = createCaptureState(1000), good = { face: true, direction: true, size: true, brightness: true };
    assert.equal(state.submitOnce(), false);
    let token = state.token();
    assert.equal(state.update(good, 0, token).progress, 0);
    assert.equal(state.update(good, 500, token).progress, .5);
    assert.equal(state.update({ ...good, brightness: false }, 700, token).progress, 0);
    assert.equal(state.update(good, 800, token).progress, 0);
    assert.equal(state.update(good, 600, token).ignored, true);
    assert.equal(state.update(good, 1800, token).capture, true);
    assert.equal(state.update(good, 1801, token).ignored, true);
    assert.equal(state.commit(token), true); assert.equal(state.commit(token), false);
    assert.equal(state.update(good, 2000, token).ignored, true);
    for (const direction of ['left_oblique', 'right_oblique']) {
      token = state.token(); assert.equal(token.direction, direction);
      assert.equal(state.update(good, 0, token).capture, false);
      assert.equal(state.update(good, 1000, token).capture, true);
      assert.equal(state.commit(token), true);
    }
    assert.equal(state.submitOnce(), true); assert.equal(state.submitOnce(), false);
    state.reset(); assert.equal(state.update(good, 5000, token).ignored, true); assert.equal(state.token().direction, 'front');
    for (const condition of Object.keys(good)) {
      state.reset(); token = state.token(); state.update(good, 0, token);
      assert.equal(state.update({ ...good, [condition]: false }, 999, token).progress, 0);
      assert.equal(state.update(good, 1000, token).capture, false);
    }
    return { conditions: Object.keys(good), hardware_used: false, fixtures: 'condition booleans only' };
  });
  await check('NDJSON pure: split UTF-8, missing newline, malformed and interrupted streams', async () => {
    const bytes = new TextEncoder().encode(JSON.stringify({ type: 'progress', stage: '얼굴 정렬' }) + '\r\n' + JSON.stringify({ type: 'result', data: normal }));
    const stages = [];
    const stream = new ReadableStream({ start(c) { for (let i = 0; i < bytes.length; i += 7) c.enqueue(bytes.slice(i, i + 7)); c.close(); } });
    const parsed = await readNDJSON(new Response(stream, { headers: { 'Content-Type': 'application/x-ndjson' } }), s => stages.push(s));
    assert.deepEqual(parsed, normal); assert.deepEqual(stages, ['얼굴 정렬']);
    for (const body of ['{"type":"progress","stage":"깊이 분석"}\n', '{broken}\n', JSON.stringify({ type: 'result', data: normal }) + '\n' + JSON.stringify({ type: 'progress', stage: 'late' })]) await assert.rejects(readNDJSON(new Response(body, { headers: { 'Content-Type': 'application/x-ndjson' } }), () => {}));
  });
  const fontsRoot = process.platform === 'win32' ? 'C:/Windows/Fonts' : '/mnt/c/Windows/Fonts';
  server = http.createServer(async (req, res) => {
    if (req.method === 'GET') {
      const staticFiles = { '/pilot': ['web/pilot.html', 'text/html'], '/pilot.js': ['web/pilot.js', 'text/javascript'], '/pilot.css': ['web/pilot.css', 'text/css'] };
      if (req.url === '/favicon.ico') { res.writeHead(204); return res.end(); }
      if (req.url === '/__qa-font-ko') { res.setHeader('Content-Type', 'font/ttf'); return fs.createReadStream(path.join(fontsRoot, 'malgun.ttf')).pipe(res); }
      if (req.url === '/__qa-font-zh') { res.setHeader('Content-Type', 'font/collection'); return fs.createReadStream(path.join(fontsRoot, 'msyh.ttc')).pipe(res); }
      if (staticFiles[req.url]) { const [file, type] = staticFiles[req.url]; res.setHeader('Content-Type', type); return res.end(fs.readFileSync(path.join(root, file))); }
      res.writeHead(404); return res.end();
    }
    if (req.url !== '/api/v1/analysis' || req.method !== 'POST') { res.writeHead(404); return res.end(); }
    let body = Buffer.alloc(0);
    for await (const chunk of req) body = Buffer.concat([body, chunk]);
    const text = body.toString('latin1');
    const fields = [...text.matchAll(/Content-Disposition: form-data; name="([^"]+)"/g)].map(m => m[1]);
    report.requests.push({ scenario, method: req.method, path: req.url, fields, only_solid_fixture: body.includes(solidPNG), input_mode: /name="input_mode"\r\n\r\nupload/.test(text) ? 'upload' : 'unknown' });
    body = null;
    if (scenario === 'http-error') { res.writeHead(422, { 'Content-Type': 'application/json' }); return res.end(JSON.stringify({ code: 'FACE_NOT_FOUND', message: '얼굴 미검출 fixture', direction: 'left_oblique' })); }
    res.writeHead(200, { 'Content-Type': 'application/x-ndjson', 'Cache-Control': 'no-store' });
    for (const stage of ['얼굴 정렬', '부위 추출', '형태 분류', '깊이 분석', '문헌 선정']) { res.write(JSON.stringify({ type: 'progress', stage }) + '\n'); await new Promise(r => setTimeout(r, 45)); }
    if (scenario === 'cancel') { const timer = setTimeout(() => res.end(), 5000); timer.unref(); return; }
    if (scenario === 'interrupted') return res.end();
    if (scenario === 'malformed') return res.end('{broken}\n');
    if (scenario === 'stream-error') return res.end(JSON.stringify({ type: 'error', code: 'PREPROCESS_FAILED', message: '전처리 실패 fixture', direction: 'right_oblique' }) + '\n');
    let data = scenario === 'recorded-inference' ? recordedResult : scenario === 'overflow' ? oversized : structuredClone(normal);
    if (scenario === 'empty') { data.passages = []; data.literature_status = 'NONE'; }
    if (scenario === 'reinput') { data.items = data.items.filter(i => i.status === 'HELD'); data.passages = []; data.status = 'REINPUT_REQUIRED'; data.literature_status = 'NOT_QUERIED'; }
    if (scenario === 'xss') { data.passages[0].original_text = '<img src=x onerror="window.xss=true">'; }
    const bytes = Buffer.from(JSON.stringify({ type: 'result', data }));
    // A non-newline-terminated final event and small writes exercise streaming decode.
    for (let i = 0; i < bytes.length; i += 127) res.write(bytes.subarray(i, i + 127));
    res.end();
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const url = `http://127.0.0.1:${server.address().port}/pilot`;
  report.environment = { url, command: process.platform === 'win32' ? 'node.exe scripts/pilot/ui-check.cjs' : 'node scripts/pilot/ui-check.cjs', platform: process.platform, viewports: [{ width: 1280, height: 900 }, { width: 390, height: 844 }], fonts: 'Installed Windows Malgun Gothic + Microsoft YaHei served only by fixture QA server', linux_browser_attempt: 'Downloaded Chromium 151; launch unavailable: libnspr4.so missing. Used installed Windows Chromium with node.exe.' };
  const windowsChrome = process.platform === 'win32' ? [path.join(process.env.LOCALAPPDATA, 'ms-playwright/chromium-1228/chrome-win64/chrome.exe'), 'C:/Program Files/Google/Chrome/Application/chrome.exe'].find(p => fs.existsSync(p)) : undefined;
  const executablePath = process.env.PILOT_CHROMIUM_EXECUTABLE || windowsChrome;
  browser = await chromium.launch({ headless: true, ...(executablePath ? { executablePath } : {}), args: ['--no-sandbox'] });
  report.environment.chromium_version = browser.version();
  const context = await browser.newContext({ viewport: report.environment.viewports[0] });
  await context.addInitScript(() => {
    window.FACETELLER_PILOT_ADAPTERS = { fixture: true };
    window.__qa = { created: 0, revoked: 0, cameraRequests: 0, storageWrites: 0, printed: false, stages: [] };
    const create = URL.createObjectURL.bind(URL), revoke = URL.revokeObjectURL.bind(URL);
    URL.createObjectURL = (...args) => { window.__qa.created++; return create(...args); };
    URL.revokeObjectURL = (...args) => { window.__qa.revoked++; return revoke(...args); };
    Storage.prototype.setItem = () => { window.__qa.storageWrites++; throw new Error('Storage writes forbidden during QA'); };
    Object.defineProperty(navigator.mediaDevices, 'getUserMedia', { value: async () => { window.__qa.cameraRequests++; throw new DOMException('Fixture permission denial', 'NotAllowedError'); } });
    window.print = () => { window.__qa.printed = true; };
    new MutationObserver(() => { const stage = document.getElementById('progress-stage')?.textContent; if (stage && !window.__qa.stages.includes(stage)) window.__qa.stages.push(stage); }).observe(document, { subtree: true, childList: true, characterData: true });
  });
  const page = await context.newPage();
  page.on('pageerror', error => report.console_errors.push(error.message));
  page.on('console', message => { if (['warning', 'error'].includes(message.type())) report.console_errors.push(message.text()); });
  await page.goto(url);
  await page.addStyleTag({ content: '@font-face{font-family:QA-Korean;src:url(/__qa-font-ko)}@font-face{font-family:QA-Chinese;src:url(/__qa-font-zh)}:root{font-family:QA-Korean,QA-Chinese,sans-serif}' });
  await page.evaluate(async () => { await document.fonts.load('16px QA-Korean', '한글'); await document.fonts.load('16px QA-Chinese', '此段文字'); await document.fonts.ready; });
  async function screenshot(name) { const target = path.join(screenshotDir, name + '.png'); await page.screenshot({ path: target, fullPage: true }); report.screenshots.push(target); }
  async function uploadAll() {
    if (await page.locator('#home').isVisible()) await page.click('#choose-upload');
    for (const direction of ['front', 'left_oblique', 'right_oblique']) await page.locator(`#file-${direction}`).setInputFiles(uploadFile());
    await page.waitForFunction(() => !document.getElementById('submit').disabled);
    await page.click('#submit');
  }
  async function restart() { await page.click('#restart'); await page.locator('#home').waitFor({ state: 'visible' }); }
  async function expectAlert(part) { await page.waitForFunction(part => document.getElementById('alert').textContent.includes(part), part); }
  await check('identity / meaningful first screen / no framework overlay / desktop screenshot', async () => {
    assert.equal(page.url(), url); assert.equal(await page.title(), 'FaceTeller · 기능검증 파일럿');
    assert.match(await page.locator('#home').innerText(), /세 방향으로 입력하기/);
    assert.equal(await page.locator('vite-error-overlay,nextjs-portal').count(), 0);
    await screenshot('desktop-home');
  });
  await check('permission denied → upload, no physical camera used', async () => {
    await page.click('#choose-camera'); await expectAlert('권한이 거부');
    assert.equal(await page.locator('#input').isVisible(), true);
    assert.equal(await page.evaluate(() => window.__qa.cameraRequests), 1);
  });
  await check('triplet gate / invalid MIME / broken decode / oversize / replace and delete URLs', async () => {
    assert.equal(await page.locator('#submit').isDisabled(), true);
    await page.locator('#file-front').setInputFiles({ name: 'bad.txt', mimeType: 'text/plain', buffer: Buffer.from('not image') });
    await page.waitForFunction(() => document.getElementById('state-front').textContent.includes('재선택'));
    await page.locator('#file-front').setInputFiles({ name: 'broken.png', mimeType: 'image/png', buffer: Buffer.from('bad png') });
    await page.waitForFunction(() => document.getElementById('state-front').textContent.includes('읽을 수 없습니다'));
    await page.locator('#file-front').setInputFiles({ name: 'large.png', mimeType: 'image/png', buffer: Buffer.alloc(10 * 1024 * 1024 + 1) });
    await page.waitForFunction(() => document.getElementById('state-front').textContent.includes('10 MiB'));
    await page.locator('#file-front').setInputFiles(uploadFile()); await page.locator('#image-front').waitFor({ state: 'visible' });
    await page.locator('#file-front').setInputFiles(uploadFile());
    await page.waitForFunction(() => window.__qa.revoked >= 1 && !document.getElementById('image-front').hidden);
    await page.getByRole('button', { name: '정면 사진 지우기', exact: true }).click();
    assert.equal(await page.locator('#image-front').isVisible(), false);
    await page.locator('#file-front').setInputFiles(uploadFile());
    await page.locator('#file-left_oblique').setInputFiles(uploadFile());
    assert.equal(await page.locator('#submit').isDisabled(), true);
    assert.equal(report.requests.length, 0);
    await page.setViewportSize(report.environment.viewports[1]); await screenshot('mobile-triplet');
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.setViewportSize(report.environment.viewports[0]);
  });
  await check('multipart 3 fields + input_mode, NDJSON real progress and partial result', async () => {
    await uploadAll(); await page.locator('#results').waitFor({ state: 'visible' });
    assert.deepEqual(report.requests[0].fields, ['front', 'left_oblique', 'right_oblique', 'input_mode']);
    assert.equal(report.requests[0].input_mode, 'upload'); assert.equal(report.requests[0].only_solid_fixture, true);
    assert.equal(await page.locator('#passages .passage').count(), 5);
    const items = await page.locator('#items').innerText(); assert.match(items, /82.0%/); assert.match(items, /0.125 \(ratio\)/); assert.match(items, /방향 불충족/);
    for (const stage of ['얼굴 정렬', '부위 추출', '형태 분류', '깊이 분석', '문헌 선정']) assert.ok((await page.evaluate(() => window.__qa.stages)).includes(stage));
    assert.equal(await page.evaluate(() => window.__qa.created === window.__qa.revoked), true);
    assert.equal(await page.locator('#slots img[src]').count(), 0);
    await screenshot('desktop-results');
  });
  const pdfjs = await import(pathToFileURL(require.resolve('pdfjs-dist/legacy/build/pdf.mjs')).href);
  async function inspectPDF(filename, expected) {
    const filePath = path.join(out, filename);
    await page.pdf({ path: filePath, format: 'A4', preferCSSPageSize: true, printBackground: true, displayHeaderFooter: false });
    const pdf = await pdfjs.getDocument({ data: new Uint8Array(fs.readFileSync(filePath)), useSystemFonts: true, isEvalSupported: false }).promise;
    let full = '', imageOperators = 0; const dimensions = [];
    for (let i = 1; i <= pdf.numPages; i++) {
      const p = await pdf.getPage(i); dimensions.push(p.view);
      const content = await p.getTextContent(); full += content.items.map(item => item.str).join('');
      for (const item of content.items.filter(item => item.str?.trim())) {
        const [,,,, x, y] = item.transform;
        assert.ok(x >= 27 && x + item.width <= p.view[2] - 27, `PDF text exceeds horizontal print margin on page ${i}`);
        assert.ok(y >= 27 && y + item.height <= p.view[3] - 27, `PDF text exceeds vertical print margin on page ${i}`);
      }
      const ops = await p.getOperatorList();
      imageOperators += ops.fnArray.filter(op => [pdfjs.OPS.paintImageXObject, pdfjs.OPS.paintInlineImageXObject, pdfjs.OPS.paintImageMaskXObject].includes(op)).length;
    }
    assert.equal(imageOperators, 0, 'The result PDF must contain no photo/image raster operators');
    const normalized = normalize(full);
    for (const p of expected.passages) {
      assert.ok(normalized.includes(normalize(p.original_text)), `PDF missing original: ${p.passage_id}`);
      assert.ok(normalized.includes(normalize(p.translation)), `PDF missing translation: ${p.passage_id}`);
      for (const key of ['literature_title', 'chapter', 'passage_no']) assert.ok(normalized.includes(normalize(p.source[key])), `PDF missing source field: ${key}`);
      if (p.commentary) assert.ok(!normalized.includes(normalize(p.commentary)), 'Detailed commentary leaked into print');
    }
    for (const item of expected.items) {
      assert.ok(normalized.includes(normalize(item.basis)), 'Missing judgment basis');
      const boundary = formatMeasurementScore(item.measurement_score);
      if (boundary) assert.ok(normalized.includes(normalize(boundary)), 'Missing boundary score / not-probability notice');
    }
    assert.ok(normalized.includes(normalize(expected.notice || notice)));
    const responseOrigin = expected === recordedResult ? 'recorded-real-inference-json / HTTP stub replay' : 'synthetic fixture';
    if (expected === recordedResult) assert.ok(normalized.includes(normalize('실제 추론 결과 JSON 재생 · HTTP 전송 stub')));
    const entry = { path: filePath, response_origin: responseOrigin, pages: pdf.numPages, page_boxes_pt: dimensions, text_preserved: true, text_within_printable_bounds: true, image_operators: imageOperators, detail_commentary_excluded: true, original_characters: expected.passages.reduce((n, p) => n + p.original_text.length, 0), translation_characters: expected.passages.reduce((n, p) => n + p.translation.length, 0), extracted_characters: full.length };
    report.pdfs.push(entry); await pdf.destroy(); return entry;
  }
  await check('A4 actual PDF, 5 long CJK passages, entire original/translation/source/basis', async () => {
    await page.click('#preview'); await page.waitForFunction(() => document.getElementById('print-status').textContent.includes('본문'));
    assert.equal(await page.locator('#print').isEnabled(), true, await page.locator('#print-status').innerText());
    const metrics = await page.evaluate(() => ({ font: getComputedStyle(document.getElementById('print-sheet')).fontSize, height: document.getElementById('print-content').getBoundingClientRect().height, max: 277 * 96 / 25.4, ellipsis: [...document.querySelectorAll('#print-content *')].some(e => getComputedStyle(e).textOverflow === 'ellipsis' || getComputedStyle(e).overflow === 'hidden') }));
    assert.equal(metrics.ellipsis, false); assert.ok(metrics.height < metrics.max);
    await screenshot('print-preview');
    const pdf = await inspectPDF('printpilot.pdf', normal); assert.equal(pdf.pages, 1);
    assert.ok(Math.abs(pdf.page_boxes_pt[0][2] - 595.28) < 1); assert.ok(Math.abs(pdf.page_boxes_pt[0][3] - 841.89) < 1);
    await page.click('#back-results'); assert.equal(await page.locator('#passages .passage').count(), 5);
    return metrics;
  });
  await check('reinput / no literature / HTTP error / stream error / truncation / malformed / cancel', async () => {
    for (const [selected, expected] of [['empty', '선정 가능한 문헌이 없습니다'], ['reinput', '확정 가능한 항목이 없습니다'], ['http-error', '좌측 사선: 얼굴 미검출'], ['stream-error', '우측 사선: 전처리 실패'], ['interrupted', '최종 결과 전에 연결'], ['malformed', 'JSON'], ['cancel', '요청을 취소']]) {
      await restart(); scenario = selected; await uploadAll();
      if (selected === 'cancel') await page.click('#cancel');
      if (selected === 'empty') { await page.locator('#results').waitFor({ state: 'visible' }); assert.match(await page.locator('#passages').innerText(), /선정 가능한 문헌/); }
      else { await expectAlert(expected); assert.equal(await page.locator('#input').isVisible(), true); assert.equal(await page.locator('#slots img[src]').count(), 0); assert.equal(await page.locator('#submit').isDisabled(), true); }
      assert.equal(await page.evaluate(() => window.__qa.created === window.__qa.revoked), true);
    }
  });
  await check('response text safely rendered without HTML execution', async () => {
    await restart(); scenario = 'xss'; await uploadAll(); await page.locator('#results').waitFor({ state: 'visible' });
    assert.equal(await page.locator('#passages img').count(), 0); assert.equal(await page.evaluate(() => !!window.xss), false);
    assert.match(await page.locator('#passages').innerText(), /<img src=x/);
  });
  await check('oversize text: explicit warning, single-page blocked, full-text multipage PDF fallback', async () => {
    await restart(); scenario = 'overflow'; await uploadAll(); await page.locator('#results').waitFor({ state: 'visible' });
    await page.click('#preview'); await page.waitForFunction(() => document.getElementById('print-status').textContent.includes('여러 장'));
    assert.equal(await page.locator('#print').isDisabled(), true); assert.equal(await page.locator('#print-fallback').isVisible(), true);
    await page.screenshot({ path: path.join(screenshotDir, 'overflow-warning.png'), fullPage: false }); report.screenshots.push(path.join(screenshotDir, 'overflow-warning.png'));
    await page.click('#print-fallback'); assert.equal(await page.evaluate(() => window.__qa.printed), true);
    const pdf = await inspectPDF('printpilot-overflow-fallback.pdf', oversized); assert.ok(pdf.pages > 1);
  });
  await check('recorded real-inference JSON: separate boundary score, result rendering and PDF (HTTP stub)', async () => {
    for (const value of [null, undefined, NaN, Infinity, -.1, 1.1, '0.5']) assert.equal(formatMeasurementScore(value), '');
    assert.equal(formatMeasurementScore(0), '경계점수 0.0점 · 정답확률 아님');
    assert.equal(formatMeasurementScore(.625), '경계점수 62.5점 · 정답확률 아님');
    assert.equal(formatMeasurementScore(1), '경계점수 100.0점 · 정답확률 아님');
    const source = process.env.PILOT_REAL_RESULT || path.join(root, 'data/derived/pilot-20261014/api_result_example.json');
    const bytes = fs.readFileSync(source);
    const supplied = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
    recordedResult = supplied.type === 'result' ? supplied.data : supplied;
    assert.ok(Array.isArray(recordedResult.items)); assert.ok(Array.isArray(recordedResult.passages));
    const depth = recordedResult.items.filter(item => item.kind === 'depth');
    assert.ok(depth.length > 0, 'Expected depth measurements in recorded inference data');
    assert.ok(depth.some(item => typeof item.measurement_score === 'number'), 'Expected a numeric CDF boundary score');
    for (const item of depth) assert.equal(item.confidence, null, 'Depth score must not masquerade as confidence');
    report.recorded_result_replay = { source, sha256: createHash('sha256').update(bytes).digest('hex'), response_origin: 'actual inference result provided by main', transport: 'local HTTP stub; solid PNG request', live_inference_executed: false, item_count: recordedResult.items.length, passage_count: recordedResult.passages.length };
    const textQuality = { utf8_valid: true, replacement_characters: 0, affected_fields: [] };
    const inspectText = (value, field = '') => {
      if (typeof value === 'string' && value.includes('\uFFFD')) { textQuality.replacement_characters += [...value].filter(c => c === '\uFFFD').length; textQuality.affected_fields.push(field); }
      else if (value && typeof value === 'object') for (const [key, child] of Object.entries(value)) inspectText(child, field ? `${field}.${key}` : key);
    };
    inspectText(recordedResult);
    report.recorded_result_replay.source_text_quality = textQuality;
    await restart(); scenario = 'recorded-inference';
    await page.evaluate(() => { window.FACETELLER_PILOT_ADAPTERS.fixture = 'recorded-inference'; });
    await uploadAll(); await page.locator('#results').waitFor({ state: 'visible' });
    assert.match(await page.locator('#fixture-banner').innerText(), /실제 추론 결과 JSON 재생.*HTTP 전송 stub/);
    const cards = page.locator('#items .item'); assert.equal(await cards.count(), recordedResult.items.length);
    for (let index = 0; index < recordedResult.items.length; index++) {
      const item = recordedResult.items[index], summary = await cards.nth(index).locator('strong').innerText();
      const boundary = formatMeasurementScore(item.measurement_score);
      if (boundary) assert.ok(summary.includes(boundary)); else assert.ok(!summary.includes('경계점수'));
      if (item.kind === 'depth') assert.ok(!summary.includes('신뢰도'));
    }
    assert.equal(await page.locator('#passages .passage').count(), recordedResult.passages.length);
    if (!recordedResult.passages.length) {
      assert.equal(recordedResult.literature_status, 'NONE');
      assert.match(await page.locator('#passages').innerText(), /선정 가능한 문헌이 없습니다/);
    }
    for (const passage of recordedResult.passages) {
      const rendered = await page.locator('#passages').innerText();
      for (const text of [passage.original_text, passage.translation, passage.commentary, ...['literature_title', 'chapter', 'passage_no'].map(key => passage.source[key])]) assert.ok(rendered.includes(text), 'Missing passage text or required source field');
    }
    await screenshot('recorded-inference-results');
    if (recordedResult.passages.length) {
      const passageScreenshot = path.join(screenshotDir, 'recorded-inference-passage.png');
      await page.locator('#passages .passage').first().screenshot({ path: passageScreenshot });
      report.screenshots.push(passageScreenshot);
    }
    const boundaryScreenshot = path.join(screenshotDir, 'recorded-inference-boundary-score.png');
    await cards.nth(recordedResult.items.findIndex(item => typeof item.measurement_score === 'number')).screenshot({ path: boundaryScreenshot });
    report.screenshots.push(boundaryScreenshot);
    await page.click('#preview'); await page.waitForFunction(() => document.getElementById('print-status').textContent.includes('본문'));
    const singlePage = await page.locator('#print').isEnabled();
    if (!singlePage) await page.click('#print-fallback');
    await screenshot('recorded-inference-print-preview');
    const pdf = await inspectPDF('printpilot-recorded-inference.pdf', recordedResult);
    if (singlePage) assert.equal(pdf.pages, 1);
    assert.equal(report.requests.at(-1).scenario, 'recorded-inference');
    assert.equal(report.requests.at(-1).only_solid_fixture, true);
    report.recorded_result_replay.actual_literature_exercised = recordedResult.passages.length > 0;
    return { ...report.recorded_result_replay, pdf_pages: pdf.pages, single_page_fit: singlePage, boundary_score_not_accuracy: true };
  });
  await check('cleanup / no storage / no photo-bearing PDF / app console health', async () => {
    await restart();
    assert.equal(await page.evaluate(() => window.__qa.created === window.__qa.revoked), true);
    assert.equal(await page.evaluate(() => window.__qa.storageWrites), 0);
    assert.equal(await page.locator('#print-content').innerText(), '');
    assert.equal(await page.locator('#slots img[src]').count(), 0);
    // Browser reports expected 422 as a resource error; no app exception is allowed.
    const unexpected = report.console_errors.filter(s => !s.includes('422 (Unprocessable Entity)'));
    assert.deepEqual(unexpected, []);
    return { expected_console_resource_error: 'one deliberately injected HTTP 422', no_app_exceptions: true };
  });
  report.status = 'PASS';
  report.limitations = [
    'Vanilla JS experiment; React/TypeScript design requirement not implemented.',
    'Live API/inference integration is reserved for main; recorded actual-result JSON was replayed through an HTTP stub. No CPU/GPU inference was run by this UI test.',
    'Camera states tested with pure boolean fixtures; only permission-denied GUI exercised. No real webcam/detector claim.',
    'Upload format/size/decode checked locally; face/direction detection deferred to backend unless validateFace adapter supplied.',
    'One-page guarantee is bounded by actual content. Text exceeding minimum 8pt fit is explicitly warned and printed across pages without truncation.',
    'PDF proof uses installed Windows CJK fonts via a fixture-server-only font route; deployment must supply equivalent CJK fonts.',
    'Browser UI memory/URLs/storage checked; backend memory-only multipart/log lifecycle is outside this independent UI run.'
  ];
  if (!recordedResult.passages.length) report.limitations.push('The supplied actual result contains zero passages (literature_status NONE). Actual-literature content was not exercised; synthetic 5-passage preservation still passed. Rerun with a nonempty actual result to cover real literature.');
  if (report.recorded_result_replay.source_text_quality.replacement_characters) {
    report.status = 'PASS_WITH_SOURCE_LIMITATIONS';
    report.limitations.push(`The supplied JSON itself contains ${report.recorded_result_replay.source_text_quality.replacement_characters} U+FFFD replacement characters in ${report.recorded_result_replay.source_text_quality.affected_fields.length} fields. UI/PDF preserve those source strings; readable actual labels/basis require an upstream encoding fix. No source data was repaired or changed by this test.`);
  }
  const review = [
    'FaceTeller 파일럿 UI 검토 — 2026-10-14',
    `결과: ${report.status}, ${report.checks.length}개 검증 PASS`,
    `환경: ${url}; Chromium ${report.environment.chromium_version}; 1280×900 / 390×844; Browser plugin not available → Playwright`,
    '검증 데이터: 기존 11개는 합성 문장/단색 PNG fixture. 추가 1개는 main이 제공한 실제 추론 결과 JSON 재생. HTTP 전송은 stub이고 업로드는 단색 PNG만 사용. 얼굴 사진/원시좌표 미사용. UI 시험 중 CPU/GPU 추론 없음.',
    '신규 화면: /pilot, /pilot.js, /pilot.css. 기존 web/v2.html 및 기존 서버 파일 수정 없음.',
    '요청: POST /api/v1/analysis; front,left_oblique,right_oblique,input_mode. NDJSON progress/result/error. fixture 서버만 사용.',
    `실제 결과 JSON: ${report.recorded_result_replay.source}; SHA-256 ${report.recorded_result_replay.sha256}`,
    `실제 결과 내용: ${recordedResult.items.length}항목, 문헌 ${recordedResult.passages.length}개. 실제 문헌 출력 검증: ${recordedResult.passages.length ? '수행' : '미수행 — 제공 JSON은 정상 문헌 없음(NONE) 상태. 합성 문헌 5개 출력 검증은 별도로 통과.'}`,
    `원본 문자 품질: JSON 자체 U+FFFD 대체문자 ${report.recorded_result_replay.source_text_quality.replacement_characters}개. 깨진 원본 문자열을 보존하며 임의 복원하지 않음. 경계점수 및 UI 고정 안내 문구는 정상.`,
    '표시: measurement_score는 경계점수 0–100점으로 별도 표시하고 정답확률 아님을 명시. null은 점수 숨김. depth confidence는 신뢰도로 표시하지 않음. 결과 화면과 PDF에 동일 적용.',
    ...report.checks.map(c => `PASS: ${c.name}`),
    ...report.pdfs.map(p => `PDF: ${path.basename(p.path)} — ${p.pages}페이지, 원문 ${p.original_characters}자 / 직역 ${p.translation_characters}자 전부 추출 대조 통과; 상세해설 제외; 출처/판정 근거 포함.`),
    '인쇄: 11pt부터 8pt까지 0.25pt 간격 맞춤. 190×277mm 인쇄 영역, @page A4 10mm 여백. 숨김/ellipsis/문자 삭제 없음. 한 장 불가 시 경고 + 여러 장 출력.',
    '촬영: generation/direction token으로 지연 결과 무시, 각 조건 이탈 초기화, 1초 연속 유지, 3방향 각 1회, submit_once. UI에서 권한 거부→업로드 검증. 실제 카메라 미검증.',
    '개선/제약:', ...report.limitations.map(s => `- ${s}`),
    `재실행: ${report.environment.command} (Playwright Chromium 및 CJK 폰트 필요). 외부 API 연결 없음.`,
    '실행환경 변경: Linux Chromium 151 다운로드 후 libnspr4.so 부재로 기동 실패. 기존 Windows Chromium 149 + node.exe 사용. 의존성/기존 프로젝트 파일 변경 없음.',
    'PDF 좌표검사: 모든 추출 글자가 A4 인쇄 여백 안에 있고, 사진/래스터 이미지 연산자가 0개임을 확인.',
    '스크린샷: ', ...report.screenshots,
    'Browser 플러그인을 추가하면 앱 내 DOM/스크린샷 검토를 함께 사용할 수 있음.'
  ].join('\n') + '\n';
  fs.writeFileSync(path.join(out, 'ui-review.txt'), review);
  saveReport();
  console.log(JSON.stringify({ status: report.status, checks: report.checks.length, pdfs: report.pdfs.map(p => ({ path: p.path, pages: p.pages })), report: path.join(out, 'ui_results.json') }));
}
main().catch(error => { report.status = 'FAIL'; report.failure = error.stack; saveReport(); console.error(error); process.exitCode = 1; }).finally(async () => { await browser?.close(); server?.closeAllConnections(); await new Promise(resolve => server ? server.close(resolve) : resolve()); });
