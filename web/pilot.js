/* Independent vanilla JS functional pilot. No persistence, analytics or model inference.
 * Optional integration: window.FACETELLER_PILOT_ADAPTERS = { evaluateFrame,
 * validateFace, fixture: false }. evaluateFrame(video, direction, token) returns
 * {face,direction,size,brightness,message}; validateFace(file,direction) returns
 * {valid,message}. These hooks must not persist frames or call external services.
 */
export const DIRECTIONS = ['front', 'left_oblique', 'right_oblique'];
const LABELS = { front: '정면', left_oblique: '좌측 사선', right_oblique: '우측 사선' };
const MAX_BYTES = 10 * 1024 * 1024;
const NOTICE = '문헌 조회 결과이며 성격·지능·건강·운명을 판정하지 않습니다.';
// Short display names from label_schema_kface_v3.json; unknown features keep their ID.
const FEATURE_NAMES = {
  eyelid: '눈꺼풀', sclera_show: '흰자 노출', eye_depth: '눈 깊이',
  brow_density: '눈썹 털 밀도', brow_tail_clarity: '눈썹 꼬리 선명도',
  nose_tip: '코끝 모양', nostril_show: '정면 콧구멍 노출', alar_fullness: '콧방울 풍만함',
  lip_bow: '윗입술 활 모양', chin_shape: '턱끝 모양', cheekbone: '광대 돌출', cheek_fullness: '볼살',
  eye_size: '눈 크기', eye_roundness: '눈 세로 개방', eye_tilt: '눈꼬리 기울기', eye_asymmetry: '좌우 눈 개방 차이',
  brow_shape: '눈썹 모양', brow_thickness: '눈썹 굵기', brow_length: '눈썹 길이', brow_eye_gap: '눈썹과 눈 간격', brow_tail: '눈썹 꼬리 방향',
  nose_width: '콧방울 폭', nose_length: '코 길이', nose_projection: '코끝 돌출', nose_bridge: '코뿌리 높이',
  mouth_width: '입 너비', lip_thickness: '입술 두께', lip_balance: '위·아래 입술 두께 비', mouth_corner: '입꼬리 높이',
  face_shape: '얼굴 윤곽 형태', face_length: '얼굴 세로·가로 비', jaw_width: '아래턱 폭',
  lip_projection: '입 돌출', chin_projection: '턱 돌출'
};

export function formatMeasurementScore(value) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1
    ? `경계점수 ${(value * 100).toFixed(1)}점 · 정답확률 아님` : '';
}

export function formatMatchingScore(value) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1
    ? `종합 일치점수 ${(value * 100).toFixed(1)}점` : '종합 일치점수 산출 없음';
}

// Pure camera transition model: async evaluations carry an immutable generation token.
export function createCaptureState(holdMs = 1000) {
  let generation = 0, index = 0, since = null, lastTime = -Infinity, pending = false, submitted = false;
  return {
    token: () => ({ generation, direction: DIRECTIONS[index] }),
    reset() { generation++; index = 0; since = null; lastTime = -Infinity; pending = false; submitted = false; },
    update(conditions, now, token) {
      const direction = DIRECTIONS[index];
      if (!direction || pending || token.generation !== generation || token.direction !== direction || !Number.isFinite(now) || now <= lastTime) return { ignored: true };
      lastTime = now;
      if (!['face', 'direction', 'size', 'brightness'].every(k => conditions[k] === true)) { since = null; return { progress: 0, capture: false }; }
      if (since === null) since = now;
      const progress = Math.min(1, (now - since) / holdMs);
      if (progress === 1) pending = true;
      return { progress, capture: pending, direction };
    },
    commit(token) {
      if (!pending || token.generation !== generation || token.direction !== DIRECTIONS[index]) return false;
      index++; generation++; pending = false; since = null; lastTime = -Infinity;
      return true;
    },
    submitOnce() { if (index !== 3 || submitted) return false; submitted = true; return true; }
  };
}

export async function readNDJSON(response, onProgress) {
  if (!response.ok) {
    let error;
    try { error = await response.json(); } catch { error = {}; }
    if (error.detail && typeof error.detail === 'object') error = error.detail;
    throw Object.assign(new Error(error.message || `서버 요청 실패 (HTTP ${response.status})`), { direction: error.direction, code: error.code });
  }
  if (!response.headers.get('content-type')?.includes('application/x-ndjson') || !response.body) throw new Error('NDJSON 응답 형식이 아닙니다.');
  const reader = response.body.getReader(), decoder = new TextDecoder('utf-8', { fatal: true });
  let buffer = '', result, terminal = false;
  const line = raw => {
    if (!raw.trim()) return;
    if (terminal) throw new Error('최종 이벤트 뒤에 추가 데이터가 도착했습니다.');
    const event = JSON.parse(raw);
    if (event.type === 'progress' && typeof event.stage === 'string') onProgress(event.stage);
    else if (event.type === 'error') throw Object.assign(new Error(event.message || '서버 분석 오류'), { direction: event.direction, code: event.code });
    else if (event.type === 'result' && event.data) { result = validateResult(event.data); terminal = true; }
    else throw new Error('알 수 없는 분석 이벤트입니다.');
  };
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      if (buffer.length > 2_000_000) throw new Error('응답 크기 제한을 초과했습니다.');
      let at;
      while ((at = buffer.indexOf('\n')) !== -1) { line(buffer.slice(0, at)); buffer = buffer.slice(at + 1); }
      if (done) { line(buffer); break; }
    }
    if (!terminal) throw new Error('최종 결과 전에 연결이 종료되었습니다. 다시 입력해 주세요.');
    return result;
  } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
}

function validateResult(data) {
  if (!['COMPLETE', 'PARTIAL', 'REINPUT_REQUIRED'].includes(data.status) || !Array.isArray(data.items) || !Array.isArray(data.passages) || data.passages.length > 5) throw new Error('분석 결과 형식이 올바르지 않습니다.');
  for (const item of data.items) {
    if (!['CONFIRMED', 'HELD'].includes(item.status)) throw new Error('항목 상태가 올바르지 않습니다.');
  }
  for (const p of data.passages) {
    if (!['original_text', 'translation', 'commentary'].every(k => typeof p[k] === 'string') || !p.source || !['literature_title', 'chapter', 'passage_no'].every(k => typeof p.source[k] === 'string') || !Array.isArray(p.evidence_item_ids)) throw new Error('문헌 필수 필드가 누락되었습니다.');
  }
  return data;
}

export async function validateInputFile(file) {
  if (!file || !['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) throw new Error('JPEG, PNG, WebP 파일을 선택하세요.');
  if (!file.size || file.size > MAX_BYTES) throw new Error('각 사진은 0바이트보다 크고 10 MiB 이하여야 합니다.');
  const bitmap = await createImageBitmap(file).catch(() => { throw new Error('이미지를 읽을 수 없습니다. 다른 파일을 선택하세요.'); });
  bitmap.close();
}

function mount() {
  const $ = id => document.getElementById(id);
  const adapters = window.FACETELLER_PILOT_ADAPTERS || {};
  const slots = new Map(), versions = new Map();
  const cameraState = createCaptureState();
  let epoch = 0, mode = 'upload', stream = null, raf = null, aborter = null, result = null, busy = false, printFits = false;
  const node = (tag, text, cls) => { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (cls) el.className = cls; return el; };
  const show = id => {
    for (const name of ['home', 'input', 'camera', 'progress', 'results', 'preview-panel']) $(name).hidden = name !== id;
    $('restart').hidden = id === 'home';
  };
  const alert = text => { $('alert').textContent = text; $('alert').hidden = !text; };
  const closeCamera = () => { if (raf !== null) cancelAnimationFrame(raf); raf = null; stream?.getTracks().forEach(t => t.stop()); stream = null; $('video').srcObject = null; };
  const clearSlot = direction => {
    const old = slots.get(direction); if (old?.url) URL.revokeObjectURL(old.url); slots.delete(direction);
    const img = $(`image-${direction}`); img.removeAttribute('src'); img.hidden = true;
    $(`file-${direction}`).value = ''; $(`state-${direction}`).textContent = '미선택';
    versions.set(direction, (versions.get(direction) || 0) + 1);
  };
  const clearPhotos = () => { DIRECTIONS.forEach(clearSlot); refreshInputs(); };
  const refreshInputs = () => {
    const count = [...slots.values()].filter(s => s.valid).length;
    $('input-summary').textContent = `사진 ${count}/3장 준비`;
    $('submit').disabled = busy || count !== 3;
  };
  const reset = () => {
    epoch++; aborter?.abort(); aborter = null; busy = false; closeCamera(); cameraState.reset(); clearPhotos(); result = null;
    $('items').replaceChildren(); $('passages').replaceChildren(); $('print-content').replaceChildren();
    document.body.removeAttribute('data-print-mode'); alert(''); show('home');
  };
  const returnInput = message => {
    closeCamera(); clearPhotos(); cameraState.reset(); alert(message);
    if (mode === 'camera') { show('camera'); $('camera-retry').hidden = false; $('camera-direction').textContent = '사진을 다시 촬영해 주세요.'; $('camera-guide').textContent = '이전 임시 사진을 정리했습니다.'; setGauge(0); }
    else show('input');
  };
  const sourceText = p => `${p.source.literature_title} · ${p.source.chapter} · ${p.source.passage_no}`;
  const fixtureNotice = () => adapters.fixture === 'recorded-inference'
    ? '실제 추론 결과 JSON 재생 · HTTP 전송 stub · 실시간 추론 검증 아님 · 사진 없음'
    : 'FIXTURE 검증 · 합성 이미지/응답 · 실제 얼굴 분석 결과 아님';
  const basisText = value => Array.isArray(value) ? value.join(' / ') : typeof value === 'string' ? value : value == null ? '' : JSON.stringify(value);
  const itemText = item => {
    const boundaryScore = formatMeasurementScore(item.measurement_score);
    const boundary = boundaryScore ? ` · ${boundaryScore}` : '';
    if (item.status === 'HELD') return `${item.region} · 보류 — ${basisText(item.hold_reasons) || '확인할 수 없음'}${boundary}`;
    const score = item.kind !== 'depth' && item.confidence != null ? ` · 신뢰도 ${(item.confidence * 100).toFixed(1)}%` : '';
    const measurement = item.measurement ? ` · ${item.measurement.indicator ?? item.measurement.name ?? '상대 측정'} ${item.measurement.value ?? item.measurement.ratio} (${item.measurement.unit || 'ratio'})` : '';
    return `${item.region} · ${item.label_name || '확정'}${score}${measurement}${boundary}`;
  };
  const evidence = p => p.evidence_item_ids.map(id => result.items.find(i => i.item_id === id)).filter(Boolean).map(i => `${i.label_name}: ${basisText(i.basis)}`).join(' / ');
  // Keep the server's selection/order/rank. Scores explain the result; the UI does not rerank.
  const rankText = (p, index) => `${Number.isInteger(p.rank) && p.rank > 0 ? p.rank : index + 1}위 · ${formatMatchingScore(p.G)}`;
  const appendRanking = (box, passage, index) => {
    box.append(node('p', rankText(passage, index), 'passage-ranking'));
    const components = Array.isArray(passage.score_components) ? passage.score_components : [];
    const count = Number.isInteger(passage.score_feature_count) && passage.score_feature_count >= 0 ? passage.score_feature_count : components.length;
    if (count) box.append(node('p', `반영 항목 ${count}개 · 정답확률이 아닙니다.`, 'score-notice'));
    if (!components.length) return;
    const details = node('details', undefined, 'score-details');
    details.append(node('summary', '항목별 점수 보기'));
    details.append(node('p', '아래 항목 점수의 기하평균으로 종합 일치점수를 산출합니다. 합산 점수나 정답확률이 아닙니다.', 'score-notice'));
    const list = node('ul');
    components.forEach(component => {
      const label = component.label_name || component.label_id || component.item_id;
      const value = component.score;
      const points = typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1 ? `${(value * 100).toFixed(1)}점` : '점수 없음';
      const meaning = component.kind === 'depth' ? '깊이 측정(VGGT) · 경계점수 · 정답확률 아님' : '형태 분류(CNN) · 보정된 분류 신뢰도 환산';
      const feature = FEATURE_NAMES[component.item_id] || component.item_id;
      list.append(node('li', `${feature} · ${label} — ${points} · ${meaning}`));
    });
    details.append(list); box.append(details);
  };
  const renderResult = () => {
    if (adapters.fixture) $('fixture-banner').textContent = fixtureNotice();
    $('result-status').textContent = result.status === 'PARTIAL' ? '일부 항목 보류 · 가능한 항목의 결과입니다.' : '판정 결과';
    $('result-notice').textContent = result.notice || NOTICE;
    $('items').replaceChildren(...result.items.map(item => { const box = node('div', undefined, `item ${item.status === 'HELD' ? 'held' : ''}`); box.append(node('strong', itemText(item)), node('p', `판정 근거: ${basisText(item.basis)}`)); return box; }));
    $('passages').replaceChildren(...result.passages.map((p, index) => {
      const box = node('article', undefined, 'passage'); appendRanking(box, p, index); box.append(node('h3', sourceText(p)));
      for (const [label, content] of [['원문', p.original_text], ['직역', p.translation], ['해설', p.commentary], ['판정 근거', evidence(p)]]) box.append(node('p', `${label}: ${content}`));
      return box;
    }));
    if (!result.passages.length) $('passages').append(node('p', '선정 가능한 문헌이 없습니다. 형태 판정 결과는 위에서 확인할 수 있습니다.'));
    show('results');
  };
  async function submit() {
    if (busy || DIRECTIONS.some(d => !slots.get(d)?.valid)) return;
    busy = true; const session = epoch; aborter = new AbortController();
    const data = new FormData();
    DIRECTIONS.forEach(d => data.append(d, slots.get(d).file, `${d}.${slots.get(d).file.type.split('/')[1]}`)); data.append('input_mode', mode);
    closeCamera(); clearPhotos(); alert(''); show('progress'); $('progress-stage').textContent = '서버 응답을 기다립니다.';
    try {
      const response = await fetch('/api/v1/analysis', { method: 'POST', body: data, signal: aborter.signal, cache: 'no-store' });
      const received = await readNDJSON(response, stage => { if (session === epoch) $('progress-stage').textContent = stage; });
      if (session !== epoch) return;
      if (received.status === 'REINPUT_REQUIRED') { returnInput(`확정 가능한 항목이 없습니다. ${received.items.map(i => basisText(i.hold_reasons)).filter(Boolean).join(' / ')}`); }
      else { result = received; renderResult(); }
    } catch (error) {
      if (session !== epoch) return;
      returnInput(`${error.direction ? `${LABELS[error.direction] || error.direction}: ` : ''}${error.message}`);
    } finally { if (session === epoch) { busy = false; aborter = null; refreshInputs(); } }
  }
  DIRECTIONS.forEach(direction => {
    const card = node('div', undefined, 'slot'), label = node('label', LABELS[direction]); label.htmlFor = `file-${direction}`;
    const input = node('input'); input.type = 'file'; input.accept = 'image/jpeg,image/png,image/webp'; input.id = `file-${direction}`;
    const img = node('img'); img.id = `image-${direction}`; img.alt = `${LABELS[direction]} 선택 사진`; img.hidden = true;
    const status = node('p', '미선택'); status.id = `state-${direction}`; status.setAttribute('role', 'status');
    const remove = node('button', '사진 지우기'); remove.type = 'button'; remove.setAttribute('aria-label', `${LABELS[direction]} 사진 지우기`);
    remove.onclick = () => { clearSlot(direction); refreshInputs(); };
    input.onchange = async () => {
      const file = input.files[0]; clearSlot(direction); refreshInputs(); if (!file) return;
      const version = versions.get(direction), session = epoch; status.textContent = '검사 중';
      try {
        await validateInputFile(file);
        if (adapters.validateFace) { const check = await adapters.validateFace(file, direction); if (!check.valid) throw new Error(check.message || '얼굴 확인 실패'); }
        if (session !== epoch || version !== versions.get(direction)) return;
        const url = URL.createObjectURL(file); slots.set(direction, { file, url, valid: true }); img.src = url; img.hidden = false;
        status.textContent = adapters.validateFace ? '적합 · 얼굴 검사 완료' : '파일 적합 · 얼굴·방향은 서버 검사 대기';
      } catch (error) { if (session === epoch && version === versions.get(direction)) status.textContent = `재선택 필요: ${error.message}`; }
      refreshInputs();
    };
    card.append(label, input, img, status, remove); $('slots').append(card);
  });
  function setGauge(progress) { const gauge = document.querySelector('.gauge'); const pct = Math.round(progress * 100); gauge.style.setProperty('--value', `${pct}%`); gauge.setAttribute('aria-valuenow', String(pct)); $('gauge-label').textContent = `${pct}%`; }
  async function openCamera() {
    epoch++; const session = epoch; mode = 'camera'; clearPhotos(); cameraState.reset(); closeCamera(); alert(''); show('camera');
    $('camera-retry').hidden = true; $('camera-direction').textContent = '카메라 권한 요청 중'; setGauge(0);
    [...$('camera-steps').children].forEach(li => li.className = '');
    try {
      const opened = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
      if (session !== epoch) { opened.getTracks().forEach(t => t.stop()); return; }
      stream = opened; $('video').srcObject = stream; await $('video').play();
      if (session !== epoch) return;
      if (!adapters.evaluateFrame) throw new Error('촬영 조건 검사기가 연결되지 않았습니다. 업로드를 이용하세요.');
      const tick = async () => {
        if (session !== epoch || !stream) return;
        const token = cameraState.token(); $('camera-direction').textContent = `${LABELS[token.direction]} 촬영`;
        try {
          const conditions = await adapters.evaluateFrame($('video'), token.direction, token);
          if (session !== epoch || !stream) return;
          const next = cameraState.update(conditions, performance.now(), token);
          if (!next.ignored) { setGauge(next.progress); $('camera-guide').textContent = conditions.message || (next.progress ? '조건을 유지해 주세요.' : '얼굴·방향·크기·밝기를 확인하세요.'); }
          if (next.capture) {
            const canvas = document.createElement('canvas'); canvas.width = $('video').videoWidth; canvas.height = $('video').videoHeight;
            canvas.getContext('2d').drawImage($('video'), 0, 0);
            const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', .9)); canvas.width = 0; canvas.height = 0;
            if (session !== epoch || !stream) return;
            if (!blob) throw new Error('촬영 프레임을 읽을 수 없습니다.');
            if (cameraState.commit(token)) { slots.set(token.direction, { file: blob, valid: true }); $('camera-steps').children[DIRECTIONS.indexOf(token.direction)].className = 'done'; setGauge(0); }
            if (cameraState.submitOnce()) { await submit(); return; }
          }
          raf = requestAnimationFrame(tick);
        } catch (error) { if (session === epoch) { closeCamera(); mode = 'upload'; clearPhotos(); show('input'); alert(`카메라를 사용할 수 없습니다. ${error.message}`); } }
      };
      raf = requestAnimationFrame(tick);
    } catch (error) {
      if (session !== epoch) return; closeCamera(); mode = 'upload'; clearPhotos(); show('input');
      alert(error.name === 'NotAllowedError' ? '카메라 권한이 거부되었습니다. 사진 3장을 업로드하세요.' : `카메라를 사용할 수 없습니다. ${error.message}`);
    }
  }
  function buildPrint() {
    const content = $('print-content'); content.replaceChildren(node('h2', 'FaceTeller · 문헌 조회 결과지'));
    if (adapters.fixture) content.append(node('p', fixtureNotice(), 'print-notice'));
    content.append(node('p', result.notice || NOTICE, 'print-notice'), node('h3', '주요 판정 결과와 근거'));
    if (result.passages.length) content.append(node('p', '종합 일치점수는 형태의 일치 정도이며 정답확률이 아닙니다.', 'print-notice'));
    result.items.forEach(item => content.append(node('p', `${itemText(item)} · 근거: ${basisText(item.basis)}`, 'print-item')));
    if (!result.passages.length) content.append(node('p', '선정 가능한 문헌이 없습니다.'));
    result.passages.forEach((p, i) => {
      const box = node('section', undefined, 'print-passage');
      box.append(node('h3', `${rankText(p, i)} · ${sourceText(p)}`), node('p', `원문: ${p.original_text}`), node('p', `직역: ${p.translation}`), node('p', `판정 근거: ${evidence(p)}`)); content.append(box);
    });
  }
  function fitPrint() {
    if (!result) return false;
    const sheet = $('print-sheet'), content = $('print-content');
    document.body.dataset.printMode = 'single';
    // Millimetres are CSS absolute units (96px/in). Never clip or ellipsize text.
    const maxHeight = 277 * 96 / 25.4, maxWidth = 190 * 96 / 25.4;
    printFits = false;
    for (let size = 11; size >= 8; size -= .25) {
      sheet.style.fontSize = `${size}pt`;
      if (content.getBoundingClientRect().height <= maxHeight - 2 && content.scrollWidth <= maxWidth + 1) { printFits = true; break; }
    }
    $('print-status').textContent = printFits ? `A4 한 장 확인 · ${parseFloat(sheet.style.fontSize)}pt · 본문 생략 없음` : 'A4 한 장을 넘습니다. 본문은 생략하지 않습니다. 여러 장 인쇄를 이용하세요.';
    $('print').disabled = !printFits; $('print-fallback').hidden = printFits;
    if (!printFits) document.body.dataset.printMode = 'multipage';
    return printFits;
  }
  $('choose-upload').onclick = () => { mode = 'upload'; alert(''); show('input'); };
  $('choose-camera').onclick = openCamera; $('camera-retry').onclick = openCamera;
  $('camera-upload').onclick = () => { epoch++; closeCamera(); clearPhotos(); cameraState.reset(); mode = 'upload'; show('input'); alert(''); };
  $('submit').onclick = submit; $('restart').onclick = reset;
  $('cancel').onclick = () => { epoch++; aborter?.abort(); aborter = null; busy = false; returnInput('요청을 취소하고 임시 사진을 정리했습니다.'); };
  $('preview').onclick = async () => { buildPrint(); show('preview-panel'); await document.fonts.ready; fitPrint(); };
  $('back-results').onclick = () => show('results');
  $('print').onclick = () => { if (fitPrint()) window.print(); };
  $('print-fallback').onclick = () => { document.body.dataset.printMode = 'multipage'; $('print-sheet').style.fontSize = '10pt'; window.print(); };
  window.addEventListener('beforeprint', () => {
    if (result && !$('preview-panel').hidden) { if (document.body.dataset.printMode !== 'multipage') fitPrint(); }
    else if (result) { buildPrint(); show('preview-panel'); fitPrint(); }
  });
  window.addEventListener('pagehide', reset);
  document.addEventListener('visibilitychange', () => { if (document.hidden && stream) { epoch++; closeCamera(); cameraState.reset(); returnInput('화면이 숨겨져 촬영을 중단하고 임시 사진을 정리했습니다.'); } });
  $('fixture-banner').hidden = !adapters.fixture;
}
if (typeof document !== 'undefined') mount();
