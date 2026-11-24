import {NOTICE, safeUrl, type ReportEntry, type ReportJudgement, type ReportLine, type ReportMeasure, type ReportPassage, type ReportSource, type ResearchReport} from './types';

// TEMPORARY research view (2026-11-16): the tables of deploy/v1/run_person.py, as sent by the server.
// Nothing is computed or reworded here; the page only lays the report out.
const VALENCE: Record<string, string> = {좋음: 'favorable', 나쁨: 'unfavorable', 섞임: 'mixed', 중립: 'neutral'};
const two = (n: number | null | undefined) => typeof n === 'number' && Number.isFinite(n) ? n.toFixed(2) : '–';
/** A measured value in its unit: degrees with one decimal, scores with two, ratios with three. */
export const withUnit = (n: number | null | undefined, unit: string) => typeof n !== 'number' || !Number.isFinite(n) ? '–' : unit === '도' ? `${n.toFixed(1)}°` : unit === '점수' ? n.toFixed(2) : n.toFixed(3);
/** "F17 턱 폭:좁음" -> "턱 폭: 좁음" (the feature code is internal). */
export const conditionText = (requirement: string[]) => requirement.map(c => c.split(' 또는 ').map(alt => alt.split('+').map(a => a.replace(/^F\d+[a-z]?\s+/, '').replace(':', ': ')).join(' 그리고 ')).join(' 또는 ')).join(' + ');
/** Cuts between the classes in words: two cuts name the low and the high side, one cut names the upper class. */
export function cutText(m: ReportMeasure): string {
  const c = m.cuts, k = m.cut_classes;
  if (!c || !k) return '';
  if (c.length === 2 && k.length === 3) return `${withUnit(c[0], m.unit)} 미만 ${k[0]}, ${withUnit(c[1], m.unit)} 이상 ${k[2]}, 그 사이 ${k[1]}`;
  if (c.length === 1 && k.length === 2) return `${withUnit(c[0], m.unit)} 이상 ${k[1]}, 미만 ${k[0]}`;
  return '';
}
/** "Page:Gujin Tushu Jicheng, Volume 473 (1700-1725).djvu/29" -> "고금도서집성 스캔본 473책 29쪽"; other forms are shown as they are. */
export function scanText(page?: string | null, end?: string | null): string {
  if (!page) return '';
  const parse = (s: string) => /Gujin Tushu Jicheng, Volume (\d+).*\.djvu\/(\d+)$/.exec(s), a = parse(page), b = end && end !== page ? parse(end) : null;
  const pdf = /^pdf_page:(\d+)$/.exec(page);
  return a ? `고금도서집성 스캔본 ${a[1]}책 ${a[2]}${b && b[1] === a[1] ? `~${b[2]}` : ''}쪽` : pdf ? `스캔 PDF ${pdf[1]}쪽` : page.replace(/^Page:/, '');
}
const STATUS = (p: {annotation_status?: string | null; translation_kind?: string | null}) => /unreviewed|ai/i.test(`${p.annotation_status || ''} ${p.translation_kind || ''}`) ? '번역과 해설은 자동 작성한 초안이며 사람이 검수하기 전입니다.' : p.annotation_status ? `자료 상태: ${p.annotation_status}` : '';
const Tone = ({value}: {value: string}) => <span className="valence" data-valence={VALENCE[value] || 'neutral'}>{value}</span>;

function SourceLine({s, fallback}: {s?: ReportSource | null; fallback?: string | null}) {
  if (!s) return fallback ? <p className="report-where"><b>출처</b> {fallback}</p> : null;
  const url = safeUrl(s.url), book = [s.title_ko, s.title && s.title !== s.title_ko ? `(${s.title})` : ''].filter(Boolean).join(' ') || fallback || '';
  const span = s.char_span ? `그 절 본문의 ${s.char_span[0]}~${s.char_span[1]}번째 글자` : '', scan = scanText(s.scan_page, s.scan_page_end);
  return <div className="report-where">
    <p><b>출처</b> {book}{s.attributed_to ? ` · 책 안에서 ${s.attributed_to}의 글로 실림` : ''}</p>
    {s.chapter && <p><b>권·편</b> <span lang="zh">{s.chapter}</span></p>}
    {(s.section || span) && <p><b>위치</b> {s.section && <>「<span lang="zh">{s.section}</span>」 절</>}{s.section && span ? ' · ' : ''}{span}</p>}
    {scan && <p><b>스캔</b> {scan}</p>}
    {url && <p><b>원문 보기</b> <a href={url} target="_blank" rel="noopener noreferrer">{/wikisource\.org/.test(url) ? '위키문헌(고정된 판)' : '출처 페이지'}</a></p>}
    {s.text_witness && <p className="fine"><b>저본</b> <span lang="zh">{s.text_witness}</span></p>}
  </div>;
}

/** What was measured for one feature: the judged value with its cuts, or the CNN probabilities with a reference measure. */
function Basis({j}: {j: ReportJudgement}) {
  const m = j.measure, r = j.reference, pr = j.probabilities;
  const parts = (x: ReportMeasure) => x.components?.length ? <ul className="report-parts">{x.components.map(c => <li key={c.label}>{c.label} <b>{withUnit(c.value, c.unit)}</b></li>)}</ul> : null;
  return <div className="report-basis">
    {m && <><p>{m.label}</p>{cutText(m) && <p className="fine">경계: {cutText(m)}</p>}{parts(m)}</>}
    {pr && <p>{Object.entries(pr).map(([k, v]) => `${k} ${two(v)}`).join(' · ')}</p>}
    {r && <><p className="fine">참고 측정(판정에는 쓰지 않음): {r.label} <b>{withUnit(r.value, r.unit)}</b></p>{parts(r)}</>}
    {!m && !pr && !r && '–'}
  </div>;
}

/** The numbers behind one condition, shown on its card: for each feature it uses, the measured value (or CNN probabilities) and what it is. */
function Evidence({p, judgements}: {p: ReportPassage; judgements: Record<string, ReportJudgement>}) {
  const rows = Object.keys(p.features).map(k => ({k, j: judgements[k], use: p.features[k]})).filter(r => r.j);
  if (!rows.length) return null;
  return <div className="report-evidence">{rows.map(({k, j, use}) => <div key={k} className="report-evidence-row">
    <p className="report-evidence-head"><b>{j.name}</b> {j.measure ? <span className="report-number">{withUnit(j.measure.value, j.measure.unit)}</span> : null} <span>→ {use.class}</span> <span className="report-g">신뢰도 {two(use.confidence)}</span></p>
    <Basis j={j}/>
  </div>)}</div>;
}

/** One passage inside a merged line: its own sentence, original text and exact place. */
function EntryDetail({e}: {e: ReportEntry}) {
  const status = STATUS(e);
  return <li className="report-entry">
    <p className="report-entry-main">{e.translation || '번역 없음'} <span className="report-book">[{e.source || e.source_detail?.title_ko || '출처 미상'}]</span></p>
    {e.original_text && <p className="report-original" lang="zh">{e.original_text}</p>}
    {e.commentary && <p className="report-commentary"><b>해설</b> {e.commentary}</p>}
    <SourceLine s={e.source_detail} fallback={e.source}/>
    {e.split_of && <p className="fine">여러 생김새를 함께 다룬 조항의 일부라, 조항 전체를 보여 줍니다.</p>}
    {status && <p className="fine">{status}</p>}
  </li>;
}

/** One kind of statement: a sentence and every book that says the same kind of thing; the passages behind it on demand. */
function StatementLine({l, groupValence}: {l: ReportLine; groupValence: string}) {
  return <li className="report-line">
    <p className="report-line-main">{l.translation || '번역 없음'} <span className="report-book">[{l.sources.join(', ')}]</span>{l.valence !== groupValence && <Tone value={l.valence}/>}</p>
    <details className="report-entry-more"><summary>{l.entries.length > 1 ? `이 내용의 조항 ${l.entries.length}개: 원문과 위치` : '원문과 위치'}</summary><ul className="report-entries">{l.entries.map(e => <EntryDetail key={e.id} e={e}/>)}</ul></details>
  </li>;
}

/** A condition group: what the books say for this shape, one line per kind of statement - the sentence and its books, nothing else
 *  (user, 2026-11-16: no counts or notes around the lines; the report still carries entry_count / hidden_unfavourable / omitted_entries). */
function PassageBlock({p, index, note, judgements}: {p: ReportPassage; index: number; note?: string; judgements: Record<string, ReportJudgement>}) {
  const lines = p.lines || [];
  return <article className="report-passage" data-valence={VALENCE[p.valence] || 'neutral'}>
    <div className="report-passage-head"><span className="rank">{index + 1}</span><b>{conditionText(p.requirement)}</b><Tone value={p.valence}/><span className="report-g">신뢰도 {two(p.G)}</span>{note && <span className="partchip">{note}</span>}</div>
    <Evidence p={p} judgements={judgements}/>
    {lines.length > 0 ? <ul className="report-lines">{lines.map(l => <StatementLine key={l.topic} l={l} groupValence={p.valence}/>)}</ul> : <>
      {p.original_text && <p className="report-original" lang="zh">{p.original_text}</p>}
      <p className="report-translation">{p.translation || '번역 없음'}</p>
      {p.commentary && <p className="report-commentary"><b>해설</b> {p.commentary}</p>}
      <SourceLine s={p.source_detail} fallback={p.source}/>
      {p.split_of && <p className="fine">여러 생김새를 함께 다룬 조항의 일부라, 조항 전체를 보여 줍니다.</p>}
      {STATUS(p) && <p className="fine">{STATUS(p)}</p>}
    </>}
  </article>;
}

function PassageList({rows, note, judgements}: {rows: ReportPassage[]; note?: (p: ReportPassage) => string; judgements: Record<string, ReportJudgement>}) {
  return <div className="report-passages">{rows.map((p, i) => <PassageBlock key={p.representative} p={p} index={i} judgements={judgements} note={note?.(p)}/>)}</div>;
}
// Plan C limits (deploy/v1/config/selection_rules.json): five passages in all, at most two unfavourable ones.
const TOTAL = 5, BAD_MAX = 2;
/** Why a passage whose conditions hold is not among the selected ones. */
export function leftOutReason(p: ReportPassage, selected: ReportPassage[]): string {
  if (p.valence === '나쁨' && selected.filter(s => s.valence === '나쁨').length >= BAD_MAX) return `나쁜 풀이는 ${BAD_MAX}개까지`;
  return selected.length >= TOTAL ? `${TOTAL}개가 먼저 참` : '선정 규칙에서 빠짐';
}

export function ResearchReportView({report, notice, onReset}: {report: ResearchReport; notice?: string; onReset: () => void}) {
  const chosen = new Set(report.plan_C_selected.map(p => p.representative)), judgements = Object.entries(report.judgements);
  const rest = report.all_matching.filter(p => !chosen.has(p.representative));  // the selected ones are shown above; do not repeat them
  const confirmed = judgements.filter(([, j]) => j.status === '확정').length;
  const checks = Object.entries(report.cnn_check).flatMap(([feature, models]) => Object.entries(models).map(([model, c]) => ({feature, model, c})));
  const licenses = [...new Set([...report.all_matching, ...report.blocked_by_hold].flatMap(p => [p.source_detail?.license, ...(p.lines || []).flatMap(l => l.entries.map(e => e.source_detail?.license))]).filter((v): v is string => !!v))];
  return <section className="research-report" aria-labelledby="report-title">
    <div className="res-head"><h1 id="report-title">연구용 상세 결과</h1><span className="partchip">임시 화면</span></div>
    <p className="res-sub">판정 {judgements.length}개 중 확정 {confirmed}개 · 조건이 맞는 조항 {report.all_matching.length}개 · 선정 {report.plan_C_selected.length}개{report.structure ? ` · ${report.structure}` : ''}</p>

    <h2 className="section-title">고른 조항 (C안, 최대 5개)</h2>
    {report.plan_C_selected.length ? <PassageList rows={report.plan_C_selected} judgements={report.judgements}/> : <p className="report-empty">조건이 맞는 조항이 없습니다.</p>}

    <h2 className="section-title">그 밖에 해당하는 조항 ({rest.length}개)</h2>
    {rest.length ? <>
      <p className="res-sub">조건이 맞는 조항은 모두 {report.all_matching.length}개입니다. 위에 고른 {report.plan_C_selected.length}개 말고 아래 {rest.length}개가 더 있습니다.</p>
      <details className="report-fold" open><summary>그 밖의 {rest.length}개</summary><PassageList rows={rest} note={p => leftOutReason(p, report.plan_C_selected)} judgements={report.judgements}/></details>
    </> : <p className="report-empty">{report.all_matching.length ? `조건이 맞는 조항은 위에 고른 ${report.plan_C_selected.length}개가 전부입니다.` : '조건이 맞는 조항이 없습니다.'}</p>}

    <h2 className="section-title">보류 때문에 나오지 못한 조항 ({report.blocked_by_hold.length}개)</h2>
    {report.blocked_by_hold.length ? <PassageList rows={report.blocked_by_hold} judgements={report.judgements}/> : <p className="report-empty">보류 때문에 나오지 못한 조항은 없습니다.</p>}

    <h2 className="section-title">보류된 판정 ({report.held_judgements.length}개)</h2>
    {report.held_judgements.length ? <div className="report-scroll"><table className="report-table"><thead><tr><th>특징</th><th>보류된 판정</th><th>신뢰도</th><th>기준</th><th>조항이 안 나온 이유</th></tr></thead>
      <tbody>{report.held_judgements.map(h => <tr key={h.feature}><td>{h.name}</td><td>{h.class}</td><td className="num">{two(h.confidence)}</td><td className="num">{two(h.threshold)}</td><td>{h.reason}</td></tr>)}</tbody></table></div>
      : <p className="report-empty">보류된 판정이 없습니다.</p>}

    <h2 className="section-title">전체 판정과 측정값 ({judgements.length}개)</h2>
    <div className="report-scroll"><table className="report-table report-judgements"><thead><tr><th>특징</th><th>방법</th><th>측정값</th><th>판정</th><th>신뢰도</th><th>기준</th><th>상태</th><th>무엇을 쟀는지</th></tr></thead>
      <tbody>{judgements.map(([key, j]) => <tr key={key} data-status={j.status}><td>{j.name}<small className="report-region">{j.region}</small></td><td>{j.method || '–'}</td>
        <td className="num">{j.measure ? withUnit(j.measure.value, j.measure.unit) : j.value === null ? '–' : two(j.value)}</td><td>{j.class || '–'}</td>
        <td className="num">{two(j.confidence)}</td><td className="num">{j.threshold === null ? '없음' : two(j.threshold)}</td><td><span className="report-status" data-status={j.status}>{j.status}</span></td><td><Basis j={j}/></td></tr>)}</tbody></table></div>

    <h2 className="section-title">확인용 CNN (조항에는 쓰지 않음)</h2>
    {checks.length ? <div className="report-scroll"><table className="report-table"><thead><tr><th>특징</th><th>구조의 판정</th><th>확인용 모델</th><th>CNN 결과</th><th>CNN 신뢰도</th></tr></thead>
      <tbody>{checks.map(({feature, model, c}) => {const j = report.judgements[feature]; return <tr key={`${feature}-${model}`}><td>{j?.name || feature}</td><td>{j?.class ? `${j.class}${j.status === '확정' ? '' : ` (${j.status})`}` : '–'}</td><td>{model}</td><td>{c.class}</td><td className="num">{two(c.confidence)}</td></tr>;})}</tbody></table></div>
      : <p className="report-empty">확인용 CNN을 돌리지 않았습니다.</p>}

    {report.notes && report.notes.length > 0 && <ul className="report-notes">{report.notes.map(n => <li key={n}>{n}</li>)}</ul>}
    <p className="disclaimer">신뢰도는 판정이 경계에서 얼마나 떨어져 있는지를 나타내는 값이며 정답 확률이 아닙니다. {notice || NOTICE}{licenses.length > 0 && <><br/>문헌 원문의 이용 조건: {licenses.join(' / ')}</>}</p>
    <div className="actions"><button className="button" onClick={onReset}>처음으로</button></div>
  </section>;
}
