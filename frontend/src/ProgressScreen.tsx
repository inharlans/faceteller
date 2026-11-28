import type {CSSProperties} from 'react';
import {LOOK_DEFAULTS, ProcessLog, type LogLook} from './ProcessLog';
import type {Plan} from './processSteps';

/** How the progress screen is laid out (2026-11-28). side: the written account to the right of the gauges or below
 *  them. head: the title above both, or inside the gauges' box. card: the gauges in a box like the account's. rows:
 *  a thick bar under each stage name, or name, bar and percentage on one line. columns: gauges per row. split: the
 *  gauges' share of the width in percent (side by side only). width: the whole screen, height: the account's text
 *  area, gap: between the two, bar: the gauge's thickness (px). scale: size of the gauges' type and badges. font: the
 *  account's type size (px). */
export interface ScreenLayout {side: 'right' | 'below'; head: 'top' | 'inside'; card: boolean; rows: 'stack' | 'inline'; columns: 1 | 2 | 3; split: number; width: number; height: number; gap: number;
  bar: number; scale: number; font: number; spinner: boolean; status: boolean; percent: boolean; overall: boolean}
export const LAYOUT_SIDE: ScreenLayout = {side: 'right', head: 'top', card: true, rows: 'stack', columns: 1, split: 42, width: 1040, height: 400, gap: 28, bar: 10, scale: 1.1, font: 13.5,
  spinner: true, status: true, percent: true, overall: true};
// The layout confirmed on 2026-11-28 (with the default look and content of the account): the account under the gauges,
// three gauges to a row, no line of text for the running stage under the title.
export const LAYOUT_BELOW: ScreenLayout = {...LAYOUT_SIDE, side: 'below', columns: 3, width: 860, height: 300, gap: 20, bar: 8, scale: 1, status: false};
export const LAYOUT_DEFAULTS = LAYOUT_BELOW;

/** The whole progress screen: title, the gauges of the stages, the written account (ProcessLog) and the cancel button.
 *  paced: the running stage (stages.length when all are done); percents: one gauge per stage; elapsed: ms since the request. */
export function ProgressScreen({stages, paced, percents, status, plan, fraction, elapsed = 0, look = LOOK_DEFAULTS, layout = LAYOUT_DEFAULTS, onCancel}:
  {stages: readonly string[]; paced: number; percents: readonly number[]; status: string; plan: Plan; fraction: number; elapsed?: number; look?: LogLook; layout?: ScreenLayout; onCancel: () => void}) {
  const l = layout, overall = Math.round(stages.reduce((sum, _, i) => sum + (percents[i] ?? 0) * (plan.bounds[i + 1] - plan.bounds[i]), 0));
  const vars = {'--ps-w': `${l.width}px`, '--ps-gap': `${l.gap}px`, '--ps-columns': `minmax(0,${l.split}fr) minmax(0,${100 - l.split}fr)`, '--ps-cols': l.columns, '--ps-bar': `${l.bar}px`, '--ps-scale': l.scale,
    '--log-h': `${l.height}px`, '--log-fs': `${l.font}px`} as CSSProperties;
  const head = <header className="progress-head">{l.spinner && <div className="spinner" aria-hidden="true"/>}<h1 id="progress-title">얼굴 형태와 문헌을 확인하고 있습니다</h1>
    <p role="status" aria-live="polite" hidden={!l.status}>{status}</p></header>;
  return <section className={`progress-screen center ${l.side === 'right' ? 'side' : 'below'} rows-${l.rows}${l.percent ? '' : ' no-percent'}`} style={vars} aria-labelledby="progress-title">
    {l.head === 'top' && head}
    <div className="progress-body">
      <div className={`progress-main${l.card ? ' card' : ''}`}>
        {l.head === 'inside' && head}
        {l.card && <header className="progress-main-head"><h2>진행 단계</h2>{l.overall && <span>전체 {overall}%</span>}</header>}
        <ol className="progress-list">{stages.map((s, i) => {const p = percents[i] ?? 0;
          return <li key={s} className={paced > i ? 'done' : paced === i ? 'active' : ''}><span>{paced > i ? '✓' : i + 1}</span><em>{s}</em><i className="gauge" aria-hidden="true"><b style={{width: `${p}%`}}/></i><small>{p}%</small></li>;})}</ol>
      </div>
      <ProcessLog plan={plan} fraction={fraction} look={look} elapsed={elapsed} overall={overall}/>
    </div>
    <button className="button ghost" type="button" onClick={onCancel}>분석 취소</button>
  </section>;
}
