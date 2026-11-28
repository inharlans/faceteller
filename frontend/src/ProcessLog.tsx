import {useEffect, useRef, type ReactNode} from 'react';
import {BEFORE, logView, type Plan} from './processSteps';

/** How the account is drawn (2026-11-28). typing: the newest line is written out letter by letter. collapse: a finished
 *  stage keeps only its title. card: a box around the account. theme: the
 *  build of the stages and lines (paper: numbered badges; timeline: a rail with a node per stage; terminal: a system
 *  log with [3/6] tags; agent: a command-line agent's transcript with ● steps and ⎿ results). tone: light or dark (dark
 *  needs the box). marker: what stands before a line. emphasis: how numbers are set off. result: the class after "→" as
 *  a tag. chrome: the heading, a window bar, or nothing. status: a running line at the bottom (spinner, stage, real
 *  elapsed time, overall percentage). dim: finished stages fade. */
export interface LogLook {typing: boolean; collapse: boolean; card: boolean; theme: 'paper' | 'timeline' | 'terminal' | 'agent'; tone: 'light' | 'dark'; font: 'sans' | 'mono' | 'serif';
  marker: 'dot' | 'arrow' | 'check' | 'number' | 'none'; emphasis: 'bold' | 'color' | 'chip' | 'none'; result: boolean; chrome: 'title' | 'window' | 'none'; status: boolean;
  cursor: 'block' | 'bar' | 'underline'; dim: boolean; density: 'tight' | 'normal' | 'loose'}
export const LOOK_DEFAULTS: LogLook = {typing: true, collapse: false, card: true, theme: 'paper', tone: 'light', font: 'sans', marker: 'dot', emphasis: 'bold', result: false, chrome: 'title', status: false,
  cursor: 'block', dim: false, density: 'normal'};
/** Ready-made combinations (the example page offers them; the app uses LOOK_DEFAULTS). */
export const LOOKS: Record<string, LogLook> = {
  paper: LOOK_DEFAULTS,
  bare: {...LOOK_DEFAULTS, card: false, emphasis: 'color'},
  timeline: {...LOOK_DEFAULTS, theme: 'timeline', marker: 'none', emphasis: 'chip', result: true, dim: true},
  terminal: {...LOOK_DEFAULTS, theme: 'terminal', tone: 'dark', font: 'mono', marker: 'number', emphasis: 'color', result: true, chrome: 'window', status: true, density: 'tight'},
  log: {...LOOK_DEFAULTS, theme: 'terminal', tone: 'light', font: 'mono', marker: 'arrow', emphasis: 'color', result: true, chrome: 'title', status: true, density: 'tight'},
  agent: {...LOOK_DEFAULTS, theme: 'agent', tone: 'dark', font: 'mono', marker: 'none', emphasis: 'color', result: false, chrome: 'none', status: true, dim: true},
  agentLight: {...LOOK_DEFAULTS, theme: 'agent', tone: 'light', font: 'mono', marker: 'none', emphasis: 'color', result: false, chrome: 'none', status: true, card: false},
};
const TYPED_BY = .6;   // the newest line is fully written after this part of its own time; the rest is for reading it
const calm = () => typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
const SPIN = {agent: ['✢', '✳', '✶', '✻', '✽', '✻', '✶', '✳'], other: ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']};

type Token = {t: string; k: 'text' | 'num' | 'res' | 'lab'};
/** A line in pieces: the name before the first colon of a figure line, the numbers, and the class after the arrow. */
export function lineTokens(text: string, live: boolean): Token[] {
  const out: Token[] = [], arrow = text.lastIndexOf(' → '), body = arrow < 0 ? text : text.slice(0, arrow + 3), colon = live ? body.indexOf(': ') : -1;
  const plain = (s: string) => {let at = 0; for (const m of s.matchAll(/−?\d+(?:\.\d+)?°?/g)) {if (m.index > at) out.push({t: s.slice(at, m.index), k: 'text'}); out.push({t: m[0], k: 'num'}); at = m.index + m[0].length;}
    if (at < s.length) out.push({t: s.slice(at), k: 'text'});};
  if (colon > 0 && colon <= 16) {out.push({t: body.slice(0, colon), k: 'lab'}); plain(body.slice(colon));} else plain(body);
  if (arrow >= 0) out.push({t: text.slice(arrow + 3), k: 'res'});
  return out;
}
/** The first `cut` letters of a line as marked-up pieces, the rest kept in place but unseen (so the line does not reflow while it is written). */
function Line({text, live, cut, look}: {text: string; live: boolean; cut: number; look: LogLook}) {
  const shown: ReactNode[] = []; let left = cut, rest = '';
  lineTokens(text, live).forEach((tok, i) => {
    const part = tok.t.slice(0, Math.max(0, left)); rest += tok.t.slice(part.length); left -= tok.t.length;
    if (!part) return;
    if (tok.k === 'num' && look.emphasis !== 'none') shown.push(<b key={i} className="num">{part}</b>);
    else if (tok.k === 'res' && look.result) shown.push(<span key={i} className={`res${part.startsWith('보류') ? ' hold' : ''}`}>{part}</span>);
    else if (tok.k === 'lab') shown.push(<span key={i} className="lab">{part}</span>);
    else shown.push(part);
  });
  return <><span>{shown}</span>{rest && <span className="rest">{rest}</span>}</>;
}

/** The written account beside the gauges: what the server does at each stage, opened line by line on the clock of the
 *  gauges (processSteps.ts). Scrolls to the newest line unless the reader has scrolled back. elapsed (ms since the
 *  request, the real time) and overall (percent) are shown in the status line. */
export function ProcessLog({plan, fraction, look = LOOK_DEFAULTS, elapsed = 0, overall = 0}: {plan: Plan; fraction: number; look?: LogLook; elapsed?: number; overall?: number}) {
  const {blocks, newest} = logView(plan, fraction), box = useRef<HTMLDivElement>(null), follow = useRef(true), touched = useRef(0);
  const shown = blocks.reduce((n, b) => n + b.lines.length, 0), started = blocks.filter(b => b.state !== 'waiting'), typing = look.typing && newest < TYPED_BY && !calm();
  useEffect(() => {const el = box.current; if (el && follow.current) el.scrollTop = el.scrollHeight;}, [shown, look.collapse, look.theme, look.density, look.font]);
  // Keep the newest line in view when the box itself changes size (window, layout).
  useEffect(() => {const el = box.current; if (!el || typeof ResizeObserver !== 'function') return;
    const watch = new ResizeObserver(() => {if (follow.current) el.scrollTop = el.scrollHeight;}); watch.observe(el); return () => watch.disconnect();}, []);
  // Only the reader's own scrolling stops the following; the box's own smooth scrolling passes through the same event.
  const byReader = () => {touched.current = Date.now();};
  const dark = look.tone === 'dark' && look.card, running = [...started].reverse().find(b => b.state === 'active'), finished = fraction >= 1, frames = look.theme === 'agent' ? SPIN.agent : SPIN.other;
  const cls = ['process-log', `theme-${look.theme}`, dark ? 'tone-dark' : 'tone-light', `font-${look.font}`, `marker-${look.marker}`, `emph-${look.emphasis}`, `cursor-${look.cursor}`, `dense-${look.density}`, look.card ? '' : 'no-card', look.dim ? 'dim' : ''].filter(Boolean).join(' ');
  let n = BEFORE.lines.length;
  const two = (k: number) => String(k).padStart(2, '0');
  return <aside className={cls} aria-labelledby={look.chrome === 'title' ? 'process-log-title' : undefined} aria-label={look.chrome === 'title' ? undefined : '처리 과정'}>
    {look.chrome === 'title' && <header><h2 id="process-log-title">처리 과정</h2></header>}
    {look.chrome === 'window' && <header className="window" aria-hidden="true"><i/><i/><i/><span>faceteller · 처리 과정</span></header>}
    <div className="process-scroll" ref={box} role="log" aria-label="분석 처리 과정" tabIndex={0} onWheel={byReader} onTouchMove={byReader} onPointerDown={byReader} onKeyDown={byReader}
      onScroll={e => {const el = e.currentTarget; if (el.scrollHeight - el.scrollTop - el.clientHeight < 24) follow.current = true; else if (Date.now() - touched.current < 1500) follow.current = false;}}>
      <section className="done"><h3><span className="mark" data-n="0" data-of={plan.blocks.length} aria-hidden="true"/>{BEFORE.title}</h3>{!look.collapse && <ul>{BEFORE.lines.map((t, k) => <li key={t} data-n={two(k + 1)}><Line text={t} live={false} cut={t.length} look={look}/></li>)}</ul>}</section>
      {started.map((b, i) => {const folded = look.collapse && b.state === 'done', last = i === started.length - 1, first = n; n += b.lines.length;
        return <section key={b.title} className={b.state}><h3><span className="mark" data-n={i + 1} data-of={plan.blocks.length} aria-hidden="true"/>{b.title}{folded && <small>{b.lines.length}줄</small>}</h3>
          {!folded && <ul>{b.lines.map((l, k) => {const cut = typing && last && k === b.lines.length - 1 ? Math.max(1, Math.ceil(l.text.length * newest / TYPED_BY)) : l.text.length;
            return <li key={k} data-n={two(first + k + 1)} className={[l.live ? 'live' : '', cut < l.text.length ? 'typing' : ''].join(' ').trim() || undefined}><Line text={l.text} live={l.live} cut={cut} look={look}/></li>;})}</ul>}</section>;})}
    </div>
    {look.status && <footer className="process-status" aria-hidden="true"><span className="spin">{finished ? '✓' : frames[Math.floor(elapsed / 110) % frames.length]}</span>
      <span>{finished ? '완료' : running ? `${running.title} 중…` : '준비 중…'}</span><span className="meter"><i style={{width: `${overall}%`}}/></span><span className="count">{Math.floor(elapsed / 1000)}초 · {overall}%</span></footer>}
  </aside>;
}
