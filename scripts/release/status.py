"""Read-only artifact/process status. Standard library only; never imports workers.

Usage: .venv/Scripts/python.exe scripts/release/status.py [--no-processes]
Presence and recorded completion are not a deployment validation or GPU health test.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
WORKERS = {
    'scripts/model_selection_train.py': 'comparison trainer',
    'scripts/model_selection_evaluate.py': 'comparison evaluation',
    'scripts/release/run_training_queue.py': 'training queue',
    'scripts/release/finish_training_inputs.py': 'input preparation',
    'scripts/release/finish_project.py': 'completion runner',
    'scripts/release/calibrate_depth.py': 'depth calibration',
    'scripts/release/train_models.py': 'release training',
    'scripts/release/annotate_marks_local.py': 'local annotation',
    'scripts/release/prepare_crops.py': 'crop preparation',
    'scripts/release/serve.py': 'release service',
    'scripts/release/verify_service.py': 'HTTP verification',
    'scripts/release/build_report.py': 'report builder',
}


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        return (value, None) if isinstance(value, dict) else ({}, 'not an object')
    except FileNotFoundError:
        return {}, 'missing'
    except (OSError, ValueError) as exc:
        return {}, type(exc).__name__


def present(path):
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def comparison(root):
    directory = root/'models/selection-20261014'
    protocol, error = read_json(directory/'protocol.json')
    if error:
        return ['Comparison: unknown (protocol '+error+')']
    try:
        digest = hashlib.sha256((directory/'protocol.json').read_bytes()).hexdigest()
        arches, seeds, parts = (protocol[k] for k in ('architectures','seeds','parts'))
        expected = len(arches)*len(seeds)*len(parts)
    except (KeyError, TypeError, OSError):
        return ['Comparison: unknown (invalid/unreadable protocol)']
    completed = Counter(); unfinished = []; invalid = 0
    for arch in arches:
        for seed in seeds:
            for part in parts:
                base = directory/arch/str(seed)/part
                record, error = read_json(base.with_suffix('.json'))
                meta = record.get('meta', {})
                done = (not error and isinstance(meta, dict) and meta.get('arch') == arch
                        and meta.get('seed') == seed and meta.get('part') == part
                        and meta.get('protocol_sha256') == digest
                        and type(meta.get('selected_epoch')) is int and meta['selected_epoch'] > 0
                        and present(base.with_suffix('.pt')))
                if done:
                    completed[arch] += 1
                else:
                    if error != 'missing': invalid += 1
                    progress_path = base.with_name(part+'_progress.json')
                    progress, _ = read_json(progress_path)
                    history = progress.get('history')
                    if isinstance(history, list) and history and isinstance(history[-1], dict):
                        try:
                            unfinished.append((progress_path.stat().st_mtime, arch, seed, part,
                                               history[-1].get('epoch', '?')))
                        except OSError:
                            pass
    breakdown = ', '.join(f'{arch}={completed[arch]}/{len(seeds)*len(parts)}' for arch in arches)
    lines = [f'Comparison: {sum(completed.values())}/{expected} completed checkpoint+metadata pairs ({breakdown})']
    if invalid: lines.append(f'  Incomplete/invalid completion records: {invalid} (not counted)')
    if unfinished:
        stamp, arch, seed, part, epoch = max(unfinished)
        age = max(0, int(datetime.now(timezone.utc).timestamp()-stamp))
        lines.append(f'  Last unfinished progress: {arch}/{seed}/{part} epoch {epoch}, {age}s ago (not liveness proof)')
    log = root/'outputs/model-selection-20261014/train.log'
    try:
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size-65536))
            ended = b'ALL_TRAINING_COMPLETE' in stream.read()
        lines.append('  Training end marker: '+('seen' if ended else 'not seen in log tail'))
    except OSError:
        lines.append('  Training end marker: log unavailable')
    return lines


def windows_processes():
    """Read Win32_Process once. Command lines are used for matching, never printed."""
    if os.name != 'nt':
        return None, 'use .venv/Scripts/python.exe for Windows process inspection'
    command = ("$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[Text.UTF8Encoding]::new(); "
               "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,ExecutablePath,CommandLine "
               "| ConvertTo-Json -Compress")
    try:
        result = subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',command],
                                capture_output=True, timeout=8, check=True,
                                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        rows = json.loads(result.stdout.decode('utf-8-sig'))
        if isinstance(rows, dict): rows = [rows]
        if not isinstance(rows, list): raise ValueError('invalid process response')
        return rows, None
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return None, type(exc).__name__  # no stderr/command-line disclosure


def process_lines(root, states, rows, error):
    if rows is None:
        return ['Processes: unknown ('+str(error)+'); artifact status above is recorded only']
    live = {int(p['ProcessId']):p for p in rows}
    observed = []
    for p in rows:
        if not str(p.get('Name','')).lower().startswith(('python','pythonw')): continue
        command = str(p.get('CommandLine') or '').replace('\\','/').lower()
        for script, label in WORKERS.items():
            if re.search(r'(?:^|[\s"/])'+re.escape(script)+r'(?:[\s"]|$)',command):
                executable = str(p.get('ExecutablePath') or '').replace('\\','/').lower()
                scope = '' if root.as_posix().lower() in command+' '+executable else '; project path unconfirmed'
                observed.append(f"  {label}: PID {p['ProcessId']} (parent {p['ParentProcessId']}{scope})")
                break
    recorded = []
    documents = list(states)
    for directory in (root/'outputs/release',root/'outputs/model-selection-20261014'):
        for path in sorted(set(directory.glob('*process*.json')) | set(directory.glob('*pid*.json'))):
            value, error = read_json(path)
            if not error: documents.append((path.name, value))
        for path in sorted(directory.glob('*.pid')):
            try: recorded.append((path.name, int(path.read_text().strip())))
            except (OSError, ValueError): pass
    for name, value in documents:
        for key, pid in value.items():
            if (key == 'pid' or key.endswith('_pid')) and type(pid) is int and pid > 0:
                recorded.append((name+':'+key,pid))
    lines = ['Processes (read-only Windows snapshot; parent/child PIDs may be one job):']
    lines.extend(observed or ['  No known worker script observed; not proof of job completion'])
    if not recorded: lines.append('  Recorded release/comparison PIDs: none')
    for name, pid in sorted(set(recorded)):
        state = ('exists as '+str(live[pid].get('Name'))+'; PID identity not guaranteed') if pid in live else 'not found (stale/exited record)'
        lines.append(f'  Recorded {name}={pid}: {state}')
    return lines


def render(root=ROOT, probe=True):
    root = Path(root)
    lines = ['FaceTeller status '+datetime.now(timezone.utc).isoformat(timespec='seconds')]
    lines.extend(comparison(root)); states = []
    for label, filename in [('Queue','training_queue.json'),('Input prep','input_preparation.json'),
                            ('Completion','completion_queue.json')]:
        value, error = read_json(root/'outputs/release'/filename)
        states.append((filename,value))
        lines.append(f'{label}: '+(error if error else
            f"{value.get('stage','?')} / {value.get('status','?')} (recorded {value.get('updated_utc',value.get('utc','?'))})"))
    crops, error = read_json(root/'data/derived/release/crop_manifest.json')
    people = crops.get('participants',[])
    if not error and isinstance(people,list):
        counts = [p.get('ready_captures',0) for p in people if isinstance(p,dict)]
        count = sum(n for n in counts if type(n) is int)
        lines.append(f"Crop manifest: {len(people)} people, {count} ready captures, complete={crops.get('complete')}")
    for label, file in [('Automatic-label readiness','extended_ready.json'),('Final corpus marker','literature_ready.json')]:
        marker, error = read_json(root/'data/release'/file)
        detail = error or f"ready={marker.get('ready')} (recorded; not independently validated here)"
        if file == 'literature_ready.json':
            if not error and marker.get('ready') is True:
                corpus, corpus_error = read_json(root/'data/release/literature/reference_corpus_v1.json')
                try:
                    digest = hashlib.sha256(json.dumps(corpus,ensure_ascii=False,sort_keys=True,
                            separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()
                    detail = ('matches current canonical corpus; DB not checked' if not corpus_error
                              and marker.get('corpus_sha256') == digest else 'MISMATCH/unreadable corpus; not ready')
                except (TypeError,ValueError): detail = 'invalid corpus; not ready'
            else: detail += '; live corpus drafts are not final'
        lines.append(label+': '+detail)
    groups = [('Comparison artifacts','outputs/model-selection-20261014',
               ['choice_frozen.json','results.json']),
              ('Release artifacts','', ['data/release/depth_config.json','models/release/deployment.json',
                                        'outputs/release/test_metrics.json']),
              ('Final artifacts','outputs/release',['service-verification.json','score-influence.json',
                'FaceTeller_전체구현_실험보고서.pdf','FaceTeller_전체구현_실험보고서.xlsx'])]
    for label, folder, files in groups:
        lines.append(label+' (presence only): '+', '.join(Path(f).name+'='+
                     ('present' if present(root/folder/f) else 'missing') for f in files))
    verification, error = read_json(root/'outputs/release/service-verification.json')
    if not error:
        participants = verification.get('participants',[])
        lines.append(f"HTTP verification: {len(participants)} records; complete={verification.get('complete')}, "
                     f"all400={verification.get('all400')} (recorded; browser acquisition separate)")
    if probe:
        rows, error = windows_processes(); lines.extend(process_lines(root,states,rows,error))
    else: lines.append('Processes: skipped (--no-processes)')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-processes',action='store_true',help='Only read local artifacts')
    args = parser.parse_args()
    print(render(probe=not args.no_processes))


if __name__ == '__main__':
    main()
