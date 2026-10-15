"""Observe the owned Windows server during HTTP400; never open request images."""
import ctypes
from ctypes import wintypes
import datetime
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/release'


class Memory(ctypes.Structure):
    _fields_=[('cb',wintypes.DWORD),('page_faults',wintypes.DWORD)]+[
        (name,ctypes.c_size_t) for name in ('peak_working_set','working_set','peak_paged','paged',
                                         'peak_nonpaged','nonpaged','pagefile','peak_pagefile','private_bytes')]


def main():
    path=OUT/'server-resource-observations.json'
    if path.exists():raise RuntimeError('Existing resource observations preserved')
    record=json.loads((OUT/'server-process.json').read_text())
    # Windows venv python.exe is a launcher; the model process is its child.
    children=json.loads(subprocess.check_output([
        'powershell.exe','-NoProfile','-Command',
        f"@(Get-CimInstance Win32_Process -Filter 'ParentProcessId={int(record['pid'])}' | Select-Object ProcessId,CommandLine) | ConvertTo-Json -Compress"],text=True))
    if isinstance(children,dict):children=[children]
    candidates=[p for p in children if p['CommandLine'].endswith('scripts/release/serve.py --port 8767')]
    if len(candidates)!=1:raise RuntimeError('Cannot identify the unique owned Python model process')
    model_pid=candidates[0]['ProcessId']
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD)
    kernel.CloseHandle.argtypes=(wintypes.HANDLE,)
    handle=kernel.OpenProcess(0x0400|0x0010,False,model_pid)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    psapi=ctypes.WinDLL('psapi',use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes=(wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD)
    proof={'scope':'Windows process private bytes and working set during actual HTTP400; not GPU memory or a proof of forensic erasure',
           'observed_process_role':'Python model process, verified child of the owned venv launcher',
           'samples':[],'complete':False}
    try:
        while True:
            memory=Memory();memory.cb=ctypes.sizeof(memory)
            if not psapi.GetProcessMemoryInfo(handle,ctypes.byref(memory),memory.cb):
                raise ctypes.WinError(ctypes.get_last_error())
            state=json.loads((OUT/'completion_queue.json').read_text())
            service_path=OUT/'service-verification.json'
            service=json.loads(service_path.read_text()) if service_path.exists() else {}
            proof['samples'].append({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'participants_processed':len(service.get('participants',[])),
                'working_set_mib':memory.working_set/2**20,'private_mib':memory.private_bytes/2**20,
                'stage':state['stage']})
            if state['stage']=='actual-verification' and state['status'] in ('complete','failed'):
                proof['complete']=state['status']=='complete';proof['completion_status']=state['status']
            temp=path.with_suffix('.tmp');temp.write_text(json.dumps(proof,indent=2));temp.replace(path)
            if 'completion_status' in proof:break
            time.sleep(15)
    finally:kernel.CloseHandle(handle)


if __name__=='__main__':main()
