#!/usr/bin/env python3
"""Re-analysis, once, of the realism run after its stopped must-hold (30 Oct 2026; Daya's OK: "Diagnose, then re-analyse").

The registered analysis (realism_check.analyse_mode) stopped because its landing-excess must-hold failed in 17 cells
of H2a, H2b and ALL. This script runs only if the diagnostic replay (DIAG_DECOVER.json, realism_diag_decover.py)
says 'confirmed': every flagged event is a handover that started and failed too late in the same slot, which the
harness then charged to the re-establishment station, and the excess recounted on the decision's own target is 0.

It runs the registered analysis function on the stored cells with one line changed: the landing-excess must-hold
exempts H2a, H2b and ALL, as it already exempts C2a and C2b (a target changed after the decision). Everything else is
the frozen code: the freeze check, the pairing, the other must-holds, the tests, the intervals and the bootstrap seeds
from 14000. The outputs go to reanalysis/ next to this script (REAL_RESULT.json, REAL_RESULT.txt, STATUS.txt and
REANALYSIS.json). The stopped REAL_RESULT.json of the registered run stays where it is. It changes no frozen file and no
stored cell.
"""
import os
import sys
import json
import time
import hashlib
import inspect

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(HERE, '..', 'code'))
sys.path.insert(0, CODE)
import realism_check as RC

DIAG = os.path.join(HERE, 'DIAG_DECOVER.json')
OUT_RE = os.path.join(HERE, 'reanalysis')
REAL = os.path.join(CODE, 'models', 'realism_check')
CELLS = os.path.join(REAL, 'cells')
STOPPED_RESULT_MD5 = '20b00d61fe02132f13d52dbf2f5caf68'
MANIFEST = os.path.join(HERE, 'REALISM_CELLS_MD5_20261030.txt')
MANIFEST_MD5 = '704b7e84cf64ccbfa04c17c18f7567ec'
OLD = "        if key[0] not in ('C2a', 'C2b') and int(d.dec_over.sum()) != 0:\n"
NEW = "        if key[0] not in ('C2a', 'C2b', 'H2a', 'H2b', 'ALL') and int(d.dec_over.sum()) != 0:\n"


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    res_path = os.path.join(OUT_RE, 'REAL_RESULT.json')
    assert not os.path.exists(res_path), 'the re-analysis already ran: it runs once'
    diag = json.load(open(DIAG))
    assert diag['summary']['confirmed'] is True, 'the diagnostic replay did not confirm the cause: no re-analysis'
    assert md5(os.path.join(REAL, 'REAL_RESULT.json')) == STOPPED_RESULT_MD5, 'the stopped REAL_RESULT.json changed'
    assert md5(MANIFEST) == MANIFEST_MD5, 'the cell manifest changed'
    want = {}
    for line in open(MANIFEST, encoding='utf-8'):
        h, name = line.split()
        want[name] = h
    assert len(want) == 163
    bad = [n for n, h in want.items() if md5(os.path.join(CELLS, n)) != h]
    assert not bad, 'stored cells changed: ' + ', '.join(bad[:10])
    src = inspect.getsource(RC.analyse_mode)
    assert src.count(OLD) == 1, 'the must-hold line was not found exactly once'
    patched = src.replace(OLD, NEW)
    assert patched != src and patched.replace(NEW, OLD) == src, 'more than the one line would change'
    rec = dict(started=time.strftime('%Y-%m-%d %H:%M:%S'), diag_md5=md5(DIAG), this=md5(os.path.abspath(__file__)),
               realism_check=md5(RC.__file__), line_registered=OLD.strip(), line_used=NEW.strip(),
               why='DIAG_DECOVER.json confirmed: every flagged event in H2a, H2b and ALL is a handover moved by a '
                   'same-slot too-late re-establishment; recounted on the decision\'s own target the excess is 0')
    os.makedirs(OUT_RE, exist_ok=True)
    RC.OUT = OUT_RE

    def cell_path_stored(w, c, arm, A):
        return os.path.join(CELLS, RC.cell_name(w, c, arm, A) + '.csv')

    RC.cell_path = cell_path_stored
    exec(compile(patched, RC.__file__, 'exec'), RC.__dict__)
    RC.analyse_mode()
    res = json.load(open(res_path))
    rec.update(written=time.strftime('%Y-%m-%d %H:%M:%S'), ok=res.get('ok'), problems=res.get('problems'),
               result_json_md5=md5(res_path), result_txt_md5=md5(os.path.join(OUT_RE, 'REAL_RESULT.txt'))
               if os.path.exists(os.path.join(OUT_RE, 'REAL_RESULT.txt')) else None,
               last_boot_seed=res.get('last_boot_seed'))
    json.dump(rec, open(os.path.join(OUT_RE, 'REANALYSIS.json'), 'w'), indent=1)
    print(json.dumps({k: rec[k] for k in ('ok', 'problems', 'result_json_md5', 'last_boot_seed')}), flush=True)


if __name__ == '__main__':
    main()
