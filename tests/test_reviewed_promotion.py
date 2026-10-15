"""Promotion must preserve the previous bundle on a failed startup check."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.release.export_reviewed import activate, sha


def candidate(tmp_path):
    path=tmp_path/'experiments/candidate';stage=path/'staged-release'
    relative='models/release/deployment.json'
    original=tmp_path/relative;replacement=stage/relative
    original.parent.mkdir(parents=True);replacement.parent.mkdir(parents=True)
    original.write_text('original',encoding='utf8');replacement.write_text('replacement',encoding='utf8')
    (path/'test_metrics.json').write_text('{}',encoding='utf8')
    receipt={'status':'staged_not_activated','files':{relative:sha(replacement)},
             'baseline_files':{relative:sha(original)},'calibrated_heads':1}
    (path/'staging_receipt.json').write_text(json.dumps(receipt),encoding='utf8')
    return path,original


def socket_mock():
    return patch('scripts.release.export_reviewed.socket.socket')


def test_failure_after_replacement_restores_original(tmp_path):
    path,original=candidate(tmp_path)
    with socket_mock() as sock, patch('facework.release_runtime.validate_deployment',side_effect=[None,RuntimeError('bad bundle')]):
        sock.return_value.__enter__.return_value.connect_ex.return_value=1
        with pytest.raises(RuntimeError,match='bad bundle'):activate(path,tmp_path)
    assert original.read_text()=='original'
    assert not (path/'activation_receipt.json').exists()
    assert list(path.glob('baseline-before-promotion-*/models/release/deployment.json'))


def test_running_service_prevents_replacement(tmp_path):
    path,original=candidate(tmp_path)
    with socket_mock() as sock:
        sock.return_value.__enter__.return_value.connect_ex.return_value=0
        with pytest.raises(RuntimeError,match='Stop the local'):activate(path,tmp_path)
    assert original.read_text()=='original'


def test_changed_baseline_cannot_be_overwritten(tmp_path):
    path,original=candidate(tmp_path);original.write_text('other change',encoding='utf8')
    with socket_mock() as sock:
        sock.return_value.__enter__.return_value.connect_ex.return_value=1
        with pytest.raises(RuntimeError,match='baseline changed'):activate(path,tmp_path)
    assert original.read_text()=='other change'


def test_receipt_write_failure_also_restores_original(tmp_path):
    path,original=candidate(tmp_path)
    with socket_mock() as sock, patch('facework.release_runtime.validate_deployment'), patch(
            'scripts.release.export_reviewed.write',side_effect=OSError('disk full')):
        sock.return_value.__enter__.return_value.connect_ex.return_value=1
        with pytest.raises(OSError,match='disk full'):activate(path,tmp_path)
    assert original.read_text()=='original'
    assert not (path/'activation_receipt.json').exists()
