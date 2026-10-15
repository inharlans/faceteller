"""Prewritten Korean readings and audited morphology-to-text links.

No text is generated during a request. The archived corpus remains unchanged;
editorial and semantic revisions have separate hashes in the response.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

VALENCES = {'favorable', 'unfavorable', 'mixed', 'neutral'}


def load_reading_catalog(root):
    if (Path(root)/'data/release/gwansang/bindings.json').is_file():
        from facework.release_gwansang import GwansangCatalog
        return GwansangCatalog.load(root)
    return ReadingCatalog.load(root)


class ReadingCatalog:
    def __init__(self, document, corpus, policy):
        self.document = deepcopy(document)
        self.policy = deepcopy(policy)
        self.passages = {p['passage_id']: p for p in corpus['passages']}
        expected = {k for k, p in self.passages.items() if p['service_eligible']}
        readings = document['readings']
        if set(readings) != expected:
            raise ValueError('Editorial coverage must match the archived eligible passages')
        for key, reading in readings.items():
            for field in ('title', 'translation', 'meaning', 'valence_label'):
                if not isinstance(reading.get(field), str) or not reading[field].strip():
                    raise ValueError(f'{key}: missing editorial field {field}')
            if reading['valence'] not in VALENCES or not isinstance(reading['topics'], list):
                raise ValueError(f'{key}: invalid editorial categories')
        self.excluded = set()
        for row in policy.get('excluded_passages', []):
            key = row['passage_id']
            if key not in expected or not row.get('reason'):
                raise ValueError('Unknown semantic exclusion')
            digest = hashlib.sha256(self.passages[key]['original_text'].encode()).hexdigest()
            if row['original_sha256'] != digest:
                raise ValueError(f'{key}: semantic review applies to a different text')
            self.excluded.add(key)
        vocabulary = {label for p in corpus['passages'] for label in p['required_labels']}
        seen = set()
        for kind in document.get('named_types', []):
            if kind['id'] in seen or not kind['required_labels'] or not set(kind['required_labels']) <= vocabulary:
                raise ValueError('Invalid named type definition')
            seen.add(kind['id'])
            if not kind['source_passage_ids'] or any(p not in self.passages for p in kind['source_passage_ids']):
                raise ValueError('Named type must cite an archived source')
            if kind['valence'] not in VALENCES:
                raise ValueError('Invalid named type reading')

    @classmethod
    def load(cls, root):
        base = Path(root)/'data/release/literature'
        paths = [base/'readings_ko_v1.json', base/'reference_corpus_v1.json', base/'semantic_review_v2.json']
        raw = [p.read_bytes() for p in paths]
        result = cls(*(json.loads(b) for b in raw))
        result.identity = {'readings_sha256': hashlib.sha256(raw[0]).hexdigest(),
                           'semantic_review_sha256': hashlib.sha256(raw[2]).hexdigest(),
                           'readings_version': result.document['version'],
                           'semantic_review_version': result.policy['version']}
        return result

    def repository(self, original, *, measurements=None):
        excluded = self.excluded

        class AuditedRepository:
            def load_candidates(self, labels):
                # Filter BEFORE the top-five cap so removed entries are replaced.
                return [p for p in original.load_candidates(labels) if p.passage_id not in excluded]

        return AuditedRepository()

    def apply(self, result):
        for passage in result['passages']:
            reading = self.document['readings'][passage['passage_id']]
            passage['translation'] = reading['translation']
            passage['commentary'] = reading['meaning']
            passage['reading'] = {k: deepcopy(reading[k]) for k in ('title','meaning','valence','valence_label','topics')}
            passage['source']['editorial_version'] = self.document['version']
            passage['source']['editorial_status'] = self.document['authoring']
        confirmed = {i['label_id']: i for i in result['items'] if i['status']=='CONFIRMED'}
        result['named_types'] = []
        for kind in self.document.get('named_types', []):
            if not set(kind['required_labels']) <= confirmed.keys():
                continue
            if any(p in self.excluded for p in kind['source_passage_ids']):
                continue
            value = {k: deepcopy(v) for k,v in kind.items() if k != 'required_labels'}
            value['evidence_item_ids'] = [confirmed[k]['item_id'] for k in kind['required_labels']]
            result['named_types'].append(value)
        result['ranking_policy']['semantic_review_version'] = self.policy['version']
        return result
