"""User-curated readings: one shape, one card, one representative quotation.

The supplied numeric conditions are proposals, not new training labels. Only
explicitly reviewed links to existing observable features are served. The full
source archive stays in PostgreSQL; source count never affects the match score.
"""
from copy import deepcopy
import hashlib
import json
import math
import unicodedata
from pathlib import Path

from facework.pilot_contract import Passage


def digest(document):
    raw = json.dumps(document, ensure_ascii=False, sort_keys=True,
                     separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


class GwansangCatalog:
    def __init__(self, document):
        self.document = deepcopy(document)
        data, bindings = self.document['data'], self.document['bindings']
        self.shapes = {s['생김새코드']: s for s in data['생김새']}
        if len(self.shapes) != len(data['생김새']) or len(self.shapes) != data['생김새수']:
            raise ValueError('Duplicate shape or incorrect shape count')
        self.bindings = bindings['shapes']
        if set(self.shapes) != set(self.bindings):
            raise ValueError('Every shape must have an explicit binding or exclusion')
        source_ids = set()
        self.aliases = {}
        for key, shape in self.shapes.items():
            binding = self.bindings[key]
            if not binding.get('reason') or not isinstance(binding.get('enabled'), bool):
                raise ValueError('Binding needs status and rationale')
            if not shape['종합풀이'].strip() or not shape['생김새'].strip():
                raise ValueError('Missing reading')
            ranks = set()
            for source in shape['출처']:
                sid, rank = source['출처id'], source['순위']
                if sid in source_ids or rank in ranks or type(rank) is not int or rank < 1:
                    raise ValueError('Duplicate source/rank or invalid source rank')
                source_ids.add(sid); ranks.add(rank)
                for field in ('원문', '번역', '책_한글', '편', 'github'):
                    if not isinstance(source[field], str) or not source[field].strip():
                        raise ValueError('Incomplete source')
                if source['적용성별'] not in ('공통', '남', '여'):
                    raise ValueError('Unknown source audience')
            if len(shape['출처']) != shape['출처수']:
                raise ValueError('Incorrect source count')
            labels = binding.get('required_labels', [])
            presentation = binding.get('presentation', {})
            if not isinstance(presentation.get('title'), str) or not presentation['title'].strip():
                raise ValueError('Every shape needs a reviewed display title')
            for field in ('aliases', 'related_types', 'topics'):
                values = presentation.get(field)
                if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
                    raise ValueError('Invalid presentation terms')
                if len({self.normalize_name(v) for v in values}) != len(values):
                    raise ValueError('Duplicate presentation term')
            names = {self.normalize_name(n) for n in [presentation['title'], *presentation['aliases']]}
            if names & {self.normalize_name(n) for n in presentation['related_types']}:
                raise ValueError('A related type is not an alias')
            for name in names:
                if name in self.aliases and self.aliases[name] != key:
                    raise ValueError('Ambiguous shape alias')
                self.aliases[name] = key
            selected = presentation.get('representative_source_id')
            excluded = presentation.get('excluded_sources', {})
            ids = {s['출처id'] for s in shape['출처']}
            if (not isinstance(excluded, dict) or any(not isinstance(v, str) or not v.strip() for v in excluded.values())
                    or selected in excluded or set(excluded) | ({selected} if selected else set()) != ids
                    or not presentation.get('selection_reason')):
                raise ValueError('Every source needs an explicit selection or exclusion')
            if binding['enabled']:
                if not labels or len(set(labels)) != len(labels):
                    raise ValueError('Enabled shape needs distinct required labels')
                if shape['대상성별'] != '공통' or not self.representative(shape):
                    raise ValueError('Do not infer gender from a face')
                if not isinstance(presentation.get('meaning'), str) or not presentation['meaning'].strip() or not presentation['topics']:
                    raise ValueError('Enabled shape needs a source-specific meaning')
            elif selected is not None or presentation.get('meaning') is not None:
                raise ValueError('Excluded shape cannot supply an automatic reading')
            for gate in binding.get('measurement_gates', []):
                if gate['feature'] != 'brow_length' or gate['source'] != 'segmentation' or gate['unit'] != 'ratio':
                    raise ValueError('Unknown measured condition')
                if not isinstance(gate['min'], (int, float)) or not math.isfinite(gate['min']):
                    raise ValueError('Invalid measured boundary')
        if len(source_ids) != data['출처수']:
            raise ValueError('Incorrect total source count')
        self.identity = {
            'reading_catalog_kind': 'gwansang_grouped',
            'gwansang_catalog_sha256': digest(self.document),
            'readings_version': bindings['version'],
            'catalog_shape_count': len(self.shapes),
            'catalog_source_count': len(source_ids),
            'active_mapping_count': sum(b['enabled'] for b in self.bindings.values()),
        }

    @classmethod
    def load(cls, root):
        base = Path(root) / 'data/release/gwansang'
        raw = (base / 'gwansang.json').read_bytes()
        provenance = json.loads((base / 'provenance.json').read_bytes())
        if hashlib.sha256(raw).hexdigest() != provenance['json_sha256']:
            raise ValueError('Supplied source file differs from its recorded provenance')
        result = cls({'data': json.loads(raw),
                      'bindings': json.loads((base / 'bindings.json').read_bytes()),
                      'provenance': provenance})
        # Validate against the shipped schema, not a vocabulary invented by the mapping.
        schemas = ['data/label_schema_kface_v3.json', 'data/release/extended_schema.json']
        vocabulary = set()
        for name in schemas:
            for key, feature in json.loads((Path(root)/name).read_bytes())['features'].items():
                vocabulary.update(f'{key}:{value}' for value in feature['classes'])
        for binding in result.bindings.values():
            if not set(binding.get('required_labels', [])) <= vocabulary:
                raise ValueError('Binding references an unknown trained label')
        return result

    @staticmethod
    def normalize_name(name):
        return ''.join(unicodedata.normalize('NFKC', name).split())

    def shape_for_alias(self, name):
        """Resolve a reviewed name only; related types never create extra cards."""
        return self.aliases.get(self.normalize_name(name))

    def representative(self, shape):
        selected = self.bindings[shape['생김새코드']]['presentation']['representative_source_id']
        return next((s for s in shape['출처'] if s['출처id'] == selected and s['적용성별'] == '공통'), None)

    def verify_snapshot(self, repository):
        stored, recorded_hash = repository.gwansang_catalog()
        expected = self.identity['gwansang_catalog_sha256']
        if recorded_hash != expected or digest(stored) != expected:
            raise ValueError('Active grouped catalog differs from the deployed catalog')

    def candidates(self, labels, *, measurements=None):
        measurements = measurements or {}
        out = []
        for key, shape in self.shapes.items():
            binding = self.bindings[key]
            required = frozenset(binding.get('required_labels', []))
            if not binding['enabled'] or not required <= set(labels):
                continue
            passes = True
            for gate in binding.get('measurement_gates', []):
                actual = measurements.get(gate['feature'], {})
                value = actual.get('value')
                if (actual.get('source') != gate['source'] or actual.get('unit') != gate['unit']
                        or type(value) not in (int, float) or not math.isfinite(value) or value < gate['min']):
                    passes = False
            if not passes:
                continue
            source = self.representative(shape)
            out.append(Passage('GW-'+key, required, source['원문'], source['번역'], source['해설'], {
                'title': source['책_한글'], 'chapter': source['편'], 'locator': f"제{source['조']}조",
                'url': source['github'], 'annotation_status': '제공 자료 기준',
                'shape_id': key, 'source_id': source['출처id'], 'source_rank': source['순위'],
                'representative': True, 'archived_source_count': shape['출처수'],
                'reading_basis': 'reviewed_representative',
            }, literature_priority=0, passage_order=int(key[1:])))
        return out

    def repository(self, original, *, measurements=None):
        self.verify_snapshot(original)
        catalog = self

        class GroupedRepository:
            def load_candidates(self, labels):
                return catalog.candidates(labels, measurements=measurements)

        return GroupedRepository()

    def apply(self, result):
        for passage in result['passages']:
            key = passage['source']['shape_id']
            presentation = self.bindings[key]['presentation']
            passage['reading'] = {
                'title': presentation['title'],
                'meaning': presentation['meaning'],
                'meaning_basis': 'representative',
                'valence': 'favorable', 'valence_label': '길하게 본 기록',
                'topics': list(presentation['topics']),
            }
        result['named_types'] = []
        result['ranking_policy'].update(
            grouping='one_per_shape_before_top_five',
            representative_policy='reviewed_compatible_favorable_common_source',
            catalog_version=self.document['bindings']['version'])
        return result
