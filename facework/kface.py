"""K-FACE High_Resolution access: participant ZIPs read in place, no loose copies."""
import io, re, threading, zipfile
from collections import OrderedDict
from pathlib import Path
from PIL import Image
from .features import ROOT

RAW = ROOT / 'data/raw/kface'
ZIPS = RAW / 'zips'
DEFAULT_SOURCE = Path.home() / 'Downloads/High_Resolution.zip'
DERIVED = ROOT / 'data/derived/kface'

ACCESSORIES = [f'S{i:03}' for i in range(1, 7)]
LIGHTS = [f'L{i}' for i in range(1, 31)]
EXPRESSIONS = ['E01', 'E02', 'E03']
CAMERAS = [f'C{i}' for i in range(1, 21)]
# Meanings checked on sample captures (2026-09-26) and on MediaPipe pose/blendshape statistics
# (data/derived/kface/capture_probe.json); not taken from provider documentation.
ACCESSORY_KO = {'S001': '없음', 'S002': '일반 안경', 'S003': '뿔테 안경', 'S004': '선글라스', 'S005': '모자', 'S006': '모자+뿔테 안경'}
EXPRESSION_KO = {'E01': '무표정', 'E02': '웃음(입 벌림)', 'E03': '찡그림(눈 감음)'}
CAMERA_YAW_PITCH = {  # median MediaPipe head yaw/pitch over 10 participants, S001/L1/E01
    'C1': 'profile (<-75)', 'C2': (-73, -2), 'C3': (-56, -6), 'C4': (-41, -7), 'C5': (-27, -7), 'C6': (-13, -6), 'C7': (0, -5),
    'C8': (12, -6), 'C9': (27, -7), 'C10': (40, -7), 'C11': (54, -4), 'C12': (65, -5), 'C13': 'profile (>75)',
    'C14': (-38, 23), 'C15': (-12, 18), 'C16': (-1, 16), 'C17': (10, 17), 'C18': (34, 25), 'C19': (-27, -23), 'C20': (26, -23)}

KEY = re.compile(r'^(S\d{3})/(L\d+)/(E\d{2})/(C\d+)\.(jpg|txt)$')

def member(s, l, e, c, ext='jpg'):
    return f'{s}/{l}/{e}/{c}.{ext}'

def image_id(subject, s, l, e, c):
    return f'kf_{subject}_{s}_{l}_{e}_{c}'

def parse_image_id(value):
    _, subject, s, l, e, c = value.split('_')
    return subject, s, l, e, c

def subjects():
    return sorted(p.stem for p in ZIPS.glob('*.zip'))

def parse_annotation(text):
    """Provider TXT: 7 points, a square face box, then part boxes (x y w h).

    Point order (checked on camera geometry, see docs): nose tip, eye centre R/L (subject),
    mouth corner R/L, ear R/L. Rows after the face box follow the image left-to-right
    convention for eyes and are kept as delivered; their count varies with visibility.
    """
    rows = [[int(v) for v in line.split()] for line in text.replace('\r', '').split('\n') if line.strip()]
    points = [r for r in rows if len(r) == 2]
    boxes = [r for r in rows if len(r) == 4]
    if len(points) != 7 or not boxes:
        raise ValueError(f'unexpected annotation layout: {len(points)} points, {len(boxes)} boxes')
    return {'points': points, 'face_box': boxes[0], 'part_boxes': boxes[1:]}

class Store:
    """Thread-safe LRU of open participant ZIPs. Each worker process owns its own Store."""
    def __init__(self, capacity=8):
        self.capacity = capacity; self.open = OrderedDict(); self.lock = threading.Lock()

    def _zip(self, subject):
        with self.lock:
            z = self.open.pop(subject, None)
            if z is None:
                z = zipfile.ZipFile(ZIPS / f'{subject}.zip')
                while len(self.open) >= self.capacity:
                    self.open.popitem(last=False)[1].close()
            self.open[subject] = z
            return z

    def read(self, subject, name):
        z = self._zip(subject)
        with self.lock:
            return z.read(name)

    def image(self, subject, s, l, e, c):
        return Image.open(io.BytesIO(self.read(subject, member(s, l, e, c)))).convert('RGB')

    def annotation(self, subject, s, l, e, c):
        try:
            return parse_annotation(self.read(subject, member(s, l, e, c, 'txt')).decode('utf-8', 'replace'))
        except KeyError:
            return None

    def names(self, subject):
        return self._zip(subject).namelist()

    def close(self):
        for z in self.open.values():
            z.close()
        self.open.clear()
