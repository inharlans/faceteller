"""Paths and switches of the deploy/v1 runner, from command-line arguments or environment variables (arguments win).

  --models-dir          FACETELLER_MODELS_DIR          our own model files (from S3), laid out as in config/manifest.json:
                                                       candidate-v2-posenorm/models/{part}_fold{k}.pt, posenorm_template.npy,
                                                       vggt3/vggt_template.npy      default: <repo>/experiments/feature17-20261109
  --public-models-dir   FACETELLER_PUBLIC_MODELS_DIR   vggt/model.pt, face_landmarker.task, face_parsing/resnet18.onnx
                                                       default: <repo>/models
  --config-dir          FACETELLER_CONFIG_DIR          the JSON files of this folder's config/        default: deploy/v1/config
  --vggt-dtype          FACETELLER_VGGT_DTYPE          auto | bf16 | fp16 | fp32   (auto: bf16 on GPUs with compute capability >= 8,
                                                       fp16 otherwise, e.g. T4)
  --check               FACETELLER_CHECK_CNN=1         also run the check-only CNNs (off by default; those models are not on S3)
  --check-models-dir    FACETELLER_CHECK_MODELS_DIR    where the check-only model folders are           default: the models dir
  --llm-url             FACETELLER_LLM_URL             Ollama address for the closing remark (summary)   default: http://127.0.0.1:11434
  --llm-model           FACETELLER_LLM_MODEL           model name in Ollama; "off" = no model, the remark is put together
                                                       from a template                                  default: config/summary_prompt.json default_model
  --llm-timeout         FACETELLER_LLM_TIMEOUT         seconds to wait for the model                    default: 60
"""
import argparse, os
from dataclasses import dataclass
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]


@dataclass
class Settings:
    models_dir: Path
    public_models_dir: Path
    config_dir: Path
    vggt_dtype: str
    check: bool
    check_models_dir: Path
    llm_url: str = 'http://127.0.0.1:11434'
    llm_model: str = ''
    llm_timeout: float = 60.0


def add_arguments(ap: argparse.ArgumentParser):
    ap.add_argument('--models-dir'); ap.add_argument('--public-models-dir'); ap.add_argument('--config-dir'); ap.add_argument('--vggt-dtype', choices=['auto', 'bf16', 'fp16', 'fp32'])
    ap.add_argument('--check', action='store_true', default=None, help='also run the check-only CNNs (off by default)'); ap.add_argument('--check-models-dir')
    ap.add_argument('--llm-url'); ap.add_argument('--llm-model'); ap.add_argument('--llm-timeout', type=float)
    return ap


def resolve(args=None) -> Settings:
    get = lambda name, env, default: Path(os.path.expanduser(str(getattr(args, name, None) or os.environ.get(env) or default))).resolve()
    models = get('models_dir', 'FACETELLER_MODELS_DIR', ROOT / 'experiments/feature17-20261109'); dtype = getattr(args, 'vggt_dtype', None) or os.environ.get('FACETELLER_VGGT_DTYPE', 'auto')
    if dtype not in ('auto', 'bf16', 'fp16', 'fp32'): raise SystemExit(f'unknown VGGT dtype: {dtype}')
    check = bool(getattr(args, 'check', None)) or os.environ.get('FACETELLER_CHECK_CNN', '').lower() in ('1', 'true', 'yes', 'on')
    cfgdir = get('config_dir', 'FACETELLER_CONFIG_DIR', HERE / 'config'); text = lambda name, env, default: str(getattr(args, name, None) or os.environ.get(env) or default)
    prompt = cfgdir / 'summary_prompt.json'; default_model = __import__('json').loads(prompt.read_text(encoding='utf8'))['default_model'] if prompt.exists() else ''
    model = text('llm_model', 'FACETELLER_LLM_MODEL', default_model); model = '' if model.lower() in ('off', 'none', '0') else model
    return Settings(models, get('public_models_dir', 'FACETELLER_PUBLIC_MODELS_DIR', ROOT / 'models'), cfgdir, dtype, check,
                    get('check_models_dir', 'FACETELLER_CHECK_MODELS_DIR', models), text('llm_url', 'FACETELLER_LLM_URL', 'http://127.0.0.1:11434'), model, float(text('llm_timeout', 'FACETELLER_LLM_TIMEOUT', 60)))


def required_files(s: Settings, parts):
    """(path, what) of every file the structure needs; used for a clear start-up error."""
    out = [(s.models_dir / f'candidate-v2-posenorm/models/{p}_fold{k}.pt', f'CNN {p} fold {k} (S3)') for p in parts for k in range(5)]
    out += [(s.models_dir / 'posenorm_template.npy', 'pose template (S3)'), (s.models_dir / 'vggt3/vggt_template.npy', 'VGGT template (S3)'),
            (s.public_models_dir / 'vggt/model.pt', 'VGGT-1B weights (public)'), (s.public_models_dir / 'face_landmarker.task', 'MediaPipe Face Landmarker (public)'),
            (s.public_models_dir / 'face_parsing/resnet18.onnx', 'face parsing (public)')]
    return out
