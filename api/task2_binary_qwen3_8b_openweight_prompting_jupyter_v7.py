#!/usr/bin/env python
# coding: utf-8

# Converted from: task2_binary_qwen3_8b_openweight_prompting_jupyter_v7.ipynb
# NOTE: Notebook cell order is preserved. Markdown cells are comments.

# %% [markdown] cell 1
# # Modelo open-weight `Qwen3-8B` para Tarea 2 (Jupyter VM)
#
# Este notebook evalua un modelo de lenguaje de pesos abiertos para la **extraccion automatica de contribuciones cientificas** en espaÃ±ol academico.
#
# A diferencia de los notebooks de BETO y SciBERT, este experimento no realiza fine-tuning. Segun la definicion del proyecto, los modelos open-weight de 1-8B se usan en **modo de inferencia** mediante prompting para clasificacion binaria de fragmentos. Por esa razon, el notebook guarda artefactos del experimento, configuracion del modelo, prompts, predicciones, metricas, matrices de confusion y analisis de errores; no genera un checkpoint entrenado salvo que se active explicitamente `SAVE_FULL_MODEL`.
#
# En esta version v5, `pos_rel` no se incluye en el prompt; se conserva en el dataset solo para trazabilidad y analisis. Se comparan las estrategias `zero_shot` y `few_shot` adaptadas desde el notebook de Gemini.

# %% [markdown] cell 2
# ## Celda 1: Instalacion de dependencias
#
# Esta celda crea un entorno reproducible temporal en `/workspace/qwen_task2/qwen-env`, instala PyTorch 2.6.0 CUDA 12.4 y las dependencias fijadas que fueron validadas en la VM. Debe ejecutarse despues de cada reinicio si `/workspace` fue limpiado.

# %% [code] cell 3
# Bootstrap reproducible para VM/Jupyter.
# Ejecuta esta celda despues de cada reinicio de la VM si desaparecio /workspace/qwen_task2/qwen-env.
# Al terminar, cambia el kernel del notebook a: Python (qwen-task2)

import json
import os
import shutil
import site
import subprocess
import sys
from pathlib import Path

WORKSPACE_BOOTSTRAP = Path('/workspace/qwen_task2')
VENV_DIR = WORKSPACE_BOOTSTRAP / 'qwen-env'
PIP_CACHE_DIR = WORKSPACE_BOOTSTRAP / 'pip-cache'
HF_HOME_DIR = WORKSPACE_BOOTSTRAP / 'huggingface'
TMPDIR = WORKSPACE_BOOTSTRAP / 'tmp'
XDG_CACHE_HOME = WORKSPACE_BOOTSTRAP / 'xdg-cache'
VIRTUALENV_APP_DATA = WORKSPACE_BOOTSTRAP / 'virtualenv-app-data'
KERNEL_DIR = Path.home() / '.local' / 'share' / 'jupyter' / 'kernels' / 'qwen-task2'

for path in [WORKSPACE_BOOTSTRAP, PIP_CACHE_DIR, HF_HOME_DIR, TMPDIR, XDG_CACHE_HOME, VIRTUALENV_APP_DATA]:
    path.mkdir(parents=True, exist_ok=True)

base_env = os.environ.copy()
base_env['PIP_CACHE_DIR'] = str(PIP_CACHE_DIR)
base_env['HF_HOME'] = str(HF_HOME_DIR)
base_env['HF_HUB_CACHE'] = str(HF_HOME_DIR / 'hub')
base_env['TMPDIR'] = str(TMPDIR)
base_env['XDG_CACHE_HOME'] = str(XDG_CACHE_HOME)
base_env['VIRTUALENV_OVERRIDE_APP_DATA'] = str(VIRTUALENV_APP_DATA)
base_env.pop('TRANSFORMERS_CACHE', None)

# Ambiente limpio para el venv: evita que Python cargue paquetes antiguos de /home/jovyan/.local.
venv_env = base_env.copy()
venv_env['PYTHONNOUSERSITE'] = '1'
venv_env.pop('PYTHONPATH', None)

# Ambiente solo para ejecutar virtualenv desde el Python del sistema si python -m venv falla.
bootstrap_env = base_env.copy()
user_site = site.getusersitepackages()
bootstrap_env['PYTHONPATH'] = user_site + (os.pathsep + bootstrap_env['PYTHONPATH'] if bootstrap_env.get('PYTHONPATH') else '')
bootstrap_env.pop('PYTHONNOUSERSITE', None)

python_bin = VENV_DIR / 'bin' / 'python'


def venv_has_pip() -> bool:
    if not python_bin.exists():
        return False
    result = subprocess.run(
        [str(python_bin), '-m', 'pip', '--version'],
        env=venv_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


if VENV_DIR.exists() and not venv_has_pip():
    print('Virtualenv existente esta incompleto o no tiene pip. Recreando:', VENV_DIR)
    shutil.rmtree(VENV_DIR)


def create_env() -> None:
    print('Creando entorno en:', VENV_DIR)
    try:
        subprocess.check_call([sys.executable, '-m', 'venv', str(VENV_DIR)], env=venv_env)
        return
    except subprocess.CalledProcessError:
        print('python -m venv fallo porque falta ensurepip/python3.10-venv. Usando virtualenv como fallback...')

    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', 'virtualenv'], env=bootstrap_env)
    subprocess.check_call([sys.executable, '-m', 'virtualenv', str(VENV_DIR)], env=bootstrap_env)


if not VENV_DIR.exists():
    create_env()
else:
    print('Virtualenv ya existe y tiene pip:', VENV_DIR)

python_bin = VENV_DIR / 'bin' / 'python'
if not venv_has_pip():
    print('El entorno creado no tiene pip. Intentando inicializar con ensurepip...')
    subprocess.check_call([str(python_bin), '-m', 'ensurepip', '--upgrade'], env=venv_env)

subprocess.check_call([str(python_bin), '-m', 'pip', 'install', '--upgrade', 'pip'], env=venv_env)
subprocess.check_call([
    str(python_bin), '-m', 'pip', 'install',
    '--index-url', 'https://download.pytorch.org/whl/cu124',
    'torch==2.6.0'
], env=venv_env)
subprocess.check_call([
    str(python_bin), '-m', 'pip', 'install',
    'transformers==4.51.3',
    'tokenizers==0.21.1',
    'huggingface-hub==0.30.2',
    'safetensors==0.4.3',
    'accelerate==1.6.0',
    'bitsandbytes==0.45.5',
    'hf_xet',
    'pandas', 'scikit-learn', 'matplotlib', 'seaborn', 'joblib', 'ipykernel'
], env=venv_env)

validation_code = """
import torch, transformers, accelerate, bitsandbytes
print('validacion:', torch.__version__, torch.version.cuda, torch.cuda.is_available(), transformers.__version__, accelerate.__version__)
print('torch path:', torch.__file__)
print('transformers path:', transformers.__file__)
print('bitsandbytes path:', bitsandbytes.__file__)
"""
subprocess.check_call([str(python_bin), '-c', validation_code], env=venv_env)

subprocess.check_call([
    str(python_bin), '-m', 'ipykernel', 'install', '--user',
    '--name', 'qwen-task2', '--display-name', 'Python (qwen-task2)'
], env=venv_env)

kernel_json = KERNEL_DIR / 'kernel.json'
if kernel_json.exists():
    kernel_data = json.loads(kernel_json.read_text(encoding='utf-8'))
    kernel_data.setdefault('env', {})
    kernel_data['env'].update({
        'PYTHONNOUSERSITE': '1',
        'PYTHONPATH': '',
        'HF_HOME': str(HF_HOME_DIR),
        'HF_HUB_CACHE': str(HF_HOME_DIR / 'hub'),
        'TMPDIR': str(TMPDIR),
        'XDG_CACHE_HOME': str(XDG_CACHE_HOME),
    })
    kernel_json.write_text(json.dumps(kernel_data, indent=2), encoding='utf-8')
    print('Kernelspec actualizado:', kernel_json)

print('\nBootstrap listo.')
print('Kernel instalado: Python (qwen-task2)')
print('Python:', python_bin)
print('HF_HOME:', HF_HOME_DIR)
print('Ahora cambia el kernel a Python (qwen-task2), reinicia el kernel y continua con la siguiente celda.')

# %% [markdown] cell 4
# ## Celda 2: Imports
#
# Se valida que el notebook este usando el kernel aislado `Python (qwen-task2)`, se configura el workspace temporal grande bajo `/workspace/qwen_task2` para cachear Qwen3-8B, y se importan utilidades para lectura del dataset, inferencia, metricas, visualizacion y guardado de artefactos.

# %% [code] cell 5
from __future__ import annotations

import sys
import os
import csv
from pathlib import Path

EXPECTED_ENV_PREFIX = '/workspace/qwen_task2/qwen-env'
if not sys.executable.startswith(EXPECTED_ENV_PREFIX):
    raise RuntimeError(
        'Este notebook debe ejecutarse con el kernel Python (qwen-task2). '
        'Ejecuta primero la celda bootstrap y cambia el kernel antes de continuar. '
        f'Python actual: {sys.executable}'
    )

HOME_ROOT = Path('/home/jovyan')
WORKSPACE_ROOT = Path('/workspace/qwen_task2')
HF_HOME_PATH = WORKSPACE_ROOT / 'huggingface'
TMPDIR_PATH = WORKSPACE_ROOT / 'tmp'

HF_HOME_PATH.mkdir(parents=True, exist_ok=True)
TMPDIR_PATH.mkdir(parents=True, exist_ok=True)

os.environ['HF_HOME'] = str(HF_HOME_PATH)
os.environ['HF_HUB_CACHE'] = str(HF_HOME_PATH / 'hub')
os.environ['TMPDIR'] = str(TMPDIR_PATH)
os.environ.pop('TRANSFORMERS_CACHE', None)

import gc
import inspect
import json
import random
import re
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython.display import display
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, set_seed

print('python:', sys.executable)
print('HOME_ROOT:', HOME_ROOT)
print('WORKSPACE_ROOT:', WORKSPACE_ROOT)
print('HF_HOME:', os.environ['HF_HOME'])
print('TMPDIR:', os.environ['TMPDIR'])
print('torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('torch path:', torch.__file__)
print('cuda disponible:', torch.cuda.is_available())
print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

# %% [markdown] cell 6
# ## Celda 3: Rutas locales en la VM
#
# Se definen rutas locales para ejecutar el experimento desde Jupyter en la VM. Se espera que el dataset este en `/home/jovyan/data`; los outputs livianos quedan en `/home/jovyan/outputs`, mientras la cache pesada del modelo queda en `/workspace/qwen_task2`.

# %% [code] cell 7
PROJECT_ROOT = Path('/home/jovyan')
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_BASE_DIR = PROJECT_ROOT / 'outputs'

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
HF_HOME_PATH.mkdir(parents=True, exist_ok=True)
TMPDIR_PATH.mkdir(parents=True, exist_ok=True)

print('PROJECT_ROOT =', PROJECT_ROOT)
print('DATA_DIR =', DATA_DIR)
print('OUTPUT_BASE_DIR =', OUTPUT_BASE_DIR)
print('WORKSPACE_ROOT =', WORKSPACE_ROOT)
print('HF_HOME =', os.environ['HF_HOME'])

# %% [markdown] cell 8
# ## Celda 4: Configuracion principal
#
# Esta celda define el modelo open-weight, las columnas del consolidado, el modo de prompting, limites de inferencia y rutas de salida locales.
#
# `Dataset_type` se usa como columna oficial de particion del CSV. La columna `dataset_type` existe en algunas versiones del archivo, pero no se usa aqui. Esta variante esta preparada para una VM con GPU NVIDIA L40-24C.

# %% [code] cell 9
PROJECT_ROOT = Path('/home/jovyan')
DATASET_FILENAME = 'DATASET_TAREA_2_CONSOLIDADO_FINAL_ORIGINAL_v3.csv'

OUTPUT_DIR = OUTPUT_BASE_DIR / 'outputs_task2_binary_qwen3_8b_openweight_jupyter_v5'
FINAL_MODEL_DIR = OUTPUT_DIR / 'final_model'
PROMPT_DIR = OUTPUT_DIR / 'prompts'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
PROMPT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'Qwen/Qwen3-8B'
RUN_NAME = 'task2_binary_qwen3_8b_openweight_prompting_jupyter_v5'

TEXT_COLUMN = 'text'
LABEL_COLUMN = 'LABEL_T2'
SPLIT_COLUMN = 'Dataset_type'
UNUSED_SPLIT_COLUMN = 'dataset_type'
ORIGINAL_RHETORICAL_LABEL_COLUMN = 'label'
POSITION_COLUMN = 'pos_rel'

RANDOM_STATE = 42
MAX_TEST_ROWS = None
MAX_EVAL_ROWS = None
MAX_TRAIN_REFERENCE_ROWS = None  # Usa todo TRAIN para reproducir el few-shot k=6 del notebook Gemini.

MAX_INPUT_CHARS = 6500
MAX_NEW_TOKENS = 60
MAX_TEXT_WORDS = 700
TEMPERATURE = 0.0
DO_SAMPLE = False

PROMPT_STRATEGIES = ['zero_shot', 'few_shot']
DEFAULT_STRATEGY_FOR_ERROR_ANALYSIS = 'zero_shot'

USE_4BIT = True
SAVE_FULL_MODEL = False
EXPECTED_GPU = 'NVIDIA L40-24C'
CACHE_LOCATION = os.environ.get('HF_HOME', '/workspace/qwen_task2/huggingface')

RUN_CONFIG_FILE = OUTPUT_DIR / f'{RUN_NAME}_run_config.json'
METRICS_FILE = OUTPUT_DIR / f'{RUN_NAME}_metrics.json'
SUMMARY_METRICS_CSV = OUTPUT_DIR / f'{RUN_NAME}_summary_metrics.csv'

print('DATASET_FILENAME =', DATASET_FILENAME)
print('OUTPUT_DIR =', OUTPUT_DIR)
print('MODEL_NAME =', MODEL_NAME)
print('PROMPT_STRATEGIES =', PROMPT_STRATEGIES)
print('MAX_TEXT_WORDS =', MAX_TEXT_WORDS)
print('EXPECTED_GPU =', EXPECTED_GPU)
print('CACHE_LOCATION =', CACHE_LOCATION)
print('GPU disponible =', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU =', torch.cuda.get_device_name(0))
else:
    print('Activa GPU en Entorno de ejecucion > Cambiar tipo de entorno de ejecucion > GPU')

# %% [markdown] cell 10
# ## Celda 5: Funciones auxiliares
#
# Estas funciones preparan el dataset binario, construyen prompts sin referencias heuristicas, parsean respuestas, calculan metricas, guardan matrices de confusion y generan analisis de errores.

# %% [code] cell 11
LABEL_NAMES = ['no_contribution', 'contribution']
label2id = {'no_contribution': 0, 'contribution': 1}
id2label = {0: 'no_contribution', 1: 'contribution'}


def find_header_row(path: Path, encoding: str) -> int | None:
    required_tokens = ['doc_id', 'filename', 'text', 'label', 'Dataset_type']
    with path.open('r', encoding=encoding, errors='replace') as handle:
        for idx, line in enumerate(handle):
            normalized = line.replace('\ufeff', '')
            if all(token in normalized for token in required_tokens):
                return idx
    return None


def read_csv_flexible(path: Path) -> pd.DataFrame:
    encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
    attempts = []
    required_columns = {'text', 'label', 'Dataset_type', 'pos_rel'}

    for encoding in encodings:
        header_row = None
        try:
            header_row = find_header_row(path, encoding)
        except UnicodeDecodeError as exc:
            attempts.append({'encoding': encoding, 'error': repr(exc)})
            continue

        read_kwargs_list = [
            {'encoding': encoding, 'sep': ',', 'engine': 'python'},
            {'encoding': encoding, 'sep': ',', 'engine': 'python', 'quoting': csv.QUOTE_NONE},
            {'encoding': encoding, 'sep': ',', 'engine': 'python', 'quoting': csv.QUOTE_NONE, 'on_bad_lines': 'warn'},
            {'encoding': encoding, 'sep': None, 'engine': 'python'},
        ]
        if header_row is not None:
            read_kwargs_list = [
                {**kwargs, 'skiprows': header_row}
                for kwargs in read_kwargs_list
            ] + read_kwargs_list

        for kwargs in read_kwargs_list:
            try:
                df = pd.read_csv(path, **kwargs)
                df.columns = [str(col).replace('\ufeff', '').strip() for col in df.columns]
                if required_columns.issubset(df.columns):
                    print('CSV leido con:', kwargs)
                    if header_row not in (None, 0):
                        print('Encabezado detectado en linea:', header_row + 1)
                    return df
                attempts.append({'kwargs': kwargs, 'columns': df.columns.tolist()[:12]})
            except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
                attempts.append({'kwargs': kwargs, 'error': repr(exc)})

    raise RuntimeError(
        'No se pudo leer el CSV con las columnas esperadas. '
        f'Archivo: {path}. Primeros intentos: {attempts[:6]}'
    )


def find_dataset_path() -> Path:
    candidates = [
        DATA_DIR / DATASET_FILENAME,
        PROJECT_ROOT / DATASET_FILENAME,
        Path.cwd() / DATASET_FILENAME,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f'No se encontro {DATASET_FILENAME}. '
        'Dejalo en /home/jovyan/data/ o en el directorio raiz /home/jovyan/.'
    )


def prepare_dataframe(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    required_columns = {TEXT_COLUMN, SPLIT_COLUMN, ORIGINAL_RHETORICAL_LABEL_COLUMN, POSITION_COLUMN}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f'Faltan columnas requeridas para Tarea 2: {sorted(missing)}')

    out = df.copy()
    out[SPLIT_COLUMN] = out[SPLIT_COLUMN].astype(str).str.strip().str.upper()
    out = out[out[SPLIT_COLUMN] == split_name].copy()
    out[TEXT_COLUMN] = out[TEXT_COLUMN].fillna('').astype(str).str.strip()
    out = out[out[TEXT_COLUMN] != ''].copy()

    out[ORIGINAL_RHETORICAL_LABEL_COLUMN] = (
        out[ORIGINAL_RHETORICAL_LABEL_COLUMN].fillna('').astype(str).str.strip().str.upper()
    )

    # El dataset v3 no trae LABEL_T2: se deriva como binaria desde la etiqueta retorica.
    # CONTR = contribucion cientifica; cualquier otra etiqueta = no contribucion.
    if LABEL_COLUMN not in out.columns:
        out[LABEL_COLUMN] = (out[ORIGINAL_RHETORICAL_LABEL_COLUMN] == 'CONTR').astype(int)
    else:
        out[LABEL_COLUMN] = pd.to_numeric(out[LABEL_COLUMN], errors='coerce')

    out = out[out[LABEL_COLUMN].isin([0, 1])].copy()
    out[LABEL_COLUMN] = out[LABEL_COLUMN].astype(int)
    out['label_name'] = out[LABEL_COLUMN].map(id2label)

    out[POSITION_COLUMN] = pd.to_numeric(out[POSITION_COLUMN], errors='coerce')
    out = out[out[POSITION_COLUMN].notna()].copy()
    out[POSITION_COLUMN] = out[POSITION_COLUMN].clip(lower=0.0, upper=1.0).astype(float)

    out['word_count'] = out[TEXT_COLUMN].str.split().map(len)
    out = out.reset_index(drop=True)
    return out


def ensure_dataframes_loaded() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    global dataset_path, raw_df, train_ref_df, test_df, eval_df

    required_names = ['dataset_path', 'raw_df', 'train_ref_df', 'test_df', 'eval_df']
    if all(name in globals() for name in required_names):
        return raw_df, train_ref_df, test_df, eval_df, dataset_path

    dataset_path = find_dataset_path()
    raw_df = read_csv_flexible(dataset_path)

    train_ref_df = prepare_dataframe(raw_df, 'TRAIN')
    test_df = prepare_dataframe(raw_df, 'TEST')
    eval_df = prepare_dataframe(raw_df, 'EVAL')

    if MAX_TRAIN_REFERENCE_ROWS is not None:
        train_ref_df = train_ref_df.sample(
            n=min(MAX_TRAIN_REFERENCE_ROWS, len(train_ref_df)),
            random_state=RANDOM_STATE,
        ).reset_index(drop=True)
    if MAX_TEST_ROWS is not None:
        test_df = test_df.sample(n=min(MAX_TEST_ROWS, len(test_df)), random_state=RANDOM_STATE).reset_index(drop=True)
    if MAX_EVAL_ROWS is not None:
        eval_df = eval_df.sample(n=min(MAX_EVAL_ROWS, len(eval_df)), random_state=RANDOM_STATE).reset_index(drop=True)

    return raw_df, train_ref_df, test_df, eval_df, dataset_path

def position_bucket(pos_rel: float) -> str:
    if pos_rel < 0.20:
        return 'inicio del documento'
    if pos_rel < 0.45:
        return 'primera mitad'
    if pos_rel < 0.70:
        return 'zona media'
    if pos_rel < 0.90:
        return 'segunda mitad'
    return 'cierre del documento'


def truncate_text(text: str, max_chars: int = MAX_INPUT_CHARS) -> str:
    text = re.sub(r'\s+', ' ', str(text)).strip()
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2].strip()
    tail = text[-max_chars // 2 :].strip()
    return head + '\\n[... texto omitido por longitud ...]\\n' + tail


# Prompt adaptado desde el punto 3 de 06_task2_gemini_classifier.ipynb.
# Estrategias: zero-shot y few-shot k=6.
GEMINI_PROMPT_SOURCE_NOTE = '06_task2_gemini_classifier.ipynb - punto 3 Diseno del prompt'
SYSTEM_PROMPT_GEMINI = """Eres un experto en anÃ¡lisis del discurso cientÃ­fico en espaÃ±ol.
Tu tarea es determinar si un fragmento de un artÃ­culo acadÃ©mico expresa una CONTRIBUCIÃ“N CIENTÃFICA ORIGINAL del propio trabajo.

Una contribuciÃ³n cientÃ­fica es una unidad textual donde el trabajo mismo presenta un aporte intelectual que no existÃ­a previamente. Puede manifestarse como un mÃ©todo desarrollado, un recurso construido, un marco conceptual propuesto, evidencia empÃ­rica nueva, un hallazgo atribuible al estudio o un avance conceptual concreto. Lo esencial es que el aporte sea propio del trabajo y represente algo nuevo con respecto al estado del conocimiento anterior.

label=1 â€” el fragmento expresa un aporte original del propio estudio: un mÃ©todo, recurso, evidencia, hallazgo, marco o avance que el trabajo presenta como su contribuciÃ³n al campo. El contenido semÃ¡ntico indica novedad y autorÃ­a del aporte.

label=0 â€” el fragmento no expresa un aporte original. Incluye: descripciÃ³n de antecedentes o trabajos de otros autores, exposiciÃ³n de metodologÃ­a sin declarar novedad, presentaciÃ³n de resultados empÃ­ricos sin atribuirlos como aporte, interpretaciÃ³n o discusiÃ³n de hallazgos, limitaciones del estudio o conclusiones que solo resumen sin declarar contribuciÃ³n.

Ante la duda, clasifica como label=0. Un fragmento que describe QUÃ‰ SE HIZO sin declarar que eso constituye un aporte nuevo no es una contribuciÃ³n.

Responde ÃšNICAMENTE con un JSON vÃ¡lido en el formato exacto:
{"label": <0 o 1>, "confidence": <nÃºmero entre 0.0 y 1.0>}

No incluyas explicaciones, markdown ni texto fuera del JSON."""
FEWSHOT_K_PER_CLASS = 3
FEWSHOT_EXAMPLES = []


def truncate_words(text: str, max_words: int = MAX_TEXT_WORDS) -> str:
    words = re.sub(r'\s+', ' ', str(text)).strip().split()
    if len(words) <= max_words:
        return ' '.join(words)
    return ' '.join(words[:max_words])


def build_system_prompt(strategy: str = 'zero_shot') -> str:
    return SYSTEM_PROMPT_GEMINI


def build_user_message(row: pd.Series) -> str:
    text = truncate_words(row[TEXT_COLUMN])
    return f"Fragmento:\n\n{text}"

def build_user_prompt(row: pd.Series, strategy: str) -> str:
    return build_user_message(row)


def build_fewshot_examples(train_df: pd.DataFrame) -> list[dict]:
    examples = []
    for label_id in [1, 0]:
        candidates = train_df[train_df[LABEL_COLUMN] == label_id].copy()
        if candidates.empty:
            raise ValueError(f'No hay ejemplos TRAIN para LABEL_T2={label_id}')
        sampled = candidates.sample(
            n=min(FEWSHOT_K_PER_CLASS, len(candidates)),
            random_state=RANDOM_STATE,
        )[[TEXT_COLUMN, LABEL_COLUMN, POSITION_COLUMN, 'word_count']]
        for _, example_row in sampled.iterrows():
            examples.append({
                TEXT_COLUMN: example_row[TEXT_COLUMN],
                LABEL_COLUMN: int(example_row[LABEL_COLUMN]),
                POSITION_COLUMN: float(example_row[POSITION_COLUMN]),
                'word_count': int(example_row['word_count']),
            })
    return examples


def ensure_fewshot_examples() -> list[dict]:
    global FEWSHOT_EXAMPLES
    if not FEWSHOT_EXAMPLES:
        ensure_dataframes_loaded()
        FEWSHOT_EXAMPLES = build_fewshot_examples(train_ref_df)
    return FEWSHOT_EXAMPLES


def messages_for_row(row: pd.Series, strategy: str) -> list[dict]:
    messages = [{'role': 'system', 'content': build_system_prompt(strategy)}]

    if strategy == 'few_shot':
        for example in ensure_fewshot_examples():
            ex_row = pd.Series(example)
            messages.append({'role': 'user', 'content': build_user_message(ex_row)})
            messages.append({
                'role': 'assistant',
                'content': json.dumps({'label': int(example[LABEL_COLUMN]), 'confidence': 1.0}, ensure_ascii=False),
            })

    messages.append({'role': 'user', 'content': build_user_message(row)})
    return messages


def prompt_examples_summary() -> dict:
    if not FEWSHOT_EXAMPLES:
        return {'few_shot_examples': 0, 'source': GEMINI_PROMPT_SOURCE_NOTE}
    counts = pd.Series([example[LABEL_COLUMN] for example in FEWSHOT_EXAMPLES]).value_counts().to_dict()
    return {
        'few_shot_examples': len(FEWSHOT_EXAMPLES),
        'label_counts': {str(k): int(v) for k, v in counts.items()},
        'source': GEMINI_PROMPT_SOURCE_NOTE,
    }

def parse_model_response(text: str) -> dict:
    raw = str(text).strip()
    json_match = re.search(r'\{.*\}', raw, flags=re.DOTALL)
    payload = {}
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            payload = {}

    raw_label = str(payload.get('label', '')).strip().lower()
    if raw_label in {'1', '1.0', 'true', 'contribution'}:
        label = 'contribution'
    elif raw_label in {'0', '0.0', 'false', 'no_contribution', 'no-contribution', 'no contr', 'no-contr'}:
        label = 'no_contribution'
    else:
        lower = raw.lower()
        if re.search(r'"label"\s*:\s*1', lower) or 'contribution' in lower and 'no_contribution' not in lower:
            label = 'contribution'
        elif re.search(r'"label"\s*:\s*0', lower) or 'no_contribution' in lower or 'no-contr' in lower:
            label = 'no_contribution'
        else:
            label = 'no_contribution'

    confidence = payload.get('confidence', np.nan)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = np.nan
    if not np.isnan(confidence):
        confidence = min(max(confidence, 0.0), 1.0)

    rationale = str(payload.get('rationale', '')).strip()
    return {
        'pred_label_name': label,
        'pred_label': int(label2id[label]),
        'confidence': confidence,
        'rationale': rationale,
        'raw_response': raw,
    }


def compute_metrics_from_arrays(y_true, y_pred, positive_scores=None) -> tuple[dict, pd.DataFrame, dict]:
    summary = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_positive': float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'recall_positive': float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'f1_positive': float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }

    if positive_scores is not None:
        scores = pd.Series(positive_scores, dtype='float64')
        valid_mask = scores.notna().to_numpy()
        y_true_valid = np.asarray(y_true)[valid_mask]
        scores_valid = scores.to_numpy()[valid_mask]
        if len(y_true_valid) > 0 and len(np.unique(y_true_valid)) == 2:
            try:
                summary['roc_auc'] = float(roc_auc_score(y_true_valid, scores_valid))
                summary['average_precision'] = float(average_precision_score(y_true_valid, scores_valid))
            except ValueError:
                pass

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    label_metrics_df = pd.DataFrame({
        'label_id': [0, 1],
        'label': LABEL_NAMES,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
    })

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return summary, label_metrics_df, report


def build_roc_curve_dataframe(y_true, positive_scores) -> pd.DataFrame:
    scores = pd.Series(positive_scores, dtype='float64')
    valid_mask = scores.notna().to_numpy()
    y_true_valid = np.asarray(y_true)[valid_mask]
    scores_valid = scores.to_numpy()[valid_mask]

    if len(y_true_valid) == 0 or len(np.unique(y_true_valid)) < 2:
        return pd.DataFrame(columns=['fpr', 'tpr', 'threshold'])

    fpr, tpr, thresholds = roc_curve(y_true_valid, scores_valid, pos_label=1)
    return pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr,
        'threshold': thresholds,
    })


def save_roc_curve(title: str, roc_curve_df: pd.DataFrame, roc_auc: float | None, path: Path) -> None:
    if roc_curve_df.empty:
        return

    plt.figure(figsize=(5, 4))
    plt.plot(roc_curve_df['fpr'], roc_curve_df['tpr'], label=f'ROC AUC = {roc_auc:.4f}' if roc_auc is not None else 'ROC')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Azar')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def save_confusion_matrix(title: str, pred_df: pd.DataFrame, path: Path) -> None:
    cm = confusion_matrix(pred_df[LABEL_COLUMN], pred_df['pred_label'], labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
    )
    plt.xlabel('Prediccion')
    plt.ylabel('Etiqueta real')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def build_error_analysis(split_name: str, strategy: str, pred_df: pd.DataFrame) -> dict:
    errors = pred_df[pred_df[LABEL_COLUMN] != pred_df['pred_label']].copy()
    false_positives = pred_df[(pred_df[LABEL_COLUMN] == 0) & (pred_df['pred_label'] == 1)].copy()
    false_negatives = pred_df[(pred_df[LABEL_COLUMN] == 1) & (pred_df['pred_label'] == 0)].copy()

    prefix = f'{RUN_NAME}_{strategy}_{split_name}'
    errors.to_csv(OUTPUT_DIR / f'{prefix}_errors.csv', index=False, encoding='utf-8-sig')
    false_positives.to_csv(OUTPUT_DIR / f'{prefix}_false_positives.csv', index=False, encoding='utf-8-sig')
    false_negatives.to_csv(OUTPUT_DIR / f'{prefix}_false_negatives.csv', index=False, encoding='utf-8-sig')

    if len(errors):
        rhetorical_groups = (
            errors.groupby(ORIGINAL_RHETORICAL_LABEL_COLUMN)
            .size()
            .sort_values(ascending=False)
            .reset_index(name='errors')
            .head(20)
        )
        position_groups = (
            errors.assign(pos_bucket=errors[POSITION_COLUMN].map(position_bucket))
            .groupby('pos_bucket')
            .size()
            .sort_values(ascending=False)
            .reset_index(name='errors')
        )
    else:
        rhetorical_groups = pd.DataFrame(columns=[ORIGINAL_RHETORICAL_LABEL_COLUMN, 'errors'])
        position_groups = pd.DataFrame(columns=['pos_bucket', 'errors'])

    rhetorical_groups.to_csv(OUTPUT_DIR / f'{prefix}_errors_by_rhetorical_label.csv', index=False, encoding='utf-8-sig')
    position_groups.to_csv(OUTPUT_DIR / f'{prefix}_errors_by_position.csv', index=False, encoding='utf-8-sig')

    return {
        'split': split_name,
        'strategy': strategy,
        'rows': int(len(pred_df)),
        'errors': int(len(errors)),
        'false_positives': int(len(false_positives)),
        'false_negatives': int(len(false_negatives)),
        'top_rhetorical_error_groups': rhetorical_groups.to_dict(orient='records'),
        'position_error_groups': position_groups.to_dict(orient='records'),
    }

# %% [markdown] cell 12
# ## Celda 6: Carga del consolidado de Tarea 2
#
# Se carga `DATASET_TAREA_2_CONSOLIDADO_FINAL_ORIGINAL_v3.csv` y se respetan las particiones definidas en `Dataset_type`.

# %% [code] cell 13
raw_df, train_ref_df, test_df, eval_df, dataset_path = ensure_dataframes_loaded()

print('Dataset usado:', dataset_path)
print('Split column usada:', SPLIT_COLUMN)
print('train_ref_df =', train_ref_df.shape)
print('test_df =', test_df.shape)
print('eval_df =', eval_df.shape)

display(train_ref_df.head(2))
display(test_df.head(2))
display(eval_df.head(2))

# %% [markdown] cell 14
# ## Celda 7: Auditoria de distribuciones
#
# Se revisa balance binario, distribucion retorica y consistencia de particiones para dejar trazabilidad del experimento.

# %% [code] cell 15
raw_df, train_ref_df, test_df, eval_df, dataset_path = ensure_dataframes_loaded()

split_summary = []
for split_name, df in [('TRAIN_REF', train_ref_df), ('TEST', test_df), ('EVAL', eval_df)]:
    counts = df[LABEL_COLUMN].value_counts().reindex([0, 1], fill_value=0)
    split_summary.append({
        'split': split_name,
        'no_contribution': int(counts.loc[0]),
        'contribution': int(counts.loc[1]),
        'rows': int(len(df)),
    })

split_summary_df = pd.DataFrame(split_summary)
display(split_summary_df)

print('Distribucion de etiqueta retorica original por split:')
display(pd.concat([
    train_ref_df.assign(split='TRAIN_REF'),
    test_df.assign(split='TEST'),
    eval_df.assign(split='EVAL'),
]).pivot_table(
    index='split',
    columns=ORIGINAL_RHETORICAL_LABEL_COLUMN,
    values=TEXT_COLUMN,
    aggfunc='count',
    fill_value=0,
))

if UNUSED_SPLIT_COLUMN in raw_df.columns:
    mismatch_count = int((raw_df[SPLIT_COLUMN].astype(str).str.upper() != raw_df[UNUSED_SPLIT_COLUMN].astype(str).str.upper()).sum())
    print(f'Filas donde {SPLIT_COLUMN} difiere de {UNUSED_SPLIT_COLUMN}:', mismatch_count)
    print(f'Nota: este notebook usa {SPLIT_COLUMN}, por decision del experimento.')

# %% [markdown] cell 16
# ## Celda 8: Tokenizador y modelo Qwen3-8B
#
# Se carga Qwen3-8B para generacion causal. Con el kernel `Python (qwen-task2)` se usa cuantizacion 4-bit por defecto; puede cambiarse con `USE_4BIT`.

# %% [code] cell 17
set_seed(RANDOM_STATE)

if not torch.cuda.is_available():
    raise RuntimeError('CUDA no esta disponible. Verifica que el kernel este ejecutandose en la VM con GPU.')

# Qwen3-8B completo requiere cache de varios GB. /workspace es temporal, pero amplio.
workspace_usage = os.statvfs(str(WORKSPACE_ROOT))
workspace_free_gb = workspace_usage.f_bavail * workspace_usage.f_frsize / 1024**3
print('Espacio libre en WORKSPACE_ROOT GB:', round(workspace_free_gb, 2))
if workspace_free_gb < 20:
    print('Advertencia: Qwen3-8B puede requerir mas espacio libre para descargarse/cachearse.')

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

quantization_config = None
if USE_4BIT:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map='auto',
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to('cuda')

model.eval()
DEVICE = next(model.parameters()).device

generation_kwargs = {
    'max_new_tokens': MAX_NEW_TOKENS,
    'do_sample': DO_SAMPLE,
    'pad_token_id': tokenizer.pad_token_id,
    'eos_token_id': tokenizer.eos_token_id,
}
if DO_SAMPLE:
    generation_kwargs['temperature'] = TEMPERATURE

generation_config = GenerationConfig(**generation_kwargs)

print(model.__class__.__name__)
print('Pad token:', tokenizer.pad_token, tokenizer.pad_token_id)
print('USE_4BIT:', USE_4BIT)
print('Quantization:', quantization_config)
print('Device:', DEVICE)
print('Memoria GPU asignada GB:', round(torch.cuda.memory_allocated() / 1024**3, 2))

# %% [markdown] cell 18
# ## Celda 9: Construccion de prompts
#
# Se guardan las plantillas `zero_shot` y `few_shot`. El few-shot es conceptual: no inserta fragmentos reales ni expresiones heuristicas de seleccion.

# %% [code] cell 19
# Inicializa datos y ejemplos few-shot antes de guardar previsualizaciones y ejecutar inferencia.
raw_df, train_ref_df, test_df, eval_df, dataset_path = ensure_dataframes_loaded()
ensure_fewshot_examples()

prompt_templates = {
    'prompt_source': GEMINI_PROMPT_SOURCE_NOTE,
    'system_prompt_zero_shot': build_system_prompt('zero_shot'),
    'few_shot_examples': FEWSHOT_EXAMPLES,
    'few_shot_summary': prompt_examples_summary(),
}

for strategy in PROMPT_STRATEGIES:
    prompt_templates[f'{strategy}_messages_preview'] = messages_for_row(test_df.iloc[0], strategy) if len(test_df) else []

(PROMPT_DIR / f'{RUN_NAME}_prompt_templates.json').write_text(
    json.dumps(prompt_templates, indent=2, ensure_ascii=False),
    encoding='utf-8',
)

print('Plantillas guardadas en:', PROMPT_DIR / f'{RUN_NAME}_prompt_templates.json')
print('Resumen few-shot:', prompt_examples_summary())
for strategy in PROMPT_STRATEGIES:
    print(f'\n--- {strategy.upper()} PREVIEW ---')
    preview_messages = prompt_templates[f'{strategy}_messages_preview']
    print(json.dumps(preview_messages[:4], indent=2, ensure_ascii=False)[:3000])

# %% [markdown] cell 20
# ## Celda 10: Inferencia con Qwen3-8B
#
# Esta celda ejecuta la clasificacion en `TEST` y `EVAL` para cada estrategia de prompt. La salida se fuerza a JSON para facilitar evaluacion reproducible.

# %% [code] cell 21
def generation_config_for_strategy(strategy: str) -> GenerationConfig:
    kwargs = dict(generation_kwargs)
    kwargs['max_new_tokens'] = MAX_NEW_TOKENS
    return GenerationConfig(**kwargs)


def generate_response_with_token_usage(row: pd.Series, strategy: str) -> tuple[str, dict]:
    messages = messages_for_row(row, strategy)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to(DEVICE)
    input_tokens = int(inputs['input_ids'].shape[-1])

    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config_for_strategy(strategy))

    generated_ids = outputs[0][input_tokens:]
    output_tokens = int(generated_ids.shape[-1])
    token_usage = {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': input_tokens + output_tokens,
    }
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip(), token_usage


def generate_response(row: pd.Series, strategy: str) -> str:
    raw_response, _ = generate_response_with_token_usage(row, strategy)
    return raw_response


def predict_dataframe(split_name: str, strategy: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    start_time = time.time()
    records = []

    for idx, row in df.iterrows():
        raw_response, token_usage = generate_response_with_token_usage(row, strategy)
        parsed = parse_model_response(raw_response)
        parsed.update(token_usage)
        records.append(parsed)
        if (idx + 1) % 25 == 0:
            print(f'{strategy}/{split_name}: {idx + 1}/{len(df)}')

    pred_df = df.copy().reset_index(drop=True)
    parsed_df = pd.DataFrame(records)
    pred_df = pd.concat([pred_df, parsed_df], axis=1)
    pred_df['correct'] = pred_df[LABEL_COLUMN] == pred_df['pred_label']
    pred_df['strategy'] = strategy
    pred_df['split_eval'] = split_name

    positive_scores = pred_df['confidence'].where(pred_df['pred_label'] == 1, 1 - pred_df['confidence'])
    pred_df['positive_score'] = positive_scores
    summary, label_metrics_df, report = compute_metrics_from_arrays(
        pred_df[LABEL_COLUMN].to_numpy(),
        pred_df['pred_label'].to_numpy(),
        positive_scores.to_numpy(),
    )
    roc_curve_df = build_roc_curve_dataframe(pred_df[LABEL_COLUMN].to_numpy(), positive_scores.to_numpy())
    elapsed_seconds = time.time() - start_time
    summary.update({
        'split': split_name,
        'strategy': strategy,
        'rows': int(len(pred_df)),
        'seconds': float(elapsed_seconds),
        'seconds_per_row': float(elapsed_seconds / max(len(pred_df), 1)),
        'input_tokens_total': int(pred_df['input_tokens'].sum()),
        'output_tokens_total': int(pred_df['output_tokens'].sum()),
        'total_tokens_inference': int(pred_df['total_tokens'].sum()),
        'input_tokens_per_row': float(pred_df['input_tokens'].mean()) if len(pred_df) else 0.0,
        'output_tokens_per_row': float(pred_df['output_tokens'].mean()) if len(pred_df) else 0.0,
        'total_tokens_per_row': float(pred_df['total_tokens'].mean()) if len(pred_df) else 0.0,
    })

    metrics_payload = {
        'summary': summary,
        'classification_report': report,
        'confusion_matrix': confusion_matrix(
            pred_df[LABEL_COLUMN],
            pred_df['pred_label'],
            labels=[0, 1],
        ).tolist(),
        'roc_curve': roc_curve_df.to_dict(orient='records'),
    }
    return pred_df, metrics_payload, label_metrics_df, roc_curve_df

# %% [markdown] cell 22
# ## Celda 11: Guardado de configuracion del experimento
#
# Se guarda la configuracion antes de ejecutar predicciones para dejar trazabilidad del modelo, dataset, prompts y decisiones metodologicas.

# %% [code] cell 23
tokenizer.save_pretrained(str(FINAL_MODEL_DIR))
generation_config.save_pretrained(str(FINAL_MODEL_DIR))

if SAVE_FULL_MODEL:
    model.save_pretrained(str(FINAL_MODEL_DIR), safe_serialization=True)

run_config = {
    'task': 'task2_contribution_extraction_binary',
    'approach': 'open_weight_llm_prompting_inference',
    'dataset_filename': DATASET_FILENAME,
    'dataset_path_used': str(dataset_path),
    'model_name': MODEL_NAME,
    'run_name': RUN_NAME,
    'execution_environment': 'jupyter_vm',
    'project_root': str(PROJECT_ROOT),
    'expected_gpu': EXPECTED_GPU,
    'labels': id2label,
    'text_column': TEXT_COLUMN,
    'label_column': LABEL_COLUMN,
    'split_column': SPLIT_COLUMN,
    'unused_split_column': UNUSED_SPLIT_COLUMN,
    'original_rhetorical_label_column': ORIGINAL_RHETORICAL_LABEL_COLUMN,
    'position_column': POSITION_COLUMN,
    'position_use': 'not_used_in_prompt_v5_retained_for_analysis',
    'prompt_strategies': PROMPT_STRATEGIES,
    'prompt_source': GEMINI_PROMPT_SOURCE_NOTE,
    'few_shot_summary': prompt_examples_summary(),
    'max_input_chars': MAX_INPUT_CHARS,
    'max_new_tokens': MAX_NEW_TOKENS,
    'max_text_words': MAX_TEXT_WORDS,
    'temperature': TEMPERATURE,
    'do_sample': DO_SAMPLE,
    'use_4bit': USE_4BIT,
    'cache_location': CACHE_LOCATION,
    'workspace_root': str(WORKSPACE_ROOT),
    'save_full_model': SAVE_FULL_MODEL,
    'random_state': RANDOM_STATE,
    'test_rows': int(len(test_df)),
    'eval_rows': int(len(eval_df)),
    'definition_alignment_notes': [
        'Open-weight 1-8B se evalua en modo inferencia, no fine-tuning.',
        'pos_rel no se incluye en el prompt v5; se conserva solo para analisis y trazabilidad del dataset.',
        'Prompts adaptados desde 06_task2_gemini_classifier.ipynb: zero-shot y few-shot k=6, sin pos_rel en el mensaje de inferencia.',
        'Dataset_type se usa como particion oficial del CSV.',
        'Notebook adaptado para Jupyter en VM local: datos/outputs en /home/jovyan y cache pesada en /workspace/qwen_task2.',
        'Se usa PyTorch global de la VM y dependencias Hugging Face instaladas en user-site.',
        'La variante Jupyter usa Qwen3-8B con cuantizacion 4-bit mediante bitsandbytes cuando USE_4BIT=True.',
    ],
}
RUN_CONFIG_FILE.write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding='utf-8')
joblib.dump(run_config, OUTPUT_DIR / f'{RUN_NAME}_run_config.joblib')

print('Artefactos del modelo/tokenizador guardados en:', FINAL_MODEL_DIR)
print('Configuracion guardada en:', RUN_CONFIG_FILE)

# %% [markdown] cell 24
# ## Celda 12: Predicciones y metricas en TEST y EVAL
#
# Se generan predicciones, metricas globales, ROC, consumo de tokens de inferencia, metricas por etiqueta y matrices de confusion para ambos conjuntos y ambas estrategias.

# %% [code] cell 25
all_metrics = {}
all_label_metrics = []
all_prediction_files = []

for strategy in PROMPT_STRATEGIES:
    for split_name, split_df in [
        ('test', test_df),
        ('eval', eval_df),
    ]:
        pred_df, metrics_payload, label_metrics_df, roc_curve_df = predict_dataframe(split_name, strategy, split_df)

        prefix = f'{RUN_NAME}_{strategy}_{split_name}'
        predictions_file = OUTPUT_DIR / f'{prefix}_predictions.csv'
        split_metrics_file = OUTPUT_DIR / f'{prefix}_metrics.json'
        label_metrics_file = OUTPUT_DIR / f'{prefix}_label_metrics.csv'
        confusion_png = OUTPUT_DIR / f'{prefix}_confusion_matrix.png'
        roc_curve_csv = OUTPUT_DIR / f'{prefix}_roc_curve.csv'
        roc_curve_png = OUTPUT_DIR / f'{prefix}_roc_curve.png'

        pred_df.to_csv(predictions_file, index=False, encoding='utf-8-sig')
        split_metrics_file.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding='utf-8')
        label_metrics_df.to_csv(label_metrics_file, index=False, encoding='utf-8-sig')
        roc_curve_df.to_csv(roc_curve_csv, index=False, encoding='utf-8-sig')
        save_confusion_matrix(f'{strategy} - {split_name}', pred_df, confusion_png)
        save_roc_curve(
            f'{strategy} - {split_name}',
            roc_curve_df,
            metrics_payload['summary'].get('roc_auc'),
            roc_curve_png,
        )

        key = f'{strategy}_{split_name}'
        all_metrics[key] = metrics_payload
        label_metrics_df = label_metrics_df.copy()
        label_metrics_df.insert(0, 'strategy', strategy)
        label_metrics_df.insert(1, 'split', split_name)
        all_label_metrics.append(label_metrics_df)
        all_prediction_files.append(str(predictions_file))

        print(f'Predicciones {key}:', predictions_file)
        print(f'Metricas {key}:', split_metrics_file)
        print(f'Curva ROC {key}:', roc_curve_png)
        print(f'Total tokens inferencia {key}:', metrics_payload['summary']['total_tokens_inference'])
        display(pd.DataFrame([metrics_payload['summary']]))
        display(label_metrics_df)

METRICS_FILE.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding='utf-8')

summary_metrics_df = pd.DataFrame([
    payload['summary'] for payload in all_metrics.values()
])
summary_metrics_df.to_csv(SUMMARY_METRICS_CSV, index=False, encoding='utf-8-sig')

combined_label_metrics_df = pd.concat(all_label_metrics, ignore_index=True)
combined_label_metrics_df.to_csv(OUTPUT_DIR / f'{RUN_NAME}_all_label_metrics.csv', index=False, encoding='utf-8-sig')

print('Metricas consolidadas guardadas en:', METRICS_FILE)
display(summary_metrics_df)

# %% [markdown] cell 26
# ## Celda 13: Analisis de errores
#
# Se guardan falsos positivos, falsos negativos y tablas por etiqueta retorica, posicion relativa y longitud del fragmento. Esto cubre el analisis cualitativo solicitado para Tarea 2.

# %% [code] cell 27
error_summaries = []

for strategy in PROMPT_STRATEGIES:
    for split_name in ['test', 'eval']:
        pred_file = OUTPUT_DIR / f'{RUN_NAME}_{strategy}_{split_name}_predictions.csv'
        pred_df = pd.read_csv(pred_file)
        error_summary = build_error_analysis(split_name, strategy, pred_df)
        error_summaries.append(error_summary)

ERROR_ANALYSIS_FILE = OUTPUT_DIR / f'{RUN_NAME}_error_analysis_summary.json'
ERROR_ANALYSIS_FILE.write_text(json.dumps(error_summaries, indent=2, ensure_ascii=False), encoding='utf-8')

print('Resumen de analisis de errores:', ERROR_ANALYSIS_FILE)
for item in error_summaries:
    print('\nSplit:', item['split'], '| Strategy:', item['strategy'])
    print('Errores:', item['errors'])
    print('Falsos positivos:', item['false_positives'])
    print('Falsos negativos:', item['false_negatives'])
    display(pd.DataFrame(item['top_rhetorical_error_groups']))
    display(pd.DataFrame(item['position_error_groups']))

# %% [markdown] cell 28
# ## Celda 14: Cierre
#
# Al terminar, revisa en Google Drive la carpeta de salida. Debe contener:
#
# - `final_model/`: tokenizer y generation config. Los pesos completos no se guardan por defecto.
# - `prompts/`: plantillas de prompt usadas en el experimento.
# - predicciones sobre `TEST` y `EVAL` para `zero_shot` y `few_shot`.
# - metricas JSON/CSV.`n- curvas ROC CSV/PNG y consumo de tokens de inferencia como input_tokens, output_tokens y total_tokens.
# - matrices de confusion.
# - falsos positivos, falsos negativos y resumen de errores.

# %% [code] cell 29
print('Carpeta de salida:', OUTPUT_DIR)
for path in sorted(OUTPUT_DIR.glob('*')):
    print(path)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

# %% [code] cell 30

