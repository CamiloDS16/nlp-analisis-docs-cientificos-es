#!/usr/bin/env python
# coding: utf-8

# Converted from: task1_multiclass_qwen2_3b_openweight_v16.ipynb
# NOTE: Notebook cell order is preserved. Markdown cells are comments.

# %% [code] cell 1
# Bootstrap reproducible para VM/Jupyter - Tarea 1.
# Ejecuta esta celda despues de cada reinicio de la VM si desaparecio /workspace/qwen_task1/qwen-env.
# Al terminar, cambia el kernel del notebook a: Python (qwen-task1)

import json
import os
import shutil
import site
import subprocess
import sys
from pathlib import Path

WORKSPACE_BOOTSTRAP = Path('/workspace/qwen_task1')
VENV_DIR = WORKSPACE_BOOTSTRAP / 'qwen-env'
PIP_CACHE_DIR = WORKSPACE_BOOTSTRAP / 'pip-cache'
HF_HOME_DIR = WORKSPACE_BOOTSTRAP / 'huggingface'
TMPDIR = WORKSPACE_BOOTSTRAP / 'tmp'
XDG_CACHE_HOME = WORKSPACE_BOOTSTRAP / 'xdg-cache'
VIRTUALENV_APP_DATA = WORKSPACE_BOOTSTRAP / 'virtualenv-app-data'
KERNEL_NAME = 'qwen-task1'
KERNEL_DISPLAY_NAME = 'Python (qwen-task1)'
KERNEL_DIR = Path.home() / '.local' / 'share' / 'jupyter' / 'kernels' / KERNEL_NAME

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
    'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn', 'joblib', 'ipython', 'ipykernel'
], env=venv_env)

validation_code = """
import torch, transformers, accelerate, bitsandbytes
import pandas, numpy, sklearn, matplotlib, seaborn, joblib, huggingface_hub, IPython
print('validacion:', torch.__version__, torch.version.cuda, torch.cuda.is_available(), transformers.__version__, accelerate.__version__)
print('torch path:', torch.__file__)
print('transformers path:', transformers.__file__)
print('bitsandbytes path:', bitsandbytes.__file__)
"""
subprocess.check_call([str(python_bin), '-c', validation_code], env=venv_env)

subprocess.check_call([
    str(python_bin), '-m', 'ipykernel', 'install', '--user',
    '--name', KERNEL_NAME, '--display-name', KERNEL_DISPLAY_NAME
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
print('Kernel instalado:', KERNEL_DISPLAY_NAME)
print('Python:', python_bin)
print('HF_HOME:', HF_HOME_DIR)
print('Ahora cambia el kernel a Python (qwen-task1), reinicia el kernel y continua con la siguiente celda.')

# %% [markdown] cell 2
# # Modelo open-weight `Qwen/Qwen3-8B` para Tarea 1 con prompt comercial DeepSeek
#
# Este notebook evalua el modelo de pesos abiertos usando el mismo prompt de inferencia empleado en el notebook comercial DeepSeek para la Tarea 1. La salida solicitada al modelo es un JSON de probabilidades por clase; para calcular metricas se toma la clase con mayor probabilidad y se mapea `METHO` a la etiqueta del dataset `METH`.

# %% [markdown] cell 3
# ## Celda 2: Imports
#
# Se importan utilidades para lectura flexible del dataset, inferencia generativa, calculo de metricas multiclase, visualizacion y persistencia de resultados.

# %% [code] cell 4

from __future__ import annotations

import gc
import json
import random
import re
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython.display import display
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, set_seed

# %% [markdown] cell 5
# ## Celda 3: Rutas de trabajo
#
# Esta celda permite ejecutar el notebook tanto en el equipo local como en Colab. Si se ejecuta en Colab, se intenta montar Google Drive de forma opcional; si se ejecuta localmente, se usa la ruta absoluta del proyecto indicada para este experimento.

# %% [code] cell 6

try:
    from google.colab import drive  # type: ignore
    IN_COLAB = True
except Exception:
    IN_COLAB = False

DATASET_FILENAME = "Dataset_consolidado_final_v4.csv"
DATASET_GOOGLE_DRIVE_FILE_ID = "1cwvxLBHeEzFUbs74E_037HhCcK2bb6ie"
DATASET_GOOGLE_DRIVE_URL = f"https://drive.google.com/file/d/{DATASET_GOOGLE_DRIVE_FILE_ID}/view?usp=sharing"

LOCAL_PROJECT_ROOT = Path('/home/jovyan')
LOCAL_DATASET_PATH = LOCAL_PROJECT_ROOT / "data" / DATASET_FILENAME
LOCAL_OUTPUT_BASE_DIR = LOCAL_PROJECT_ROOT / "models" / "tarea_1" / "open_weight_v14"

VM_PROJECT_ROOT = Path("/home/jovyan")
VM_DATA_DIR = VM_PROJECT_ROOT / "data"
VM_OUTPUT_BASE_DIR = VM_PROJECT_ROOT / "outputs_t1_ow"

if IN_COLAB:
    drive.mount("/content/drive")
    # Ajusta estas rutas si copias el dataset a otra carpeta de Google Drive.
    DRIVE_ROOT = Path("/content/drive/MyDrive/ProyectoMAIA")
    DATASET_PATH = DRIVE_ROOT / "data" / DATASET_FILENAME
    OUTPUT_BASE_DIR = DRIVE_ROOT / "outputs" / "task1_multiclass_qwen3_8b_openweight_v14"
elif VM_PROJECT_ROOT.exists() or Path("/workspace").exists():
    DATASET_PATH = VM_DATA_DIR / DATASET_FILENAME
    OUTPUT_BASE_DIR = VM_OUTPUT_BASE_DIR / "task1_multiclass_qwen3_8b_openweight_commercial_prompt_v14"
else:
    DATASET_PATH = LOCAL_DATASET_PATH
    OUTPUT_BASE_DIR = LOCAL_OUTPUT_BASE_DIR / "outputs_task1_multiclass_qwen3_8b_openweight_commercial_prompt_v14"

DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

print("IN_COLAB =", IN_COLAB)
print("DATASET_PATH =", DATASET_PATH)
print("DATASET_GOOGLE_DRIVE_URL =", DATASET_GOOGLE_DRIVE_URL)
print("OUTPUT_BASE_DIR =", OUTPUT_BASE_DIR)

# %% [markdown] cell 7
# ## Celda 4: Configuracion principal
#
# Se definen modelo, columnas oficiales, etiquetas validas, estrategias de prompting y limites de inferencia. Por defecto el notebook calcula resultados en `EVAL`; si se desea comparar contra `TEST`, se puede agregar `"test"` a `INFERENCE_SPLITS`.

# %% [code] cell 8

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
RUN_NAME = "task1_multiclass_qwen3_8b_openweight_commercial_prompt_v13"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
SPLIT_COLUMN = "dataset_type"

LABEL_NAMES = ["BACK", "INTRO", "METH", "RES", "DISC", "CONC", "CONTR", "LIM"]
label2id = {label: idx for idx, label in enumerate(LABEL_NAMES)}
id2label = {idx: label for label, idx in label2id.items()}

RANDOM_STATE = 42
MAX_TRAIN_REFERENCE_ROWS = 240
MAX_TEST_ROWS = None
MAX_EVAL_ROWS = None

MAX_INPUT_CHARS = 6500
MAX_NEW_TOKENS = 96
TEMPERATURE = 0.0
DO_SAMPLE = False

PROMPT_STRATEGIES = ["cot"]
INFERENCE_SPLITS = ["eval"]  # Cambiar a ["test", "eval"] si tambien se quiere medir TEST.
DEFAULT_STRATEGY_FOR_ERROR_ANALYSIS = "cot"

USE_4BIT = True
SAVE_FULL_MODEL = False

OUTPUT_DIR = OUTPUT_BASE_DIR
FINAL_MODEL_DIR = OUTPUT_DIR / "final_model"
PROMPT_DIR = OUTPUT_DIR / "prompts"
FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
PROMPT_DIR.mkdir(parents=True, exist_ok=True)

RUN_CONFIG_FILE = OUTPUT_DIR / f"{RUN_NAME}_run_config.json"
METRICS_FILE = OUTPUT_DIR / f"{RUN_NAME}_metrics.json"
SUMMARY_METRICS_CSV = OUTPUT_DIR / f"{RUN_NAME}_summary_metrics.csv"

print("MODEL_NAME =", MODEL_NAME)
print("RUN_NAME =", RUN_NAME)
print("LABEL_NAMES =", LABEL_NAMES)
print("PROMPT_STRATEGIES =", PROMPT_STRATEGIES)
print("INFERENCE_SPLITS =", INFERENCE_SPLITS)
print("GPU disponible =", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU =", torch.cuda.get_device_name(0))
else:
    print("Para Qwen3-8B se recomienda ejecutar con GPU.")

# %% [markdown] cell 9
# ## Celda 4B: Prompt comercial DeepSeek
#
# Se replica la estructura de ejemplos del notebook comercial. La inferencia usara `strategy="cot"`, que fue la estrategia usada en la evaluacion comercial.

# %% [code] cell 10
# Prompt extraido de Deepseek_Comercial_Model_T1_II_VF.ipynb
PROMPT_EXAMPLES = {

# ======================================================
# FEW SHOT
# ======================================================
"few-shot": {


"ejemplos": [

# ---------- CONTR (2) ----------
{"texto":"evaluar los indicadores de gestin del rea de desorcin de carbn activado con la propuesta de implementacin de un sistema automatizado en la etapa de carguo se estima mejorar de 7 200 segundos a 1 200 segundos equivalente a un 833 la reduccin de tiempo total anual en el carguo es de 1 728 000 segundos equivalente a 480 el proceso proyectado de carguo automtico de carbn activado a la columna es de 20 minutos reduccin en tiempo de carga de 100 minutos obteniendo un ahorro estimado de s 3 25000 por ao evaluar los resultados de la mejora y su relacin con los costos de produccin del rea de desorcin de carbn activado se estima una mejora en la ergonoma de la operacin en relacin con los operarios de planta en un 38 se estima una mejora en el ndice de productividad de un 10 en comparacin a la media aritmtica de la productividad de junio y julio el clculo del tir es de 74 mayor al cok de la empresa que es de 8 el clculo del valor del van es de s 32 38206 en un periodo de 5 aos el retorno de la inversin es de 143 aos 1 ao y 5 meses automatizacion control y optimizacion de los procesos de produccion de un complejo metalurgico autor jorge morales castelan mxico df 2011 el objetivo de este estudio de caso es el de compartir las experiencias y resultados en la concepcin desarrollo implantacin y mantenimiento del proyecto estratgico de automatizacin control y optimizacin de las reas de produccin de cinco empresas del negocio metales del grupo peoles en el ao 2005 se nos asign la responsabilidad de la automatizacin de los procesos de produccin de las plantas de la divisin metales del grupo peoles la primera tarea que realizamos en colaboracin con un grupo de expertos en la automatizacin de procesos industriales fue la de evaluar en forma general los aspectos ms importantes en automatizacin entre las que se encuentran 1 eficiencia de los sistemas instalados 2 capacidad de los sistemas instalados 3 nivel de automatizacin 4 complejidad de los procesos susceptibles de ser automatizados 5 nivel de satisfaccin de las reas operativas del funcionamiento de los procesos 6 nivel de estandarizacin de la tecnologa 7 capacidades de los proveedores de tecnologa 8 nivel de eficiencia en la gestin de desarrollo y mantenimiento de los sistemas de control el resultado de este primer estudio resalt dos aspectos importantes 1 que tenamos grandes reas de oportunidad de mejora en todos los aspectos de automatizacin desde la forma en que se seleccionan los procesos a ser automatizados la forma en que se gestionan estos proyectos la seleccin de tecnologa a ser implantada la seleccin de proveedores el buen uso y el mantenimiento de los sistemas instalados entre otros 2 el tamao de las plantas y la complejidad de los procesos operativos nos indic que la automatizacin de los mismos debera ser tratada como un proyecto de alto nivel de complejidad y que los niveles de inversiones seguramente seran elevados por tal motivo con el objetivo final de mejorar la eficiencia de los procesos de produccin propusimos y fue aceptada la elaboracin de un proyecto estratgico de automatizacin control y optimizacin que identificara los procesos claves a ser automatizados el nivel de inversiones requeridas y los tiempos de ejecucin de los proyectos el concebir vender la idea planear ejecutar y mantener un proyecto estratgico para la automatizacin control y optimizacin de los procesos de produccin de uno de los complejos metalrgicos ms grandes del mundo es un trabajo retador complejo y altamente gratificante para el desarrollo e implantacin de este proyecto se requiri la participacin de profesionistas de las siguientes especialidades ingenieros qumicos ingenieros en electrnica ingenieros elctricos ingenieros en mecatrnica ingenieros o licenciados en informtica maestros en administracin y contadores principalmente estos especialistas aportaron sus conocimientos en disciplinas tales como operacin de los procesos de produccin automatizacin control avanzado y optimizacin planeacin estratgica administracin de proyectos instrumentacin ingeniera elctrica ingeniera mecnica evaluacin de proyectos seguridad ecologa ingeniera de procesos y de detalle evaluacin de tecnologas de medicin y control relaciones interpersonales administracin del cambio administracin del conocimiento administracin de la calidad control presupuestal planes de contingencia redes de control administracin del ciclo de vida de las inversiones tecnolgicas entre otros las experiencias que podemos compartir despus de aproximadamente 5 aos de trabajar en este importante y complejo proyecto son las siguientes 1 las inversiones en automatizacin son altamente rentables 2 los proyectos deben estar perfectamente alineados a los planes estratgicos del negocio 3 el nivel de automatizacin de una empresa es un indicador significativo en su nivel de 4 se requiere trabajar en una forma profesional en la seleccin de tecnologas de medicin control y optimizacin 5 los resultados de las inversiones en automatizacin deben reflejarse en los resultados financieros y operativos del negocio 6 la automatizacin requiere de expertos en sus diferentes disciplinas 8 se requieren expertos en la administracin de proyectos 9 debe haber un fuerte nivel de liderazgo","label":"CONTR"},
{"texto":"gestion por procesos mediante un estudio mixto empleo los nstrumentos de la ficha herramienta de gestion elimino el exceso de tiempos muertos cuellos de botella desfaces en el control de inventarios perdidas fisicas de productos duplicidad de actividades y a su vez potencializo la sistematizacion de procesos y procedimientos tambien promovio la practica continua de las bpfa bpa y la revision de reglamentos y normativas vigentes el estudio es importante porque permite formular las posibles respuestas del estudio respecto a que la gestion de procesor logra disminuir tiempos que pueden originarse por cuellos de botella asi como un inadecuado control de los inventarios como parte del ciclo de almacenamiento cardenas 2017 en el estudio denominado procedimiento para la gestion de inventarios de medicamentos en la farmacia nuestra senora de regla con el objetivo general de aplicar una herramienta para la gestion del inventario mediante una investigacion de tipo cualitativo biblografico se aplico una ficha de observacion para los procesos de almacenamiento cuya toma de datos se realizaron en 3 dias con el metodo de toma de datos de los tiempos asi como el metodo abc de inventarios concluyendo que existe una amplia base conceptual para gestionar los inventarios marcadorretorico es de vital importancia ajustar los sistemas y procedimientos a las caracteristicas inherentes de cada entidad como parte de la gestion de procesos en el lamacenamiento facilmnete se puede recurrir a herramientas que permitan de cierto modo establecer el orden adecuados de los productos esto facilita el porceso de despacho y se puede realizar mas rapido estudios en el territorio peruano respecto a las variables de estudio se tiene rivas 2019 en el estudio titulado gestion de procesos para mejorar el almacenamiento en una empresa comercializadora de productos farmaceuticos e instrumental medico con el objetivo general de proponer estrategias para mejorar la gestion de almacen mediante el metodo inductivo deductivo la poblacion fueron 30 trabajadores de almacen aplico como tecnicas la encuesta y analisis de datos concluyendo que se elaboraron distintos formatos como el mapa de procesos diagrama de ishikawa y de pareto y diagrama de analisis de procesos dap enfocados en la preparacion de pedidos y despacho lograndose determinar que las existencias de productos no conformes generan la insatisfaccion del cliente en la evaluacion de procesos de requieren de herramientas de disgnostico como causa y efecto para la correcta identificacion de los problemas de la empresa junto con el dap que se pueden emplear para el ciclo de almacenamiento ochoa 2018 abordo la gestion por procesos para mejorar el almacen de una empresa comercializadora de repuestos del sector automotriz para elaborar un plan de mejoras en los procesos de la gestion de almacen de la empresa la metodologia consistio en un enfoque mixto de tipo proyectiva constituido por 57 colaboradores y 3 jefes de la empresa aplicpo como instrumentos la ficha de recoleccion de datos y la ficha de clasificacion abc asi segmentando las marcas por el nivel de rotacion alto medio y bajo y un sistema de codificacion por pasillos con el fin de agilizar la busqueda de los productos para el proceso de la preparacion de pedidos la investigacion es importante porque destaca entre las mejorar del almacenamiento la clasificacion que es un herramienta logistica con alto grado de aceptacion para agilizar y evitar los cuellos de botellas en el proceso de despacho y para la identificacion rapida de un producto en el almacen la codificacion de cada uno de los pasillos tambien estandariza la ubicacion con el cual el personal muy facilmente puede identificar la ubicacion del producto leon 2018 en la tesis para fortalecer las buenas practicas de almacenamiento de medicamentos en el personal responsable de farmacia en la red de salud pacifico sur con el objetivo general de elaborar adecuadas practicas de almacenamiento mediante de la empresa concluyendo sobre el personal necesita un adecuado conocimiento para la implementacion de las buenas practicas de almacenamiento de los productos farmaceuticos debe recibir capacitacion sobre la calidad y eficacia asi como las mejores practicas de almacenamiento respecto al ciclo de almacenamiento el estudio no solo repara en los procesos sino que involucra al personal quien debe ser parte para que se apliquen las buenas practicas de manufactura asi como las constante capcitacion para que se logre las capacidades y competencias necearias para lograr procesos eficiente chafloque y huari 2017 en el estudio denominado propuesta de mejora en los procesos de reposicion de stock recepcion e indicadores de gestion del almacen en la trabajadores a fin de conocer la situacion actual del almacen y el trabajo de campo mediante la observacion concluyendo que los operarios son claves para la conservacion de los inventarios en la empresa tambien muestra respecto a la metodologia que la mejor tecnica es la observacion con el cual se puede obtener importante informacion sobre el diagnostico del proceso de alamacenamiento de este se pueden despegar las mejoras del mismo","label":"CONTR"},

# ---------- INTRO (2) ----------
{"texto":"la apicultura constituye una actividad de gran importancia no solo por los aportes economicos sino por los beneficios medioambientales la gestion del sector apicola en castilla y leon espana la regula la consejeria de agricultura y ganaderia conforme a lo previsto en la orden ayg21552007 de de diciembre donde se regula el registro de explotaciones apicolas el movimiento de colmenas y se aprueba el modelo de libro de registro de explotacion apicola todas las explotaciones apicolas de la provincia de leon se encuentran registradas en la base de datos del registro de explotaciones ganaderas existiendo explotaciones y colmenas una vez analizada esta base de datos se identificaron varias deficiencias que generan conflictos a la hora de dar de alta a nuevas explotaciones debido a la cantidad de informacion a procesar al incremento de colmenares que se esta produciendo durante los ultimos anos en espana y al estado actual del registro de explotaciones ganaderas de castilla y leon se ha propuesto desarrollar una plataforma basada en sistemas de informacion geografica para la creacion y mantenimiento de una base de datos georreferenciada existen otros trabajos donde se han utilizado los sig para planificar y gestionar el inventario de recursos naturales en este trabajo se ha creado una base de datos georreferenciada basada en la base topografica nacional a escala btn25 gestionada con software libre y organizada en seis unidades tematicas se ha realizado un inventario georreferenciado de","label":"INTRO"},
{"texto":"las rocas igneas son aquellas que se producen a partir del enfriamiento y la solidificacion del magma es muy importante poder caracterizar quimicamente este tipo de rocas ya que esto hace posible clasificarlas modelar los procesos de genesis y evolucion de los magmas que llevaron a su formacion y determinar el ambiente tectonico generador la composicion quimica de las rocas es evaluada mediante la utilizacion de diagramas especificos y es relevante para el mapeo geologico basico para estudios petrogeneticos y de evolucion tectonica y para la prospeccion de recursos minerales desde hace muchos anos se utiliza la informacion de la composicion quimica de las rocas para caracterizar los diferentes tipos de rocas e identificar sus origenes elementos quimicos mayoritarios asi como otros presentes en concentraciones trazas son seleccionados de acuerdo a sus caracteristicas de movilidad y representatividad del tipo de roca por este motivo varios elementos han sido estudiados por diferentes autores para la construccion de diagramas de discriminacion adecuados para la clasificacion en este trabajo de tesis se busco desarrollar metodos analiticos de tratamiento de muestras y de analisis de elementos traza y mayoritarios en rocas para su aplicacion en la clasificacion y caracterizacion de rocas igneas de uruguay buscando hacer disponible una plataforma analitica a nivel nacional para grupos de investigacion para esto se optimizo un procedimiento de tratamiento de muestras que fuera eficiente para los fines propuestos ","label":"INTRO"},

# ---------- BACK (2) ----------
{"texto":"memoria presente y perspectivas de un eauee teologico 1 zarazagac omentario a la exposicion de marcelo cionzalezdebate39205205209209218 palabras de aperturadel rector de las facultades de filosofia y teologiajose maria canto sibuenos dias a todos su excelencia reverendisima monsenor dr juan carlos maccarone obispo de santiago del estero autoridades academicas de distintos centros que nos acompanan estimados colegas profesores queridos alumnos con estas palabras quiero darles la bienvenida a esta jomada este ano estamos celebrando los 70 anos de este colegio maximo y el ano proximo celebraremos los 70 anos de la aprobacion pontificia de estas facultades de filosofia y teologia lo cual significa toda una trayectoria a lo largo de la cual desde esta casa se han acompanado los cambios sociales politicos y culturales desde un centro de pensamiento que ha permanecido siempre en dialogo abierto con la realidad de nuestro pais y con america latina hemos sido tambien a lo largo de estos anos testigos de una gran transformacion en la iglesia centrada sobre todo en el gran acontecimiento del concilio vaticano ii y tambien de una forma renovada de producir y de trasmitir la reflexion teologica hoy en esta jomada academica queremos celebrar dos aniversarios significativos para la filosofia y la teologia en la argentinavoy a pedirle ahora al p juan carlos scannone que durante diez anos y hasta hace poco ha sido rector de estas facultades y sobre quien ha recaido el principal peso de la organizacion de esta jomada que nos presente brevemente cual es el sentido que hemos querido darle a esta reflexion filosofica y teologicastromata 58 2002 1218","label":"BACK"},
{"texto":"negociacion todas estas caracteristicas ayudan a que la persona afronte dichas situaciones de manera positiva sin embargo en la actualidad se observo que los menores no desarrollan las habilidades de interaccion social de forma adecuada lo cual afecta de modo global a la salud somatica mental y social en el proceso del ser humano entre ellos se destaca el bullying y la violencia familiar por otro lado esta la violencia escolar bullying actualmente este problema se encuentra en diversos colegios del pais asociados a carencias de habilidades sociales a bajos niveles de autoestima y auto percepcion comportamientos delictuosos en la pubertad y juventud y evidentemente dificultades de acomodacion al contexto escolar holguin 2016 con respecto a la i e san juan bautista hemos observado que los estudiantes no desenvuelven apropiadamente sus destrezas sociales generando algunos de los problemas anteriormente mencionados investigando sobre el curso historico hallamos a monjas 2002 manifestando que a traves de las relaciones interpersonales los ninos experimentan ciertos conflictos generados durante la infancia estigmatizando que podrian ser entre un y un de la poblacion repercutiendo en aspectos como el rendimiento escolar y la autoestima esto se prolonga en la ninez y en la juventud por otro lado el diario peru 2014 senala que en este ano el de escolares ha recibido agresiones por parte de sus padres de esta manera se comprueba que el de adultos solucionan los problemas en casa mediante la violencia puesto que dentro del sistema familiar no existe un adecuado soporte socioemocional entre los miembros el cual no permite que el nino se desarrolle apropiadamente en su entorno social del mismo modo el diario peru 2016 anuncio un caso de violencia escolar en la ciudad de lima la victima fue una nina de anos con iniciales r f a y su agresor un companero de aula de anos de iniciales m g b el la molesta","label":"BACK"},

# ---------- METHO (2) ----------
{"texto":"a los alarifes juan ruiz y sebastian ruiz comprueban que ciertas obras cumplen lo estipulado archio de protocolos notariales de granada protocolo de juan de ahedo 15434 fol 57 4r fol 574r juan ruyz e sebastian ruyz alarifes desta ibdad de baa nonbrado por la ibdad della sabed que fue rematada ierta obra de seys pilares que se han de hazer en el alcaava desta ibdad en juan de bandoman en preio de veynte e un mili maravedis con iertas condiiones e agora el dicho juan de bandoma dize que tiene hechos los dichos seys pilares conforme a lo que se obligo e tiene hechas iertas mejorias que convenian a la dicha obra hazerse que me pedia que lo hiziese ver de como el tenia conplido lo que estava obligado a su magestad e al contador chriptoval lopez en su nonbre e ansi mismo lo jorias que convenian a la dicha obra hazerse que me pedia que lo hiziese ver de como el tenia conplido lo que estava obligado a su magestad e al contador chriptoval lopez en su nonbre e ansi mismo lo mejorado dello e si convenian hazerse e lo que valen e yo visto su pedimento di el dicho presente por el qua os mando que veays las dichas condiciones e obligaion que hizo el dicho juan de bandoma ante juan de ahedo escrivano publico e ved si estan hechos los dichos seys pilares e arcos conforme a la dicha obligaion e demas desto o mejorado que hizo e si convenia hazerse e lo que pueden valer e todo con juramento venid a lo declarad ante mi para que yo provea justiia fecho en baa a illl de junio de i u dxliili anos el bachiller mexia rubrica juan de ahedo escrivano publico rubr","label":"METHO"},
{"texto":"superficie total 639000 ha cultivada con soja en el chaco inta 2011 el equipo de investigacion definio el criterio para seleccionar los sujetos que podian facilitar informacion por consiguiente la seleccion de la poblacion en cuanto se refiere al tamano de empresa y zona geografica fue de caracter intencional dado que incluyo a ochenta y cuatro productores sojeros con explotaciones de hasta 200 hectareas cultivadas ubicadas en la sub zona xvib que configuraron la poblacion de este estudio muestra para la aplicacion del instrumento ad hoc se efectuo una seleccion probabilistica del grupo definitivo asi la muestra aceptante fox 1981 de pequenos productores sojeros chaquenos quedo conformada por individuos escogidos utilizando el metodo aleatorio simple teniendo todos ellos o el metodo aleatorio simple teniendo todos ellos la misma probabilidad de integrarla la representatividad se encuentra respaldada por las siguientes caracteristicas a contiene todos los departamentos de la zona agricola que aporto la mayor parte de la superficie sembrada con soja en el chaco durante el ano 2011 y anterioresb la ubicacion geografica de las empresas se realizo a partir de estudios previos realizados por el instituto nacional de tecnologia agropecuaria intac el tamano muestral es equivalente al 3214 de la poblacion por lo que se presume que el numero de observaciones realizadas es razonabled el metodo probabilistico de extraccion de los casos particulares que fueron entrevistados consolidaron las fortalezas de la representatividadconforme lo expresado sobre pobla epresentatividadconforme lo expresado sobre poblacion y muestra y a efectos de sustentar las decisiones metodologicas asumidas se sostiene que 1 el muestreo no probabilistico para determinar la poblacion significo la adopcion de un criterio subjetivo e implico a un esfuerzo deliberado para obtener una muestra representativa mediante la inclusion de sujetos tipicos y b la optimizacion de los recursos disponibles sin embargo no resulto arbitrario sino de la conjuncion de antecedentes relevantes y pertinentes lo que acoto las limitaciones emergentes de este tipo de muestreo 2 el muestreo probabilistico utilizado en la extraccion de la muestra aceptante concede mayor rigurosidad metodologica y representatividad asi las conclusiones obtenidas explican a toda la poblacion aunque","label":"METHO"},

# ---------- RES (2) ----------
{"texto":"componentes del sistemala diferencia entre un estado de paro por falla y un estado de reparacion por fallaconsiste en que en e primer caso no se realizan trabajos de reposicion de lacapacidad de trabajo del objeto interrumpida a consecuencia de la falla en tantoque en el segundo si 4si al considerarlo como un componente se analiza un sistema electrico o unaparte de este para determinar el estado de sus componentes normalmente seasume los conceptos de regimen normal de falla y de postfallala contabilidad la caracterizan las ocurrencias de paro total o parcial fallalocalizacion del paro y reposicionfalla se considera como el paso del componente de un estado de capacidad detrabajo o de un nivel de funcionamiento relativo a otro relativamente mas bajocon una grave interrupcion d o y reposicionfalla se considera como el paso del componente de un estado de capacidad detrabajo o de un nivel de funcionamiento relativo a otro relativamente mas bajocon una grave interrupcion del regimen de trabajoparo consiste en la destruccion parcial o total de un componente es decir es elpaso del componente de un nivel de capacidad de trabajo o de funcionamiento aotro mas bajo o a un nivel de inutilidad totalsalida es el hecho de que un componente no se encuentre disponible para laoperacion pudiendo ser causado por una falla u otra causa cualquiera no todasalida es provocada por una falla sin embargo toda falla ocasiona que undeterminado componente salga de serviciola localizacion del paro de funcionamiento consiste en delimitar lasconsecuencias del paro del compone","label":"RES"},
{"texto":"a calidad de vida empeoro mucho por el dolor incluso despues de uno o dos anos del procedimiento quirurgico por estas razones se han propuesto otros accesos en los que se deben preservar los musculos principales la toracotomia conservadora del musculo mst 24 mas que esto la reduccion del dolor se puede obtener con toracotomias con preservacion nerviosa nst porque el problema es que durante la cirugia se danan dos ramas neurovasculares en la costilla superior el nervio se contrae mientras que en la costilla inferior se comprime debido al cierre sutura asi evitando esto se disminuye el dolor la literatura sugiere que incluso una tecnica quirurgica avanzada como la nst no reduce todas las morbilidades postoperatorias y estas estan condicionadas por muchos factores identificados el dolor la literatura sugiere que incluso una tecnica quirurgica avanzada como la nst no reduce todas las morbilidades postoperatorias y estas estan condicionadas por muchos factores identificados 10 en el intraoperatorio como la longitud quirurgica y en el postoperatorio como la presencia y el numero de tubos toracicos y especialmente la tecnica analgesica que se utiliza un gran cambio se produjo con el advenimiento de la cirugia toracica asistida por video vats en las ultimas dos decadas que hizo que todas las atenciones se centraran en la evaluacion de la calidad de vida entre la vats y la cirugia abierta sobre la base de varios estudios se ha demostrado que en el primer periodo postoperatorio la vats garantiza mejores resultados en terminos de puntaje de dolor y deterioro de","label":"RES"},

# ---------- DISC (2) ----------
{"texto":"una practica de fronteras borrosas se trata de un concepto operativo que arrastra desde sus origenes con una referencia un tanto vaga y generica su ambiguedad semantica en parte no es sino el resultado de una larga discusion acerca de si ella debe ser entendida como un instrumento medico o como una practica etica subsidiaria de la salud espiritual mientras que algunas escuelas o tradiciones de psicoterapia se esfuerzan por mostrarse emparentadas con la practica medica pensemos los desarrollos de la psiquiatria moderna otras se definen como una terapia eminentemente moral y en este ultimo caso recordemos que en el siglo xix la psicoterapia se llamaba originalmente terapia moral justamente en oposicion a la terapia corporal10 esta discusion epistemologica que atraviesa la conformacion moderna de las ciencias de la salud mental da cuenta de la existencia del modelo etico y del modelo medico como los dos paradigmas antiteticos fundamentales en los que oscila fundamentalmente la discusion acerca de estatuto epistemologico de la psicoterapia11 con la intencion de radicalizar el planteo podriamos ilustrar el modelo medico con lo que se reconoce hoy en dia como la medicina o la psiquiatria basada en la evidencia esta tradicion que ubica la psicoterapia en el locus de la medicina concibe los problemas mentales primariamente como una disfuncion objetiva por cuyos sintomas es susceptible de ser clasificada en categorias discretas de enfermedades al igual que las disfunciones fisicas pensemos los esfuerzos que caracterizan a la psiquiatria desde la segunda mitad del siglo xviii por diferenciar y clasificar progresivamente las enfermedades mentales existe a la sazon un intento primario de encontrar una base anatomopatologica de dichas enfermedades es decir una lesion cerebral especifica en la que sustentar esta clasificacion luego junto al fracaso de esta orientacion por no encontrarse ningun tipo de lesion en la mayoria de las enfermedades psiquicas la posterior adopcion del criterio nosotaxico segun el cual los trastornos son discernibles en virtud de su etiologia su clinica su evolucion su pronostico y su respuesta a un particular tratamiento en nuestros dias podriamos decir que las diversas ediciones del manual diagnostico y estadistico de los trastornos 10 ciertamente que cuando se hablaba de terapia moral se tomaba en sentido amplio como sinonimo de lo que hoy se llama psiquico o conductual 11 la medicina moderna tiende a distinguir entre la salud fisica y espiritual dependiendo si los sintomas de una enfermedad son primariamente fisicos o mentales naturalmente este dualismo es absolutamente ajeno a la comprension que el primer monacato cristiano tiene acerca del hombre y la salud los padres del desierto conciben la salud como un fenomeno holistico e inclusivo que se refiere a la persona en su integridad cuerpo alma y espiritu ahora bien en cuanto que nuestro objetivo primario es delimitar el alcance psicoterapeutico del tratamiento que hace los padres del desierto nos limitaremos a discernir los dos modelos de salud mental vigentes en la psicologia contemporanea marcadorretorico es necesario hacer la salvedad de que se trata de un recurso de simplificacion que no corresponde con la vision holistica de las fuentes proyecciones y alcances psicoterapeuticos de la doctrina evagriana de la salud del alma 417 mentales junto a los desarrollos de las neurociencias forman parte medular de los contenidos y estructuras indispensables para pensar la tarea de la psicoterapia desde este modelo medico en contraposicion a este modelo medico de salud mental se ha formulado en las ultimas decadas lo que se reconoce como el modelo positivo de salud mental se trata de una nueva aproximacion al concepto de salud humana que presenta las virtudes como una especie de antitesis de los trastornos psicopatologicas categorizados en el dsm en lugar de entender la salud mental como la ausencia de enfermedad tal como lo hacen las clasificaciones psicopatologicas del dsm la entiende como el desenvolvimiento positivo de las potencialidades humanas a traves de la actividad virtuosa seligman and csikszentmihalyi 2000 snyder and mccullough 2000 seligman 2002 emmons 2004 la salud humana no es sino el estado de plenitud emocional y espiritual que resulta de la practica virtuosa desde este marco comprehensivo el estatuto epistemologico de la psicoterapia se inscribiria fundamentalmente en una dimension etica y pedagogica ella es concebida no como parte de la medicina sino mas bien como una poiesis orientada a la de formacion moral del paciente mccullough kilpatrick emmons y larson 2001 davidson 2005 seligman steen park and peterson 2005 peterson and seligman 2004 dahlsgaard peterson and seligman 2005 peterson 2006 un antecedente importante de estos planteos los hallamos en la obra del psiquiatra vienes rudolf allers 1963 1516 desarrollada en la primera mitad del siglo xx quien formula una psicologia antropologica en oposicion al modelo medico argumenta fundamentalmente que en cuanto que la psicoterapia procura modificar desordenes animicos o conductuales apelando a arte medico dicho en otros terminos la psicoterapia al entrar en relacion directa con la libertad del paciente es una actividad de orden moral que implica necesariamente la luz practica de la ciencia moral ciertamente la inteligibilidad del valor moral agrega dependera del ideario antropologico de la tradicion psicoterapeutica y del paciente mas ello no es obice para asignarle y reconocerle a la tarea psicoterapeutica un estatuto etico sirvan los modelos citados para ilustrar una compleja y extensa discusion que se extiende hasta nuestros dias acerca del estatuto epistemologico de la sea que los desarrollos de evagrio se entiendan como un aporte al modelo medico de salud mental o en el marco del cuidado moral del alma debe entenderse la continuidad que existe para el entre estos dos abordajes que las 418 maria teresa gargiulo santiago hernan vazquez","label":"DISC"},
{"texto":"en diferentes localidades los pobladores matan a la vbora de cascabel crotalus simus por el uso medicinal que se le atribuye o para venderlas a personas que provienen principalmente de otras comunidades a un precio que vara desde 50 hasta 3000 pesos dependiendo del tamao del ejemplar el uso medicinal ms frecuente fue para curar el cncer 53 aunque tambin se mencionaron espanto 16 dolor en general 13 dolor de anginas fiebre tos cicatrizacin de heridas y para tener hijos slo una persona coment que tambin la coralillo micrurus sp es usada con fines medicinales el 73 piensan que no se deben de proteger a las serpientes venenosas aunque hubo diferencias significativas en las respuestas entre personas con distintos niveles de escolaridad x2 2072 gl 4 p 005 las que creen que se debe proteger cuentan con estudios ej primaria secundaria preparatoria etc no hubo diferencias significativas analizando las respuestas por grupos tnicos x2 0198 gl 1 p se obtuvieron mejores resultados en la modelacin para c simus m diastema y m elegans valores de auc mayores a 09 y p 005 con los que se consiguieron modelos con un alto poder predictivo los valores para m diastema auc 0865 p 005 son considerados razonablemente buenos peterson et al 2011 el sustento estadstico de los modelos podra estar asociado al proceso de depuracin de datos mediante el cual fueron eliminados los registros que se encontraban fuera del rea conocida y espacio ambiental de las especies estudiadas la distribucin de m elegans y b aurifer hacia el sur parece estar limitada por el valle de motagua en guatemala este valle ha ocasionado la divergencia de serpientes de los gneros bothriechis atropoides y cerrophidion y acta como una barrera geogrfica para las especies de serpientes que bothriechis aurifer es una de las especies de serpientes de la altiplanicie de chiapas ms amenazadas por su distribucin restringida y la acelerada destruccin de su hbitat durante el trabajo de hidalgogarca et al serpientes venenosas y distribucin en chiapas campo se pudo apreciar que su hbitat est siendo severamente afectado principalmente por la prctica de rozatumbaquema esta tendencia de disminucin de sus poblaciones por prdida de hbitat tambin ha sido reportada recientemente en guatemala iucn 2017 su distribucin al norte lmites de la altiplanicie de chiapas con las montaas del norte se encuentra limitada por la presencia de bothriechis rowleyi una especie emparentada castoe et al 2009 y al sur por el valle de motagua el cual ha actuado observado en santiago guelatao y el ejemplar colectado en felipe carrillo puerto cerca de cruz del rosario en el municipio de las margaritas corresponden a dos nuevos registros de localidad con los cuales su distribucin se extiende aproximadamente 43 km al noreste hidalgogarca j a lunareyes r cedeovzquez j r gonzlezsols d datos no publicados tomado como referencia el registro previo ms cercano reportado por campbell lamar 2004 en el parque nacional lagunas de montebello la temperatura media anual fue la variable ms importante para b aurifer que es una especie caracterstica de lugares fros generalmente por encima de los 1500 msnm marcadorretorico la humedad tambin parece ser un factor importante en su distribucin ya que se encuentra en bosques mesfilos de montaa y vegetacin asociada campbell lamar 2004 la distribucin de c simus en la vertiente del pacfico se encuentra demarcada por el istmo de tehuantepec el cual ha sido una barrera geogrfica para especies caractersticas de tierras bajas y altas y","label":"DISC"},

# ---------- CONC (2) ----------
{"texto":"este resultado es menor a los que obtuvo sandoval m 2016 obteniendo un rho 0751 y sig bilateral 0000 p005 lo cual nos muestra que si existe relacion entre el marketing relacional y fidelizacion de los clientes de la micro empresa ev comunicaciones munoz e 2015 marketing relacional y fidelizacion del cliente la diferencia con la siguiente investigacion es que nuestra encuesta va dirigida a los clientes externos de la empresa mientras que la encuesta de munoz va dirigida a los clientes internos y externos asimismo se encontro la relacion entre el marketing relacional y la fidelizacion de clientes pues munoz encontro una relacion x215612 781 mientras nosotros tenemos criterio distinto de medicion mediante rho spearman usando el chicuadrado palate e 2015 determina la relacion entre el marketing relacional y la fidelizacion de clientes pues el x21738 1259 mediante el contexto de coop de ahorro y credito el estudio de merino s 2016 esta orientado a los clientes de la cooperativa de ahorro y credito coop indigena agencia ambato donde las insuficientes estrategias de fidelizacion debido al deficiente servicio al cliente la diferencia con el presente estudio es que merino estudia las dimensiones del marketing relacional desde 3 perspectivas distintas desde el punto de vista cliente y directivos otra diferencia es el metodo para hallar la relacion entre el marketing relacional y la fidelizacion del cliente usando x255792103 en resumen los resultados de correlacion muestran la relacion entre el marketing relacional y la fidelizacion del cliente con los estudios realizados por gutierrez 2015 segun rho spearman rho 0546 sig bilateral 0000 sandoval 2016 determina que existe relacion entre el marketing relacional y la fidelizacion de los clientes rho0751 sig bilateral0000 respecto a la relacion en su dimension satisfaccion y fidelizacion del cliente segun gutierrez m 20169 concluye que no existe relacion entre la satisfaccion y fidelizacion del cliente este resultado segun rho0171 y un nivel de significancia0115 lo cual en contraste con mi investigacion no es semejante ya que existe relacion entre la satisfaccion y la fidelizacion del cliente con rho0453 y un nivel de significancia0002 pero si se fortalece con los resultados de sandoval m 2016 donde determina que si existe relacion entre la satisfaccion y la fidelizacion del cliente rho0656 y sig0000 en seccion se podria afirmar que hay la evidencia necesaria para indicar que existe relacion entre el marketing relacional y la fidelizacion del cliente por otra parte se observa que esto se puede dar en distintos tipos de empresa sin importar el giro del negocio tamano y ubicacion geografica v conclusiones luego de obtener los resultados de la presente investigacion respecto a la relacion del marketing relacional y la fidelizacion de los clientes se llegaron a las siguientes conclusiones teniendo como fundamento los objetivos generales y especificos planteados en esta investigacion 1 se identifico que si existe relacion entre el marketing relacional y la fidelizacion de los clientes con un coeficiente de correlacion moderada rho 0647 y un nivel de significancia de 0000 esto se fundamenta en que el marketing relacional empleado por la empresa jimenez rueda es percibido como bueno por una cantidad considerable de 511 de los clientes y el 511 declara que la fidelizacion de los clientes tambien es buena esta similitud se debe a que el marketing relacional es regular y no estan fidelizando a toda su poblacion conllevando a la empresa a una disminucion de sus ingresos tabla n 1","label":"CONC"},
{"texto":"teoria del desarrollo del ciclo vital esta teoria se ocupa del estudio de la secuencia en la que se produce la transformacion del comportamiento de la persona durante toda su vida baltes 1987 la existencia esta marcada por distintas situaciones en el desarrollo ocasiones notables de cambio por ejemplo entrar a una institucion educativa enamorarse dejar de trabajar etc en cada fase el ser humano confronta los quehaceres que debe vencer para alcanzar un desarrollo excelente erickson 19031994 su principal exponente asevera que en cada momento de nuestra existencia combatimos una disyuntiva psicosocial especifica la firmeza de cada dilema origina una reciente consistencia entre un individuo y la sociedad cuando las personas tienen triunfo en conseguir las tareas encomendadas se produce un progreso beneficioso lo adverso da lugar al manifiesto de dificultades y descontento esta teoria resalta la importancia de conseguir programas de adquisicion durante todo el proceso del desarrollo humano viana y lima 2011 teoria de las inteligencias multiples gardner 1993 habla sobre la compensacion ideal de un intelecto unico gardner planteo que la existencia de los seres humanos exige del desarrollo de diferentes modelos de inteligencia de esta manera de esta manera gardner no discrepa con la idea de que la habilidad de resolver conflictos y generar bienes beneficiosos sea la argumentacion irrefutable de la inteligencia howard gardner y los asistentes ilustres de la universidad de harvard alertaron sobre su capacidad y rendimiento academico para en grados de titulacion y meritocracia educativos esto indica que la inteligencia de un individuo no es un agente decisivo el estudio de howard gardner consiguio relacionar y concretar 8 tipos de inteligencias entre ella la interpersonal el intelecto interpersonal habilita el dominio de exhortar los objetos y cosas de los individuos esto trasciende mas alla de nuestros sentidos la inteligencia nos conlleva a dilucidar las expresiones y sena o fines y logros de cada disertacion gardner sostiene que los individuos adquieren al menos una de las ocho inteligencias en ocasiones sobre sale una mas que las demas al momento de la indagacion de las relaciones interpersonales con los maestros de esta forma se solicita vencer la mayor para contrapones la existencia relativamente esto cambia dependiendo de la ocupacion oficio profesion u otras actividades que desarrollan todos estos trabajos requieren el desarrollo de estas inteligencias teoria de la penetracion social tps la teoria propuesta por altman y taylor 1988 sostiene que en las relaciones se pueden percibir distintos estados de intimidad o familiaridad de interrelacion o nivel de penetracion social este principio se logra entender con otros axiomas que sustentan la teoria en seccion la tps sustenta la existencia de ciertos grados de desarrollo de las relaciones humanas que se transforman principalmente de relaciones intimas a no intimas esta etapa de perfeccionamiento atraviesa un proceso de desarrollo minucioso e imaginable que resultara en la disolucion de dicho vinculo altman y taylor 1973 afirman que la auto revelacion logra que las relaciones no intimas","label":"CONC"},

# ---------- LIM (2) ----------
{"texto":"se consiguio hubo problemas no fue tan faci l despues l legan los ladri l los no se si eran 40 o 60 mil pero de las fundaciones de san luis y de otros lados l legaban ladri l los para ayudar a esta gente y bueno hubo dif icultades bastante grandes de sortear como que los ladri l los se habian yo eso todavia no lo ent iendo que porque se bajo ladri l los eran duenos de los ladri l los creo que fue un error de p haber dicho eso desde buenos aires porque si era asi eramos mas los de este lado que bajamos que lo que les correspondia a el los pasaron un mes o dos de esto la gente del barrio nos empezo a presionar muy fuerte con la entrega de ladri l los no se habia entregado a nadie porque sale que el gobierno iba a dar las viviendas aipo7 entonces lo que se queria era usar los ladri l los como completamiento de las viviendas distr ibuir mejor en funcion de las viviendas que se iban a bajar entonces iba a haber mas casas para todos aparte de eso ya el gobierno al comprar las t ierras reorganizo las t ierras y s i tenian los ladri l los no iban a tener las casas aipo m el 23 de diciembre de 1998 se realiza el acto por el cual el gobernador r hace entrega de los certificados de propiedad a los pobladores hay discursos por parte del gobernador y el ministro de obras publicas s y la celebracion de la palabra por parte de p se habia previsto que hablara uno de los pobladores lo que no sucedio convirtiendose en un acto politico exclusivo del gobierno con el silenciamiento de la comunidad el logro del objetivo de la propiedad de la tierra por parte de sus pobladores fue el punto de inflexion a partir del cual la comunidad comienza a desmovilizarse el lote 133 ya es el barrio 10 de mayo a la hora de evaluar los hechos los protagonistas hacen pesar aspectos diferentes los mismos que los habian separado durante todo el proceso es una mezcla de exito y derrota exito porque estan pero derrota porque no sabes hasta que punto tomaron conciencia estaban tocando el punto neuralgico del s istema que es la propiedad privada eso no se lo podian perdonar objet ivamente era una lucha muy desigual cuando me encuentro con el los t ienen cierta verguenza me dicen cuando va a veni r hablando con el los dicen que han hecho la experiencia mas importante de toda su vida creo que no t ienen conciencia de lo que han hecho en todas las asambleas les decia ustedes se dan cuenta de lo que signif ica para el gobierno el municipio etc lo que estan haciendo que se apropien de 10 12 has de t ierra c","label":"LIM"},
{"texto":"desde la secciondel tac en el asunto se logro demostrar la existencia de razones suficientes para proteger los derechos colectivos entre otras porque desde la coordinacion interinstitucional ya se venian generando alertas sobre la problematica al punto que se habia desplegado un esquema de trabajo donde coincidian las entidades demandas desde el ano 2018 el ministro de agricultura y desarrollo rural y el director general del ica habian solicitado al ministerio de ambiente y desarrollo sostenible y a la autoridad nacional de licencias ambientales aplicar el principio de precaucion y limitar el uso del principio activo fipronil en los cultivos de aguacate citricos pasifloras y cafe por la probada relacion causal entre su uso y los consecuentes episodios de muertes masivas de abejas estimo que a partir de la informacion disponible en el proceso teniendo en cuenta lo manifestado por varios intervinientes lo procedente era ordenar a las autoridades colombianas ajustar el uso de los productos fitosanitarios comprometidos en la demanda teniendo en cuenta las condiciones geograficas del pais que al ser propias del tropico difieren de las europeas situacion que condiciona el comportamiento de los cultivos y de las abejas frente al uso de las moleculas cuestionadas acto seguido ordeno la conformacion de la mesa de trabajo49 sobre la utilizacion de neonicotinoides y fipronil en colombia que hace las veces de comite de verificacion para el cumplimiento de la sentencia tiene como funciones establecer la base cientifica sobre la relacion entre el uso de las moleculas y la muerte masiva de abejas y polinizadores e impulsar las decisiones pertinentes de conformidad con el principio de precaucion la base cientifica debe encontrarse en un primer momento recolectando la informacion disponible es decir haciendo una revision bibliografica en los diferentes medios de divulgacion cientifica locales y globales actividad que implica la centralizacion de la informacion que sobre eventos relacionados con neonicotinoides y fipronil se encuentren en manos de las diferentes autoridades ambientales en el territorio nacional con el fin de definir si con tales evidencias hay suficiente ilustracion para proponer o se hace necesario generar mayor conocimiento para proponer soluciones 49 integrada por los ministerios de agricultura y desarrollo rural y de ambiente y desarrollo sostenible el instituto colombiano agropecuario la autoridad nacional de licencias ambientales la sociedad de agricultores de colombia la fundacion natura colombia y el actor popular abejas y otros insectos polinizadores frente al uso indiscriminado de neonicotinoides y fipronil en colombia luis domingo gomez maldonado derecho animal forum of animal law studies vol 122 215 establecida la suficiente ilustracion sin que implique certeza cientifica absoluta se deben tomar las medidas necesarias para disminuir o eliminar gradualmente el uso de las moleculas en labores agricolas al tiempo que se deberan proponer alternativas para disciplinar su uso en caso de que sea viable en condiciones controladas en caso de no poderse seguir usando se debera trabajar en la identificacion de productos alternativos para controlar los insectos garantizando en todo caso la proteccion de abejas y otros polinizadores una vez se conocio el fallo el ministerio de agricultura y desarrollo rural y la autoridad nacional de licencias ambientales solicitaron su aclaracion exigiendo la exclusion de la molecula fipronil al no pertenecer a la familia de los neonicotinoides la solicitud fue resuelta negativamente mediante decision del 6 de febrero de 2020 el tac recordo que procesalmente la figura de aclaracion de fallo procede para esclarecer puntos o frases que ofrezcan dudas corregir errores aritmeticos o superar una incongruencia entre los extremos de la litis no procedentes de puntos de vista de las partes sino de redacciones ininteligles o del alcance de un concepto que evidencien una discordancia con la parte resolutiva de la decision judicial a juicio del tac la pretension de excluir la molecula fipronil ademas de ser impertinente procesalmente es contraria al deber que tienen las partes de obrar con lealtad y buena fe en cada uno de sus actos razon por la cual no es posible introducir argumentos de fondo cuando en su oportunidad pudieron cuestionar la pretendida familiaridad entre el fipronil y los neonicotinoides","label":"LIM"}

]
},

# ======================================================
# CHAIN OF THOUGHT
# ======================================================
"cot": {

"ejemplos": [

# ---------- CONTR (2) ----------
{"texto":"cita sugerida para esta ponencia gomez geneiro a d 2017 el proceso de investigacion en bibliotecologia y ciencia de la informacion trabajo presentado en v jornadas de intercambio y reflexion acerca de la investigacion en el proceso de investigacion en bibliotecologia y ciencia de la adelaida del carmen gomez geneiro1 informacion resistencia chaco argentina se aborda el proceso de investigacion cientifica en bibliotecologia y ciencia de la informacion bci el enfoque permite analizar las condiciones de realizacion o medios de desarrollo de la investigacion los cursos de accion o metodos aplicados para el descubrimiento y validacion de resultados y el objeto o producto de investigacion que refiere al conocimiento cientifico generado el trabajo marcadorretorico resenar las conceptualizaciones de la metodologia de investigacion cientifica centrada en el proceso de investigacion en bci describir el estado de la cuestion en la produccion disciplinar existente en argentina brasil y mexico en el periodo 20062016 e identificar la relevancia socio institucional para esta comunidad cientifica el analisis inicial de resultados da cuenta sobre desarrollo historico del tema en el campo disciplinar tanto a nivel mundial latinoamericano y regional reconoce la tematica general del estudio y observa escaso tratamiento en comparacion a otras tematicas de interes para sus investigadores e identifica una comunidad de docentes e investigadores en bci que analizan y reflexionan sobre la necesidad de la formacion de los cuerpos academicos y cientificos en metodologia de la investigacion cientifica para contribuir a alcanzar la autonomia de la disciplina concluye que la produccion cientifica individualizada refleja abordajes fragmentados respecto a las dimensiones del proceso de investigacion y que sus resultados estan centrados en reconocer y aplicar estructuras desprovistas de analisis integrados palabras claves proceso de investigacion investigacion cientifica metodologiabibliotecologiaciencia de la informacion america latina 5tas jornadas de intercambio y reflexion acerca de la investigacion en bibliotecologia ensenada 23 y 24 de noviembre de 2017 1 introduccion el trabajo se inscribe en la tematica metodologia de la investigacion cientifica al respecto se asigna a la metodologia el rol de investigar la forma en que las distintas ciencias forman sus conceptos rickert 1961 citado por samaja 2002 y en particular al proceso de investigacion cientifica en bibliotecologia y ciencia de la informacion bci se define el tema a partir del valor de la investigacion cientifica para consolidar una disciplina dado que esta se mide por la actividad cientifica que genera y por la caracterizacion de su perspectiva y naturaleza cientifica en este sentido y siguiendo al epistemologo argentino juan samaja 2004 la mirada desde el proceso de investigacion cientifica permite el analisis de las condiciones de realizacion o medios en los que se desarrolla la investigacion de los cursos de accion o metodos aplicados para el descubrimiento y validacion de los datos obtenidos y el objeto o producto de investigacion que refiere al conocimiento cientifico generado una observacion de caracter holistico que considera al sujeto investigador a su objeto de estudio la naturaleza de su produccion y su contexto de realizacion permite comprender como influyen las dimensiones del proceso de investigacion en la generacion de conocimiento cientifico veraz confiable y fundamentado sobre metodologia de la investigacion cientifica en bci desde este enfoque el trabajo tiene por objetivos resenar las conceptualizaciones de la metodologia de investigacion cientifica centrada en el proceso de investigacion en bci describir el estado de la cuestion en la produccion disciplinar existente en argentina brasil y mexico en el periodo 20062016 e identificar la relevancia socio institucional para esta comunidad cientifica 2 metodologia desde una investigacion de tipo descriptiva cualicuantitativa este trabajo se propone describir la naturaleza del fenomeno ampliar y precisar algunas caracteristicas identificadas a traves del estado de la cuestion en el contexto de argentina brasil y mexico este recorte latinoamericano se funda en los sostenidos desarrollos en investigacion y en el interes observado en la produccion cientifica con el objeto de resenar las conceptualizaciones de la metodologia de la investigacion cientifica centrada en el proceso de investigacion se aplico recuperacion","label":"CONTR"},
{"texto":"otros archivos multimedia como sonido o video pueden ser incluidos tambin en las pginas web como parte de la pgina o mediante hipervnculos juegos y animaciones tambin pueden ser adjuntados a la pgina mediante tecnologas como adobe flash y java este tipo de material depende de la habilidad del navegador para manejarlo y que el usuario permita su visualizacin cdigo del lado del cliente como javascript o ajax pueden incluirse adjuntos al html o por separado ligados con el cdigo especfico en el html este tipo de cdigo necesita correr en la computadora cliente si el usuario lo permite y puede proveer de un alto grado de interactividad entre el usuario y la pgina web las pginas web dinmicas son aquellas que pueden acceder a bases de datos para extraer informacin que pueda ser presentada al visitante dependiendo de ciertos criterios ejemplo de esto son pginas que tienen sistemas de administracin de contenido o cms estos sistemas permiten cambiar el contenido de la pgina web sin tener que utilizar un programa de ftp para subir los cambios existen diversos lenguajes de programacin que permiten agregar dinamismo a una pgina web tal es el caso de asp php jsp y varios ms un navegador web o explorador web puede tener una interfaz de usuario grfica gui graphical user interface como internet explorer netscape navigator mozilla firefox etc o puede tener una interfaz de modo texto como lynx el ms popular es el internet explorer de microsoft los usuarios con navegadores grficos pueden deshabilitar la visualizacin de imgenes y otros contenidos multimedia para ahorrar tiempo ancho de banda o simplemente para simplificar su navegacin tambin se puede descartar la informacin de fuentes tamaos estilos y esquemas de colores de las pginas web y aplicar sus propias css estilizndola a su gusto el consorcio world wide web w3c y la iniciativa de accesibilidad web wai recomiendan que todas las pginas deben ser diseadas tomando en cuenta todas estas consideraciones elementos de una pgina web una pgina web tiene contenido que puede ser visto o escuchado por el usuario final estos elementos incluyen pero no exclusivamente texto el texto editable se muestra en pantalla con alguna de las fuentes que el usuario tiene instaladas imgenes son ficheros enlazados desde el fichero de la pgina propiamente dicho se puede hablar de tres formatos casi exclusivamente gif jpg y png hablamos en detalle de este tema en la seccin de grficos para la web audio generalmente en midi wav y mp3 adobe flash adobe shockwave grficas vectoriales svg scalable vector graphics hipervnculos vnculos y marcadores la pgina web tambin puede traer contenido que es interpretado de forma diferente dependiendo del navegador y generalmente no es mostrado al usuario final estos elementos incluyen pero no scripts generalmente javascript meta tags hojas de estilo css cascading style sheets las pginas web generalmente requieren de ms espacio del que esta disponible en pantalla la mayora de los navegadores mostrarn barras de desplazamiento scrollbars en la ventana que permitan visualizar todo el contenido la barra horizontal es menos comn que la vertical no solo porque las pginas horizontales no se imprimen correctamente tambin acarrea ms inconvenientes para el usuario una pgina web puede ser un solo html o puede estar constituido por varios formando un arreglo de marcos frames se ha demostrado que los marcos causan problemas en la navegacin e impresin sin embargo estos problemas generalmente ocurren en navegadores antiguos su uso principal es permitir que cierto contenido que generalmente est planeado para que sea esttico como una pgina de navegacin o encabezados permanezcan en un sitio definido mientras que el contenido principal puede ser visualizado y desplazado si es necesario otra caracterstica de los marcos es que solo el contenido en el marco principal es actualizado cuando las pginas web son almacenadas en un directorio comn de un servidor web se convierten en un website el website generalmente contiene un grupo de pginas web que estn ligadas entre s la pgina ms importante que hay que almacenar en el servidor es la pgina de ndice index cuando un navegador visita la pgina de inicio homepage de un website o algn url apunta a un directorio en vez de a un archivo especfico el servidor web mostrara la pgina de ndice cuando se crea una pgina web es importante asegurarse que cumple con los estndares del consorcio world wide web w3c para el html css xml etc los estndares aseguran que todos los navegadores mostrarn informacin idntica sin ninguna consideracin especial una pgina propiamente codificada ser accesible para diferentes navegadores ya sean nuevos o antiguos resoluciones as como para usuarios con incapacidades auditivas y visuales","label":"CONTR"},

# ---------- INTRO (2) ----------
{"texto":"para la organizacion mundial de la salud la lactancia materna es la opcion de mayor beneficio y eficacia para asegurar la salud y supervivencia del nino especialmente durante los primeros meses de vida se estima que para el 2020 solo la reciben el de estos menores siendo reemplazada por formulas infantiles en panama existen una gran cantidad de variedades de formulas infantiles proveniente de leche de vaca como pueden ser leche en polvo entera descremada deslactosada procesadas y modificadas el objetivo de este estudio es la car cterizacion de los parametros fisicoquimicos de diferentes tipos de formulas infantiles comercializadas localmente y su correspondiente comparacion con la leche materna para esto se identificaron siete formulas infantiles diferentes todas ellas con el perfil de leche entera cada una de ellas fueron diluidas en agua desionizada segun la descripcion y concentracion detallada por el fabric nte inmediatamente se realizaron las diferentes m diciones del contenido de grasa solidos no grasos solidos totales densidad proteina lactosa ademas de la medicion del ph yla conductividad los resultados obtenidos permiten establecer que las formulas de leche entera que se comercializan actualmente en el pais poseen niveles similares a los de la leche materna segun los parametros estudiados es posible entonces considerar que las diferencias entre la leche materna y las formulas comercializadas difieren principalmente en el aporte biologico activo como lo son la transferencia de inmunidad la capacidad para combatir y la eliminacion de patogenos entre otras palabras clave formula infantil leche en polvo leche materna","label":"INTRO"},
{"texto":"este artfculo reporta los hallazgos de una investigacion que analizo las condiciones que subyacen al conflicto entre grupos profesionales desde una perspectiva inspirada en la teorfa de la identidad social el estudio se centro en la tendencia que exhiben grupos profesionales cercanos o similares a diferenciarse recfprocamente evaluando las consecuencias psicologi cas y sociales de tal tendencia para realizar esta investigacion se disenaron cuestionarios autoadministrados que fueron aplicados a una muestra de psicologos clfnicos y psiquiatras los resultados confirmaron la presencia de conflicto asociado a la coexistencia de ambos grupos profesionales antecedentes nuestra cultura ha desarrollado un conjunto progresivamente mas complejo de profesiones destinadas a atender las necesidades especificas de la poblacion pese a que parte importante de los esfuerzos de racionalizacion que han existido en el el financiamiento de la investigacion en la que se basa este articulo fue otorgado por el fondo nacional de inves tigacion cientifica y tecnologica fondecyt median te el proyecto 0846 psicologo ph d profesor escuela de psicologia uni versidad catolica de chile direccion vicuna mackenna 4860 santiago chile psicologo profesor escuela de psicologia universidad catolica de chile direccion vicuna mackenna 4860 santiago chile desarrollo historico de tales grupos se han dirigido a establecer delimitaciones en las funciones que estos cumpliran existen aun numerosos casos donde dicha delimitacion no es suficientemente clara piensese por ejemplo en las diadas arqui tecto constructor civil psicologo clinico psi quiatra ingeniero civil ingeniero de ejecucion ingeniero de sistemas administrador por nom braralgunos de los casos en que stj generan","label":"INTRO"},

# ---------- BACK (2) ----------
{"texto":"retroalimentacion puede ser especialmente util en la formacion inicial de docentes y entre pares no obstante se enfatiza la necesidad de una formacion especifica para quienes ofrecen la retroalimentacion la retroalimentacion de la practica docente una revision sistematica de la literatura the feedback of teaching practice a systematic review of the literature vol no3 noviembre 2022 issn 1138 414x issne 1989 6395 doi 30827profesorado v26i3 16925 fecha de recepcion 27112020 fecha de aceptacion 06072022 luis horacio pedroza zuniga jihan garcia poyato falcon universidad autonoma de baja california benemerita escuela normal estatal profesor jesus prado luna e mail horacio pedrozauabc edu mx jrgarciabenejpl edu mx orcid id 0000 0002 5256 2967 0000 0002 3692 6687 la retroalimentacion de la practica docente una revision sistematica de la literatura palabras clave retroalimentacion profesion docente evaluacion formativa formacion continua formacion inicial introduccion se reconoce ampliamente que la retroalimentacion es una de las influencias mas poderosas en el aprendizaje de las personas bransford et al 2000 darling hammond et al 2017 hattie y timperley 2007 en meta analisis realizados sobre el efecto de la retroalimentacion en el aprendizaje de los ninos esta jerarquizada entre los factores de mayor influencia solo detras de la ensenanza y la habilidad previa del estudiante hattie y timperley 2007 sin embargo existen algunos tipos de retroalimentacion que pueden tener mayor efecto que otros e incluso la retroalimentacion puede tener un efecto negativo hattie y timperley 2007 wisniewski et al 2020 los sistemas educativos han implementado modelos de evaluacion del profesorado en los cuales se otorga un lugar importante a la retroalimentacion ademas de las evaluaciones con fines sumativos se han tratado de implementar evaluaciones de caracter formativo martinez et al 2016 martinez rizo 2016 la idea subyacente es","label":"BACK"},
{"texto":"resena j ignacio diez ficciones y confesiones francisco umbral y otros escritores contemporaneos por fabienne uni actio nova revista de teoria de la literatura y literatura comparada no3 2019 doi https doi org10 15366actionova2019 actio nova revista de teoria de la literatura y literatura comparada issn 2530 4437 https revistas uam esactionova j ignacio diez ficciones y confesiones francisco umbral y otros escritores contemporaneos prologo de eduardo martinez rico san fernando editorial dalya 2019 paginas isbn 9495032 fabienne uni universite de pau et des pays de ladour ficciones confesiones en su ultimo libro j ignacio diez estudia las relaciones estrechas entre literatura y veracidad relaciones ambiguas para muchos escritores y todavia mas para francisco umbral quien jugo con ambos generos lo largo de sus mas de cien libros la pregunta tambien se la plantea diez a proposito de otros dos escritores como veremos luego j ignacio diez catedratico de literatura espanola en la universidad complutense de madrid dirigio la tesis de eduardo martinez rico la obra narrativa de francisco umbral 1965 2001 quien a su vez prologa este libro destaca de su comentario que el autor es un verdadero aficionado a la prosa de francisco umbral aunque tambien se interesa en otros temas la poesia y prosa del siglo de oro asi como la del siglo xx la literatura erotica y otros autores cervantes o gracian por ejemplo de hecho diez participa de manera muy activa en el resurgimiento del interes academico por el autor y acude a cuanto acto academico o extraacademico dedicado a umbral se celebra segun martinez rico y al leer el libro se confirma el autor piensa y escribe muy bien y aunque este libro no es exclusivamente en torno a francisco umbral sino tambien a otros dos escritores juan benet y javier marias este ensayo de estudios literarios tiene una coherencia inesperada en un libro hecho de varias piezas tan dispares a","label":"BACK"},

# ---------- METHO (2) ----------
{"texto":"presento publicamente su apoyo y apuesta por los seminarios de terapia de conversion gay pues ello contextualiza sobre que trata y donde esta situada mi investigacion el bloque siguiente contiene una revision breve de literatura relacionada con el tema de investigacion mostrando que se ha investigado ya respecto al tema este bloque da algunas luces sobre cuales serian mis aportes al estado del conocimiento este proyecto se respalda en un breve marco teorico en el cual se presentan algunos referentes y disciplinas desde las cuales entable mi acercamiento al tema de estudio michael foucault el conapred judith butler y el informe born free and equals se explica tambien la metodologia que enmarco el proyecto aqui describo el tipo de investigacion que desarrolle asi como las herramie nvestigacion que desarrolle asi como las herramientas y tecnicas utilizadas y comento en que consistio mi caso de estudio los alcances y limitaciones que tuve al abordarlo asi como la forma en que estas moldearon mis posibilidades de analisis es decir como arme el analisis de mi fenomeno y como ordene finalmente la informacion de mi proyecto para dar respuesta a mis preguntas hipotesis y objetivos al contrastar los datos obtenidos en el trabajo de campo con aparatos normativos de derechos humanos desde una reflexion critica retomando a los referentes teoricos utilizados para luego dar paso a las conclusiones y recomendaciones 2 patologia o eleccion hasta poco antes de la decada de 1980 a la homosexualidad se le consideraba un trastorno mental fue gracias a la lucha durante lustro","label":"METHO"},
{"texto":"36 meses de edad atendidos en el centro de salud el obrero durante los meses de febrero y agosto sullana 2017 19 ii metodologia 21tipo y diseno de la investigacion 211 tipo el tipo fue aplicativo analitico ambispectivo aplicativo porque la investigadora hizo uso de teorias y conceptos cientificos establecidos dentro de la literatura cientifica analitico porque el presente estudio cuenta con dos variables las cuales son estudiadas y medida en su contexto natural ambispectiva porque las medidas realizadas a las variables en estudio seran tomadas en primer lugar de documentos ya establecidos y registrado que viene hacer las historias medicas dtos correspondiente a la variable 02 y en segundo lugar se tomaran medidas a los tutores de los ninos lugar de documentos ya establecidos y registrado que viene hacer las historias medicas dtos correspondiente a la variable 02 y en segundo lugar se tomaran medidas a los tutores de los ninos a traves de un cuestionario estructurado por el investigador datos correspondiente a la variable 01 212 diseno el diseno de estudio fue no experimental transversal descriptiva correlacional no experimental porque la investigadora no manipulara la variable solo la observa la mide y la describe transversal porque la investigadora medira a las unidades de estudio tutores e historias clinicas en una sola oportunidad 20 descriptiva porque la investigadora con lo datos obtenidos de las unidades en estudio describira el comportamiento de las mismas en su context e historias clinicas en una sola oportunidad 20 descriptiva porque la investigadora con lo datos obtenidos de las unidades en estudio describira el comportamiento de las mismas en su contexto natural correlacional porque la investigadora establecera la relacion que se da entre las variables estudiada el diseno de investigacion presenta el siguiente figura donde nr viene hacer la muestra o grupo de estudio elegida a traves de un muestreo no randomizado g es el grupo o muestra en estudio ox la observacion y medida realizada la variable 01 es la relacion que existe entre las variables oy la observacion y medida realizada la variable 02 22 poblacion y muestra 221 poblacion la poblacion de estudio fue representada por la totalidad de tutor","label":"METHO"},

# ---------- RES (2) ----------
{"texto":"e tienen conversatorios revision de revistas revision de casos clinicos via zoom la segunda esta enfocada a la participacion en la sede hospitalaria y centros de salud1 el internado medico enfocado en centros de salud y sedes hospitalarias ayuda mucho al estudiante a tener un campo mas amplio de la vida laboral ya que el siguiente paso el serums es similar a la rotacion en centros de salud donde hemos pasado por campanas de salud donde se genera la promocion y prevencion de diversas enfermedades es importante que la sede hospitalaria tenga la gran parte de rotaciones por que esto sera beneficioso para el futuro medico ya que a corto plazo ayudara a superar el enam pero a largo plazo genera un aspecto mas amplio en cuanto al enfoque de patologias es adecuada la proteccion que era beneficioso para el futuro medico ya que a corto plazo ayudara a superar el enam pero a largo plazo genera un aspecto mas amplio en cuanto al enfoque de patologias es adecuada la proteccion que ofrecen al interno de medicina tanto en la sede hospitalaria como en el centro de salud y no solo por las charlas preventivo promocionales y el equipo de bioseguridad que ofrecen las sedes hospitalarias al tener mas tiempo a los internos se preocupan por la alimentacion de este ya que ofrecen desayuno almuerzo y cena al estudiante que lo necesite un punto importante para aquellos que no tienen la condicion economica ni el tiempo de salir a comprar un almuerzo ya que el comedor de los hospitales al ser grande es muy eficiente en cuanto a tiempos y sobre todo ofrecen una dieta balan","label":"RES"},
{"texto":"iarias pueden redundar en una delegacion de responsabilidades encargando a las familiasla atencion de las personas con enfermedades asi la familiarizacion de la atencion de la salud puede ser unfactor en este traslado de dispositivos sanitarios motivo por el cual nos exhorta a una lectura compleja de estasmedidas que intentan un cambio del lugar de cuidado en fin de vida y del lugar de muertela atencion que se realice en el hogar tiene un abanico de posibilidades la medica vilma tripodoro y latrabajadora social elena durbano realizan su aporte en este sentido escribiendo un unico capitulo1 sobre laatencion que realizan en domicilio desde la medicina paliativa desarrollan un analisis del encuadre legal vigente en argentina en relacion a los cuidados paliativos domiciliarios concl ulo1 sobre laatencion que realizan en domicilio desde la medicina paliativa desarrollan un analisis del encuadre legal vigente en argentina en relacion a los cuidados paliativos domiciliarios concluyendo en la falta de una respuestaunificada estructurada y concreta desde el sistema publico sanitario asimismo marcan fuertemente los beneficiosque conlleva la inversion en medicina paliativa domiciliaria tanto para disminuir la utilizacion de camas deagudos como para bienestar de las familias dado que la medicina paliativa no solo se ocupa de la atencion delpaciente sino que se hace extensiva al cuidado de la familia p 114las familias que ejercen los cuidados informales en el hogar despliegan una serie de estrategias para laatencion a la persona enferma la organizacion diaria siva al cuidado de la familia p 114las familias que ejercen los cuidados informales en el hogar despliegan una serie de estrategias para laatencion a la persona enferma la organizacion diaria de estos cuidados se realiza dentro del ambito domestico274r katal florianopolis v 17 n 2 p 272275 juldez 2014ambito donde se funden los dos mundos del enfermo y del cuidador el mundo del enfermo como lo llama laautora es un mundo cargado de vivencias heterogeneas vaivenes en los procesos de saludenfermedademociones y sentires complejos y dolorosos tratamientos terapeuticos paliativos alternativos tradicionalesconvencionales ficticios corporalidad mutable identidad deteriorada contingente hibrida dependiente biografiaresignificada cambios abruptos y constantes","label":"RES"},

# ---------- DISC (2) ----------
{"texto":"ano 2020 volume 1 nmero 1 set dez de 2020 9 puede ser permisiva o coercitiva siendo a la larga perjudicial para la poblacin iek 2009 define esta violencia como la danza metafsica autopropulsada del capital lo que hace funcionar el espectculo lo que proporciona la clave de los procesos y las catstrofes de la vida real esta violencia ya no es atribuible a los individuos concretos y a sus malvadas intenciones sino que es puramente objetiva sistmica annima pp 2223 en colombia el campo de la violentologa logr de forma particular gracias al estudio detallado de la comisin de estudios sobre la violencia 2009 categorizar en contexto o colombianizar esas y otras definiciones de la violencia en el pas resultaron de dicho anlisis la violencia poltica la violencia urbana en colombia en el decenio del ochenta la violencia organizada la violencia contra minoras tnicas en colombia la violencia y medios de comunicacin y finalmente la violencia en la familia la violencia en colombia es particular precisamente al surgir de unas caractersticas propias que podran leerse tanto en contextos micros como en macros a nivel regional y departamental es por ello que no hablamos solo de un tipo de violencia stas coexisten se cruzan o superponen y que segn la mismos violentlogos dejan cada vez menos espacios al optimismo hablamos pues de unas violencias consideradas mltiples y difusas velsquez rueda 2020 de violentlogos a pazologos en la escuela es as que las mltiples violencias y los acontecimientos descritos en relacin al conflicto armado y la participacin estadounidense en las dinmicas nacionales han permitido la construccin de muchos discursos en torno a la paz una paz que tambin puede ser asumida desde diferentes referenciales tericos que ayudan a pensar y actuar sobre sta desde las regiones de nuestro pas marcadorretorico las polticas pblicas que tienen como foco la paz siguen producindose desde bogot irnicamente tenemos violentlogos en el pas pero no existen pazologos o un campo de pazologa la mayora de construcciones al respecto de la paz se realizan a nivel internacional este campo de estudio se conoce como la irenologa la paz desde la constitucin poltica de colombia 1991 en el artculo 22 es definida como derecho y un deber de obligatorio cumplimiento lo que evidencia una","label":"DISC"},
{"texto":"con lo anterior se afirma que el tiempo estndar total para crear un jean de hombre o mujer es de 1379 minutos incluyendo todas las operaciones 1169 min y el total de los suplementos 21 min teniendo en cuenta que se realizaron las repeticiones que arrojo el clculo de la muestra proyectada en la debe recopilar y agrupar la informacin de las ventas de los aos 2012 y 2013 en la tabla 13 se puede observar dicha informacin suministrada por la para mujer y hombre aos 2012 y 2013 enero 2012 1102 994 febrero 2012 1130 1019 marzo 2012 1138 1026 abril 2012 1280 1155 mayo 2012 1083 977 junio 2012 1088 981 julio 2012 1260 1136 agosto 2012 1107 999 septiembre 2012 1176 1061 octubre 2012 1824 1644 noviembre 2012 1842 1661 diciembre 2012 1893 1706 enero 2013 1157 1043 febrero 2013 1105 996 marzo 2013 1052 948 abril 2013 1315 1185 mayo 2013 1157 1043 junio 2013 1079 972 julio 2013 1281 1154 agosto 2013 1125 1014 septiembre 2013 1189 1072 octubre 2013 1840 1658 noviembre 2013 1865 1681 diciembre 2013 1920 1731 para observar claramente el comportamiento de los datos histricos el grfico 6 muestra mes a mes la tendencia y la cantidad de unidades que se han vendido durante los aos 2012 y 2013 grfico 6 datos histricos de las ventas en unidades de jeans marca propia para mujeres y hombres aos 2012 y 2013 como lo muestra el grfico 6 la demanda tiene un comportamiento similar para ambos productos en los dos aos de evaluacin la diferencia est en el nmero de unidades vendidas adicional a esto el grfico muestra como las ventas son superiores en los ltimos tres meses de cada ao esto por la llegada de la temporada navidea esta temporada resulta ser la mejor para el sector textil por lo general para los dems meses no se observa un comportamiento que vare demasiado demanda para los meses del horizonte de planeacin se realiza el anlisis de la demanda pasada por medio de la utilizacin de diferentes tcnicas determinando as la ms apropiada conociendo los volmenes razonables y basndose en el anlisis de las grficas de dispersin se conoce el comportamiento y se escogen las tcnicas de tendencia teniendo en cuenta las tcnicas de tendencia se eligen 3 tipos de pronsticos los cuales son descritos a continuacin promedio mvil este pronstico trabaja con subconjuntos de datos obteniendo de esta forma el promedio y se va moviendo hasta que se obtienen los pronsticos determinados se trabaja por subgrupos de 2 meses dadas las diferencias entre los periodos suavizacin exponencial simple este pronstico trabaja obteniendo promedios de los datos anteriores bajo valores ponderados dando mayor importancia a los datos ms recientes y menor importancia a los datos ms suavizacin exponencial doble es similar a la tcnica de pronstico anterior con la variacin que mitiga los datos por medio de la pendiente de los datos con el objetivo de arrojar resultados ms acertados los indicadores que se evaluarn para cada tipo de pronstico es desviacin absoluta media mad la frmula 19 muestra la forma de hallar este indicador tambin se tiene en cuenta indicadores como el error medio cuadrado mse y media de las desviaciones por periodo bias","label":"DISC"},

# ---------- CONC (2) ----------
{"texto":"de 33 alumnos cuyas edades fluctuan en un rango de 20 a 22 anos el instrumento fue escala de autoconcepto elaborada por valdez 1991 que contiene 37 preguntas con cinco opciones de respuesta y dentro de sus seccion manifiesta que la variable autoconcepto se adquiere y se va desarrollando en las personas segun la influencia del entorno familiar educativo social y puede traer tanto exito como fracaso nunez hernandez jerez y nnez 2018 en su articulo tuvo como meta senalar el grado de progreso de las habilidades sociales y como se relacionan con la educacion para afirmar el objetivo declarado se plantearon las interrogantes cual es el grado de atencion a las unidades biopsicosociales en el ambito academico que factores intervienen en el estudio y rendimiento academico y como se relacionan los estudiantes con sus profesores este trabajo tuvo un enfoque cualitativo y cuantitativo primero porque apoya las variables de estudio que se apoya bibliograficamente para tener una idea clara de su conceptualizacion el segundo permite la aplicacion de instrumentos para obtener datos numericos los cuales seran analizados tabulados y representados estadisticamente como antecedentes nacionales tenemos a isaza 2015 en su articulo habilidades sociales en preadolescentes y su relacion con las practicas educativas el objetivo fue reconocer el enlace del desempeno en habilidades sociales y las practicas educativas familiares con el objetivo de favorecer la adaptacion escolar con un ajuste cuantitativo tipo correlacional y transversal utilizo una muestra de 143 participantes del nivel primaria y sus familias como resultado las habilidades sociales tienen un rol significativo en los procesos de adaptacion escolar por lo cual es importante incentivar de manera directa e indirecta explicita e implicita dentro de las aulas las diversas competencias sociales para los procesos de adaptacion escolar que resulta indispensable para socializar expresar emociones y opiniones ademas de interactuar con la autoridad martinez 2017 cuya investigacion denominada autoconcepto y logro de aprendizaje del area social en estudiantes del nivel primaria cuyo meta fue indicar la correspondencia entre las variables de estudio de caracter cuantitativo el metodo hipotetico deductivo de tipo basica y diseno no experimental transversal y correlacional con una muestra de 103 estudiantes dicho estudio alcanzo la conclusion que el autoconcepto tiene relacion con el logro del aprendizaje con un coeficiente de correlacion de spearman rho 0470 y un valor de significancia p 0000005 perca 2017 en su estudio habilidades sociales y desarrollo de capacidades del area de ciencias en el nivel secundaria independencia nacional puno2014 de la universidad educacion la cantuta cuya meta es precisar la conexion que existe en las variables de estudio de tipo cuantitativo con metodo descriptivo correlacional y de diseno transversal con poblacion compuesta por 100 participantes y una muestra de 60 participantes dicho estudio llego a la conclusion que con respecto a la asertividad y el desarrollo de capacidades del area no existe relacion significativa p005 asi mismo con respecto a la comunicacion y el progreso de capacidades en el area no existe relacion significativa p005 villamares 2017 investigo sobre habilidades sociales y logro de aprendizaje la meta era indicar la conexion entre las variables de corte cuantitativo con metodo hipoteticodeductivo la poblacion compuesta por estudiantes del nivel primaria de tipo basica teorica y de diseno no experimental transversal de alcance correlacional como conclusion manifiesta la presencia de una conexion notoria y significativa entre las variables sujetas a estudio con un coeficiente de similitud de spearman de 0637 y un valor p 0000 menor al nivel de 005 confirmando la relacion entre las variables analizadas iman 2019 en su estudio habilidades sociales en estudiantes de la institucion educativa publica del callao el fundamental aporte es establecer el grado de progreso de la variable de estudio en el alumnado el estudio fue de tipo descriptivo simple y diseno no experimental con una muestra de 120 participantes con edades entre 11 y 12 anos se valio del instrumento test de goldstein 1978 por tanto se demostro que existe un nivel medio el porcentaje obtenido fue 3083 en nivel bajo 4833 en nivel medio y un 2083 en nivel alto como conclusion se presenta un nivel bajo de progreso de habilidades sociales grimaldo 2018 en su investigacion nivel de autoconcepto en estudiantes del nivel primaria de una institucion educativa de lima cuyo objetivo principal era fijar la situacion de autoconcepto en los participantes debido a lo importante del autoconcepto como componente en el desarrollo de los individuos de tipo descriptivo con caracter cuantitativo diseno no experimental y corte transversal con una muestra de 52 participantes utilizo como instrumento el test de autoconcepto de garley 2001 y aprobado en peru por matalinares 2011 de acuerdo con los analisis alcanzados un 63 de participantes muestra un nivel de autoconcepto medio sin embargo en la extension fisica e intelectual correspondiente al autoconcepto se mostro un 62 de nivel bajo","label":"CONC"},
{"texto":"una cuestion ideologica relevante es la que se refiere al valor que se concede al ejercicio fisicodeportivo como medio formativo del caracter especialmente por la trascendencia que la interpretacion del concepto de caracter suele tener en los regimenes autoritarios sin embargo en citius altius fortius no existen apenas referencias directas en ese sentido lo que permite considerarla como una revista ciertamente si no rupturista si independiente con respecto a las ideas establecidas y naturalizadas por el regimen politico es importante destacar como en las paginas de la revista si bien no hay un cuestionamiento de los mecanismos ideologicos y de organizacion politica del deporte y de la educacion fisica vigente al menos tampoco es apreciable un llamamiento claro o una propaganda explicita del uso de la educacion corporal y el deporte para el bien de la nacion del mismo modo pese a la presencia de la iglesia en las instituciones educativas y el control moral que ejerceria en apoyo del estado en las paginas de citius altius fortius no es notoria la presencia estudios con un enfoque religioso es cierto que hay autores declaradamente catolicos entre los que cabe destacar al propio jose maria cagigal sin embargo mas alla de una matizada vision cristiana del cuerpo y de algunas referencias al fomento de los valores cristianos a traves de la practica deportiva la revista supo mantenerse al margen de la vigilancia e inspeccion que el denominado nacionalcatolicismo ejercio en otros ordenes de la cultura del momento en cualquier caso la apertura a la colaboracion de intelectuales de reconocido prestigio de ambitos como la filosofia la historia la sociologia la psicologia etc cuyas miradas si no criticas al menos hechas desde el distanciamiento intelectual supuso un acicate en la aproximacion teorica a la educacion fisica y al deporte nada usuales en las publicaciones de la epoca a este respecto y teniendo en cuenta aqui la importancia tanto cuantitativa como cualitativa de las contribuciones de investigadores extranjeros se podria considerar a citius altius fortius como punto de referencia en el estudio de la cultura fisica a nivel internacional en este sentido si para el regimen franquista toda educacion y en particular la del cuerpo constituia una posibilidad mas de control ideologico al servicio de la dictadura la imagen que citius altius fortius ofrece de la educacion fisica se puede considerar incluso con todas las concesiones terminologicas a la administracion politica que la amparaba epistemologicamente innovadora y pedagogicamente comprometida con una mirada abierta y transformadora la propia concepcion de la educacion corporal y de la cultura fisica de sus directores da buena cuenta de ello","label":"CONC"},

# ---------- LIM (2) ----------
{"texto":"5to congreso internacional de investigacion de la facultad de psicologia isbn 9789503412640 1349 realizada en la cigarra hospital de dia para ninos y adolescentes dependiente del centro de salud mental no1 dr hugo rosarios de buenos aires dicha experiencia incluyo la participacion en el dispositivo de talleres ideados y puestos en practica desde una seccionpsicoanalitica para poder enmarcar tales vinetas comenzaremos presentando algunas particularidades de dicho dispositivo artificio colectivo disenado en el marco de un hospital publico y orientado por una logica que apunta al sujeto en su singularidad entre los diversos talleres que se llevan a cabo en la cigarra nos detendremos en dos de ellos el taller de la palabra y el taller del secreto en los que se enmarcan las vinetas presentadas tales fragmentos clinicos seran trabajados en articulacion con lo presentado anteriormente sobre las perturbaciones del lenguaje y a la luz de las elaboraciones lacanianas sobre la cuestion de la enunciacion tratando de precisar en cada caso la posicion del sujeto con respecto a los enunciados que profiere a partir del analisis de tales fragmentos clinicos indagaremos las particularidades del decir en las psicosis infantiles y el modo singular de cada nino de resolver la cuestion de la enunciacion al mismo tiempo se tratara de precisar la logica de las intervenciones realizadas en el marco de dicho dispositivo asi como sus efectos sobre la posicion del sujeto de esta manera a partir de un movimiento que va de la teoria a la clinica y de la clinica a la teoria se tratara de poner a prueba la nocion lacaniana de enunciacion indagando su pertinencia y sus usos posibles en la clinica con ninos palabras clave enunciacion psicosis nino psicoanalisis","label":"LIM"},
{"texto":"4 metodologia esta investigacion se ha realizado mediante una metodologia cualitativa que estudia el contenido desde la seccionde los sujetos teniendo siempre en cuenta el contexto particular quecedo y castano 2003 mas concretamente se ha llevado a cabo un estudio de caso porque permite estudiar en profundidad una sola situacion para que referida a este trabajo ilustre la transmision y reproduccion de los estereotipos de genero mediante el curriculo oculto leon y montero 2004 todo ello visto desde una perspectiva de genero que implica llevar a cabo un estudio objetivo de las interacciones entre los generos de las instituciones que permiten o castigan las normas y limites del genero y del analisis de las mujeres y los hombres como seres sociales inmersos en un determinado contexto reinoso y hernandez 2011 41 instrumento como tecnica de investigacion se ha utilizado la entrevista esta cuenta con dos ventajas principales permite extraer datos globales y favorece la reflexion sobre acontecimientos pasados asimismo se ha elegido teniendo en cuenta su gran flexibilidad y que potencia un modelo de dialogo entre iguales carbonero y caparros 2015 el lugar escogido para la entrevista en todos los casos fue un lugar tranquilo sin ruidos y en el que solo se encontraba la participante y la entrevistadora esto favorecia la creacion de un clima de confianza y comodidad que permitiese su libre expresion ademas para fomentar un ambiente idoneo previamente se realizo una explicacion sobre el motivo de la investigacion sus objetivos y su contenido y se les expuso que la entrevista tendria un caracter anonimo para que expresasen sus experiencias sin coacciones para poder recoger la informacion proporcionada por las participantes y su posterior transcripcion y analisis todas aceptaron que esta fuese grabada se llevo a cabo con un telefono movil que tiene instalada una aplicacion de grabadora de voz en la realizacion del guion de la entrevista se ha tenido en cuenta que las respuestas debian ser abiertas y pese a que este tuviese un orden secuencial la formulacion de las preguntas se iba alterando dependiendo de las respuestas de las entrevistadas siempre siendo la entrevistadora la que guiase la conversacion para obtener la informacion que se buscaba conseguir por ello se trata de una entrevista semiestructurada en el guion de la entrevista ver anexo 1 pueden observarse cuatro partes diferenciadas conocimientos y realidad del aula espacios y tiempos implicacion de las familias y centro educativo en la primera seccion se realizan preguntas que buscan conocer la opinion de la entrevistada acerca del curriculum oculto y de los estereotipos de genero tradicionales asimismo se formulan preguntas sobre situaciones que se han podido dar en el aula entre losas alumnosas y sobre como ellas han actuado en el segundo aparatado se abordan preguntas sobre las zonas de la clase mesa de trabajo asamblea y rincones de juego asi como del momento de recreo en la implicacion de las familias se busca conocer si son los padres las madres o ambos quienes mas se implican en la educacion de sus hijas e hijos y el por que finalmente en el apartado del centro se formulan preguntas que permitan averiguar las actuaciones que se llevan a cabo en la escuela para impedir la reproduccion de estereotipos de genero tradicionales y para conocer la formacion de las entrevistadas en materia de genero cabe resaltar que todas las preguntas que se han formulado son de elaboracion propia basadas en una previa busqueda bibliografica que ha permitido identificar los aspectos mas relevantes a tratar en la investigacion en una segunda fase se procedio a la transcripcion de las entrevistas ver anexo 2 para ello se establecio un codigo para cada una de ellas identificandolas por su profesion la edad y los anos de experiencia docente ver anexo 3 42 muestra en la investigacion realizada han participado un total de 7 docentes todas ellas mujeres que ejercen en la etapa de segundo ciclo de educacion infantil 36 anos en un centro escolar ubicado en un medio rural con una poblacion de 2949 habitantes censados en el ano 2017 ine 2018 este tipo de muestreo es opinatico ya que es ella investigadora quien selecciona a losas participantes mediante criterios estrategicos personales en el caso de este estudio estos criterios son el de proximidad y accesibilidad por este motivo se trata de una muestra con una representatividad estructural muy baja pero que sirve de estudio exploratorio de la situacion actual o para continuar con la investigacion","label":"LIM"}

]
},

# ======================================================
# ZERO SHOT
# ======================================================
"zero-shot": {


"ejemplos": []
}

}

# %% [markdown] cell 11
# ## Celda 5: Funciones auxiliares
#
# Estas funciones leen el CSV con codificaciones comunes, preparan las particiones, construyen prompts multiclase, parsean respuestas JSON, calculan metricas incluyendo ROC AUC, guardan matrices de confusion y generan analisis de errores.

# %% [code] cell 12

def read_csv_flexible(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def download_google_drive_file(file_id: str, destination: Path) -> Path:
    import http.cookiejar
    import urllib.parse
    import urllib.request

    destination.parent.mkdir(parents=True, exist_ok=True)
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    base_url = "https://drive.google.com/uc?export=download"

    def request_url(confirm_token: str | None = None) -> str:
        params = {"id": file_id}
        if confirm_token:
            params["confirm"] = confirm_token
        return base_url + "&" + urllib.parse.urlencode(params)

    response = opener.open(request_url(), timeout=120)
    data = response.read()
    content_type = response.headers.get("Content-Type", "").lower()

    confirm_token = None
    for cookie in cookie_jar:
        if cookie.name.startswith("download_warning"):
            confirm_token = cookie.value
            break

    if confirm_token or "text/html" in content_type:
        html_preview = data[:20000].decode("utf-8", errors="ignore")
        match = re.search(r"confirm=([0-9A-Za-z_\-]+)", html_preview)
        if match:
            confirm_token = match.group(1)
        if confirm_token:
            response = opener.open(request_url(confirm_token), timeout=120)
            data = response.read()
            content_type = response.headers.get("Content-Type", "").lower()

    if "text/html" in content_type or data.lstrip().startswith(b"<!DOCTYPE html") or data.lstrip().startswith(b"<html"):
        raise RuntimeError(
            "Google Drive devolvio HTML en lugar del CSV. Verifica que el enlace este compartido como publico o accesible."
        )

    destination.write_bytes(data)
    if destination.stat().st_size == 0:
        raise RuntimeError(f"La descarga quedo vacia: {destination}")
    return destination


def find_dataset_path() -> Path:
    candidates = [
        DATASET_PATH,
        Path("/home/jovyan/data") / DATASET_FILENAME,
        Path("/workspace/qwen_task1/data") / DATASET_FILENAME,
        Path("/content") / DATASET_FILENAME,
        Path("/content/drive/MyDrive/ProyectoMAIA/data") / DATASET_FILENAME,
    ]
    for path in candidates:
        if path.exists():
            return path

    print("Dataset no encontrado localmente. Descargando desde Google Drive...")
    downloaded_path = download_google_drive_file(DATASET_GOOGLE_DRIVE_FILE_ID, DATASET_PATH)
    print("Dataset descargado en:", downloaded_path)
    return downloaded_path


def normalize_label(value: object) -> str:
    return str(value).strip().upper()


def prepare_dataframe(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    required_columns = {TEXT_COLUMN, LABEL_COLUMN, SPLIT_COLUMN}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    out = df.copy()
    out[SPLIT_COLUMN] = out[SPLIT_COLUMN].astype(str).str.strip().str.upper()
    out = out[out[SPLIT_COLUMN] == split_name.upper()].copy()

    out[TEXT_COLUMN] = out[TEXT_COLUMN].fillna("").astype(str).str.strip()
    out = out[out[TEXT_COLUMN] != ""].copy()

    out[LABEL_COLUMN] = out[LABEL_COLUMN].map(normalize_label)
    out = out[out[LABEL_COLUMN].isin(LABEL_NAMES)].copy()
    out["label_id"] = out[LABEL_COLUMN].map(label2id).astype(int)
    out["word_count"] = out[TEXT_COLUMN].str.split().map(len)
    out["char_count"] = out[TEXT_COLUMN].str.len()
    return out.reset_index(drop=True)


def truncate_text(text: str, max_chars: int = MAX_INPUT_CHARS) -> str:
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " [...]"



COMMERCIAL_PROMPT_CLASSES = ["CONTR", "INTRO", "BACK", "METHO", "LIM", "RES", "CONC", "DISC"]
COMMERCIAL_TO_DATASET_LABEL = {"METHO": "METH", "METH": "METH"}
COMMERCIAL_TO_DATASET_LABEL.update({label: label for label in LABEL_NAMES})


def construir_ejemplos(strategy: str = "cot") -> str:
    ejemplos = ""
    lista = PROMPT_EXAMPLES[strategy]["ejemplos"]

    for ej in lista:
        ejemplos += (
            f"Texto: {ej['texto']}\n"
            f"Respuesta: {ej['label']}\n\n"
        )

    return ejemplos


def build_system_prompt() -> str:
    return "Eres un clasificador cientifico."


def build_commercial_prompt(texto: str, strategy: str = "cot") -> str:
    ejemplos = construir_ejemplos(strategy)

    return f"""
Eres un clasificador experto de textos cientificos.

Debes clasificar el texto en UNA de estas 8 clases:

CONTR = contradiccion o resultados negativos
INTRO = introduccion del tema
BACK = antecedentes o marco teorico
METHO = metodologia
LIM = limitaciones
RES = resultados
CONC = conclusiones
DISC = discusion

Ejemplos:

{ejemplos}

Texto:
{texto}

Responde SOLO en JSON con probabilidades
que sumen 1.

Formato EXACTO:

{{
"CONTR":0.xx,
"INTRO":0.xx,
"BACK":0.xx,
"METHO":0.xx,
"LIM":0.xx,
"RES":0.xx,
"CONC":0.xx,
"DISC":0.xx
}}
"""


def build_few_shot_context(train_df: pd.DataFrame, examples_per_label: int = 1) -> str:
    return construir_ejemplos("cot")


def build_user_prompt(row: pd.Series, strategy: str, few_shot_context: str = "") -> str:
    text = str(row[TEXT_COLUMN]).strip()
    return build_commercial_prompt(text, strategy=strategy)


def messages_for_row(row: pd.Series, strategy: str, few_shot_context: str = "") -> list[dict]:
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(row, strategy, few_shot_context)},
    ]


def parse_model_response(raw_response: str) -> dict:
    raw_response = str(raw_response).strip()
    parsed = None

    json_candidates = re.findall(r"\{.*?\}", raw_response, flags=re.DOTALL)
    for candidate in json_candidates:
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue

    if parsed is None:
        label_match = re.search(
            r"\b(" + "|".join(COMMERCIAL_PROMPT_CLASSES + LABEL_NAMES) + r")\b",
            raw_response.upper(),
        )
        prompt_label = label_match.group(1) if label_match else "UNKNOWN"
        pred_label = COMMERCIAL_TO_DATASET_LABEL.get(prompt_label, prompt_label)
        return {
            "pred_label": pred_label,
            "pred_label_id": label2id.get(pred_label, -1),
            "confidence": 0.0,
            "reason": "No se pudo parsear JSON de probabilidades; se uso extraccion por regex.",
            "raw_response": raw_response,
            "parse_ok": False,
        }

    probabilities = {}
    for prompt_label in COMMERCIAL_PROMPT_CLASSES:
        value = pd.to_numeric(parsed.get(prompt_label, 0.0), errors="coerce")
        probabilities[prompt_label] = 0.0 if pd.isna(value) else float(max(value, 0.0))

    total = sum(probabilities.values())
    if total > 0:
        normalized = {label: value / total for label, value in probabilities.items()}
    else:
        normalized = {label: 1.0 / len(COMMERCIAL_PROMPT_CLASSES) for label in COMMERCIAL_PROMPT_CLASSES}

    prompt_label = max(normalized, key=normalized.get)
    pred_label = COMMERCIAL_TO_DATASET_LABEL.get(prompt_label, prompt_label)
    confidence = float(normalized[prompt_label])
    parse_ok = pred_label in LABEL_NAMES

    return {
        "pred_label": pred_label,
        "pred_label_id": label2id.get(pred_label, -1),
        "confidence": confidence,
        "reason": "Prediccion por maxima probabilidad del JSON comercial.",
        "raw_response": raw_response,
        "parse_ok": bool(parse_ok),
        "probabilities_json": json.dumps(normalized, ensure_ascii=False),
    }


def probabilities_json_to_scores(
    probabilities_json: object,
    pred_label: str | None = None,
    confidence: float | None = None,
) -> list[float]:
    scores = {label: 0.0 for label in LABEL_NAMES}

    try:
        probabilities = json.loads(probabilities_json) if isinstance(probabilities_json, str) and probabilities_json else {}
    except json.JSONDecodeError:
        probabilities = {}

    for raw_label, raw_value in probabilities.items():
        prompt_label = str(raw_label).strip().upper()
        dataset_label = COMMERCIAL_TO_DATASET_LABEL.get(prompt_label, prompt_label)
        if dataset_label not in scores:
            continue
        value = pd.to_numeric(raw_value, errors="coerce")
        if not pd.isna(value):
            scores[dataset_label] += float(max(value, 0.0))

    total = sum(scores.values())
    if total <= 0 and pred_label in scores:
        fallback_confidence = pd.to_numeric(confidence, errors="coerce")
        scores[pred_label] = 1.0 if pd.isna(fallback_confidence) else float(max(fallback_confidence, 0.0))
        total = sum(scores.values())

    if total > 0:
        scores = {label: value / total for label, value in scores.items()}
    else:
        scores = {label: 1.0 / len(LABEL_NAMES) for label in LABEL_NAMES}

    return [float(scores[label]) for label in LABEL_NAMES]


def compute_roc_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    labels = list(range(len(LABEL_NAMES)))
    y_score = np.asarray(y_score, dtype=float)
    y_true_bin = label_binarize(y_true, classes=labels)

    per_label = {}
    supports = []
    auc_values = []
    weighted_terms = []

    for idx, label_name in enumerate(LABEL_NAMES):
        support = int(y_true_bin[:, idx].sum())
        supports.append(support)
        if len(np.unique(y_true_bin[:, idx])) < 2:
            per_label[label_name] = None
            continue

        auc_value = float(roc_auc_score(y_true_bin[:, idx], y_score[:, idx]))
        per_label[label_name] = auc_value
        auc_values.append(auc_value)
        weighted_terms.append(auc_value * support)

    total_support = int(sum(supports))
    macro_auc = float(np.mean(auc_values)) if auc_values else None
    weighted_auc = float(sum(weighted_terms) / total_support) if total_support and weighted_terms else None

    return {
        "roc_auc_macro_ovr": macro_auc,
        "roc_auc_weighted_ovr": weighted_auc,
        "roc_auc_per_label_ovr": per_label,
        "roc_auc_valid_labels": [label for label, value in per_label.items() if value is not None],
    }


def compute_metrics_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> tuple[dict, pd.DataFrame, dict, dict]:
    labels = list(range(len(LABEL_NAMES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    label_metrics_df = pd.DataFrame(
        {
            "label": LABEL_NAMES,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
    }

    roc_metrics = {
        "roc_auc_macro_ovr": None,
        "roc_auc_weighted_ovr": None,
        "roc_auc_per_label_ovr": {label: None for label in LABEL_NAMES},
        "roc_auc_valid_labels": [],
    }
    if y_score is not None and len(y_true):
        roc_metrics = compute_roc_metrics(y_true, y_score)
        summary["roc_auc_macro_ovr"] = roc_metrics["roc_auc_macro_ovr"]
        summary["roc_auc_weighted_ovr"] = roc_metrics["roc_auc_weighted_ovr"]
        label_metrics_df["roc_auc_ovr"] = label_metrics_df["label"].map(roc_metrics["roc_auc_per_label_ovr"])

    return summary, label_metrics_df, report, roc_metrics


def save_confusion_matrix(title: str, pred_df: pd.DataFrame, output_path: Path) -> None:
    matrix_labels = list(range(len(LABEL_NAMES))) + [-1]
    matrix_names = LABEL_NAMES + ["UNKNOWN"]
    cm = confusion_matrix(
        pred_df["label_id"],
        pred_df["pred_label_id"],
        labels=matrix_labels,
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=matrix_names,
        yticklabels=matrix_names,
    )
    plt.title(title)
    plt.xlabel("Etiqueta predicha")
    plt.ylabel("Etiqueta real")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def build_error_analysis(split_name: str, strategy: str, pred_df: pd.DataFrame) -> dict:
    errors = pred_df[pred_df["correct"] == False].copy()
    pair_counts = (
        errors.groupby([LABEL_COLUMN, "pred_label"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(30)
    )
    by_true_label = (
        errors.groupby(LABEL_COLUMN, dropna=False)
        .size()
        .reset_index(name="errors")
        .sort_values("errors", ascending=False)
    )
    by_word_count = errors.assign(
        word_count_bin=pd.cut(errors["word_count"], bins=[0, 30, 60, 120, 240, np.inf])
    ).groupby("word_count_bin", dropna=False).size().reset_index(name="errors")

    errors_file = OUTPUT_DIR / f"{RUN_NAME}_{strategy}_{split_name}_errors.csv"
    errors.to_csv(errors_file, index=False, encoding="utf-8-sig")

    return {
        "split": split_name,
        "strategy": strategy,
        "rows": int(len(pred_df)),
        "errors": int(len(errors)),
        "error_rate": float(len(errors) / max(len(pred_df), 1)),
        "errors_file": str(errors_file),
        "top_confusion_pairs": pair_counts.to_dict(orient="records"),
        "errors_by_true_label": by_true_label.to_dict(orient="records"),
        "errors_by_word_count": by_word_count.astype({"word_count_bin": str}).to_dict(orient="records"),
    }

# %% [markdown] cell 13
# ## Celda 6: Carga del consolidado
#
# Se carga `Dataset_consolidado_final_v4.csv` y se respetan las particiones definidas en `dataset_type`. La particion `TRAIN` se usa como referencia para ejemplos few-shot, mientras que `EVAL` se conserva para medicion final.

# %% [code] cell 14

dataset_path = find_dataset_path()
raw_df = read_csv_flexible(dataset_path)

train_ref_df = prepare_dataframe(raw_df, "TRAIN")
test_df = prepare_dataframe(raw_df, "TEST")
eval_df = prepare_dataframe(raw_df, "EVAL")

if MAX_TRAIN_REFERENCE_ROWS is not None:
    train_ref_df = train_ref_df.sample(
        n=min(MAX_TRAIN_REFERENCE_ROWS, len(train_ref_df)),
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)
if MAX_TEST_ROWS is not None:
    test_df = test_df.sample(n=min(MAX_TEST_ROWS, len(test_df)), random_state=RANDOM_STATE).reset_index(drop=True)
if MAX_EVAL_ROWS is not None:
    eval_df = eval_df.sample(n=min(MAX_EVAL_ROWS, len(eval_df)), random_state=RANDOM_STATE).reset_index(drop=True)

split_dfs = {"test": test_df, "eval": eval_df}

print("Dataset usado:", dataset_path)
print("raw_df =", raw_df.shape)
print("train_ref_df =", train_ref_df.shape)
print("test_df =", test_df.shape)
print("eval_df =", eval_df.shape)

display(train_ref_df.head(2))
display(eval_df.head(2))

# %% [markdown] cell 15
# ## Celda 7: Auditoria de distribuciones
#
# Se revisan conteos por particion y etiqueta para documentar el balance de clases antes de ejecutar inferencia. Esta auditoria tambien ayuda a detectar etiquetas fuera del catalogo esperado.

# %% [code] cell 16

split_summary = []
for split_name, df in [("TRAIN_REF", train_ref_df), ("TEST", test_df), ("EVAL", eval_df)]:
    counts = df[LABEL_COLUMN].value_counts().reindex(LABEL_NAMES, fill_value=0)
    row = {"split": split_name, "rows": int(len(df))}
    row.update({label: int(counts.loc[label]) for label in LABEL_NAMES})
    split_summary.append(row)

split_summary_df = pd.DataFrame(split_summary)
display(split_summary_df)

print("Distribucion de etiquetas por split:")
display(
    pd.concat([
        train_ref_df.assign(split="TRAIN_REF"),
        test_df.assign(split="TEST"),
        eval_df.assign(split="EVAL"),
    ]).pivot_table(
        index="split",
        columns=LABEL_COLUMN,
        values=TEXT_COLUMN,
        aggfunc="count",
        fill_value=0,
    ).reindex(columns=LABEL_NAMES, fill_value=0)
)

unexpected_labels = sorted(set(raw_df[LABEL_COLUMN].dropna().map(normalize_label)) - set(LABEL_NAMES))
print("Etiquetas fuera del catalogo esperado:", unexpected_labels)

# %% [markdown] cell 17
# ## Celda 8: Tokenizador y modelo Qwen3-8B
#
# Se carga `Qwen/Qwen3-8B` para generacion causal. En GPU, la cuantizacion de 4 bits reduce el consumo de memoria; en CPU o entornos sin `bitsandbytes`, el notebook intenta cargar el modelo sin cuantizacion.

# %% [code] cell 18

set_seed(RANDOM_STATE)

#login("hf_...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

quantization_config = None
if USE_4BIT and torch.cuda.is_available():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
    quantization_config=quantization_config,
    trust_remote_code=True,
)
model.eval()

generation_kwargs = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": DO_SAMPLE,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
}
if DO_SAMPLE:
    generation_kwargs["temperature"] = TEMPERATURE

generation_config = GenerationConfig(**generation_kwargs)

print(model.__class__.__name__)
print("Pad token:", tokenizer.pad_token, tokenizer.pad_token_id)
print("Quantization:", quantization_config)

# %% [markdown] cell 19
# ## Celda 9: Construccion de prompts
#
# Se guardan las plantillas `zero_shot` y `few_shot`. El prompt obliga al modelo a elegir una unica etiqueta del catalogo y a devolver JSON parseable para evaluar de forma reproducible.

# %% [code] cell 20
commercial_examples_context = construir_ejemplos("cot")

prompt_templates = {
    "system_prompt": build_system_prompt(),
    "commercial_strategy": "cot",
    "commercial_examples_context": commercial_examples_context,
    "commercial_example_prompt": build_user_prompt(eval_df.iloc[0], "cot") if len(eval_df) else "",
}

(PROMPT_DIR / f"{RUN_NAME}_prompt_templates.json").write_text(
    json.dumps(prompt_templates, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("Plantillas guardadas en:", PROMPT_DIR / f"{RUN_NAME}_prompt_templates.json")
print("\n--- COMMERCIAL PROMPT PREVIEW ---")
print(prompt_templates["commercial_example_prompt"][:2200])

# %% [markdown] cell 21
# ## Celda 10: Inferencia con Qwen3-8B
#
# Esta celda ejecuta la clasificacion de las particiones configuradas en `INFERENCE_SPLITS`. La salida del modelo se parsea como JSON; cuando el JSON falla, se usa una recuperacion conservadora por expresion regular y se marca `parse_ok=False`.

# %% [code] cell 22
def generate_response(row: pd.Series, strategy: str, return_usage: bool = False) -> str | tuple[str, dict]:
    messages = messages_for_row(row, strategy)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    input_tokens = int(inputs["input_ids"].shape[-1])

    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    output_tokens = int(generated_ids.shape[-1])
    raw_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if return_usage:
        return raw_response, {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
    return raw_response


def predict_dataframe(split_name: str, strategy: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    start_time = time.time()
    records = []

    for position, (_, row) in enumerate(df.iterrows(), start=1):
        raw_response, token_usage = generate_response(row, strategy, return_usage=True)
        parsed = parse_model_response(raw_response)
        parsed.update(token_usage)
        records.append(parsed)
        if position % 25 == 0 or position == len(df):
            print(f"{strategy}/{split_name}: {position}/{len(df)}")

    pred_df = df.copy().reset_index(drop=True)
    parsed_df = pd.DataFrame(records)
    pred_df = pd.concat([pred_df, parsed_df], axis=1)
    pred_df["correct"] = pred_df["label_id"] == pred_df["pred_label_id"]
    pred_df["strategy"] = strategy
    pred_df["split_eval"] = split_name

    score_rows = pred_df.apply(
        lambda row: probabilities_json_to_scores(
            row.get("probabilities_json"),
            row.get("pred_label"),
            row.get("confidence"),
        ),
        axis=1,
    )
    y_score = np.vstack(score_rows.to_numpy()) if len(pred_df) else np.empty((0, len(LABEL_NAMES)))
    score_df = pd.DataFrame(y_score, columns=[f"score_{label}" for label in LABEL_NAMES])
    pred_df = pd.concat([pred_df, score_df], axis=1)

    summary, label_metrics_df, report, roc_metrics = compute_metrics_from_arrays(
        pred_df["label_id"].to_numpy(),
        pred_df["pred_label_id"].to_numpy(),
        y_score,
    )
    elapsed = time.time() - start_time
    summary.update({
        "split": split_name,
        "strategy": strategy,
        "rows": int(len(pred_df)),
        "parse_ok_rate": float(pred_df["parse_ok"].mean()) if len(pred_df) else 0.0,
        "seconds": float(elapsed),
        "seconds_per_row": float(elapsed / max(len(pred_df), 1)),
        "input_tokens": int(pred_df["input_tokens"].sum()) if len(pred_df) else 0,
        "output_tokens": int(pred_df["output_tokens"].sum()) if len(pred_df) else 0,
        "total_tokens": int(pred_df["total_tokens"].sum()) if len(pred_df) else 0,
        "tokens_per_row": float(pred_df["total_tokens"].sum() / max(len(pred_df), 1)) if len(pred_df) else 0.0,
    })

    metrics_payload = {
        "summary": summary,
        "classification_report": report,
        "roc": roc_metrics,
        "confusion_matrix": confusion_matrix(
            pred_df["label_id"],
            pred_df["pred_label_id"],
            labels=list(range(len(LABEL_NAMES))) + [-1],
        ).tolist(),
        "confusion_matrix_labels": LABEL_NAMES + ["UNKNOWN"],
    }
    return pred_df, metrics_payload, label_metrics_df

# %% [markdown] cell 23
# ## Celda 10B: Inferencia de un solo fragmento
#
# Funcion auxiliar para probar o desplegar el flujo con Qwen sobre un fragmento individual usando exactamente el prompt comercial integrado.

# %% [code] cell 24
def predict_single_fragment(
    text: str,
    strategy: str = "cot",
    return_prompt: bool = False,
    return_messages: bool = False,
) -> dict:
    """Clasifica un unico fragmento con Qwen usando el prompt comercial.

    Requiere que tokenizer, model y generation_config ya esten cargados.
    Esta funcion es util para pruebas manuales y para reutilizar el flujo en un
    despliegue en linea.
    """
    row = pd.Series({TEXT_COLUMN: str(text).strip()})
    messages = messages_for_row(row, strategy)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    input_tokens = int(inputs["input_ids"].shape[-1])
    with torch.no_grad():
        outputs = model.generate(**inputs, generation_config=generation_config)

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    output_tokens = int(generated_ids.shape[-1])
    raw_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    parsed = parse_model_response(raw_response)

    result = {
        "input_text": str(text).strip(),
        "strategy": strategy,
        "pred_label": parsed["pred_label"],
        "pred_label_id": parsed["pred_label_id"],
        "confidence": parsed["confidence"],
        "parse_ok": parsed["parse_ok"],
        "probabilities_json": parsed.get("probabilities_json"),
        "raw_response": raw_response,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }

    if return_prompt:
        result["prompt"] = prompt
    if return_messages:
        result["messages"] = messages

    return result


# Ejemplo de uso despues de cargar el modelo:
# resultado = predict_single_fragment(
#     "los resultados muestran una mejora significativa frente al metodo base",
#     return_prompt=False,
# )
# display(resultado)

# %% [markdown] cell 25
# ## Celda 11: Guardado de configuracion del experimento
#
# Se guarda la configuracion antes de ejecutar predicciones para dejar trazabilidad del modelo, dataset, etiquetas, prompts, parametros de generacion y decisiones metodologicas.

# %% [code] cell 26

tokenizer.save_pretrained(str(FINAL_MODEL_DIR))
generation_config.save_pretrained(str(FINAL_MODEL_DIR))

if SAVE_FULL_MODEL:
    model.save_pretrained(str(FINAL_MODEL_DIR), safe_serialization=True)

run_config = {
    "task": "task1_rhetorical_label_multiclass_single_label",
    "approach": "open_weight_llm_prompting_inference",
    "dataset_path_used": str(dataset_path),
    "model_name": MODEL_NAME,
    "run_name": RUN_NAME,
    "labels": id2label,
    "text_column": TEXT_COLUMN,
    "label_column": LABEL_COLUMN,
    "split_column": SPLIT_COLUMN,
    "prompt_strategies": PROMPT_STRATEGIES,
    "commercial_prompt_source": "C:\\Users\\Mate\\Downloads\\Deepseek_Comercial_Model_T1_II_VF.ipynb",
    "commercial_prompt_strategy": "cot",
    "commercial_prompt_output_format": "JSON de probabilidades por clase",
    "inference_splits": INFERENCE_SPLITS,
    "max_input_chars": MAX_INPUT_CHARS,
    "max_new_tokens": MAX_NEW_TOKENS,
    "temperature": TEMPERATURE,
    "do_sample": DO_SAMPLE,
    "use_4bit": USE_4BIT,
    "save_full_model": SAVE_FULL_MODEL,
    "random_state": RANDOM_STATE,
    "train_reference_rows": int(len(train_ref_df)),
    "test_rows": int(len(test_df)),
    "eval_rows": int(len(eval_df)),
    "definition_alignment_notes": [
        "La tarea es multiclass single-label: cada fragmento recibe exactamente una etiqueta.",
        "La columna dataset_type define TRAIN, TEST y EVAL.",
        "Las metricas principales del experimento se calculan sobre EVAL.",
        "El modelo se evalua por prompting/inferencia; no se hace fine-tuning en este notebook.",
        "El prompt replica el usado en el notebook comercial DeepSeek con strategy=\"cot\".",
        "La etiqueta METHO del prompt comercial se mapea a METH para comparar con el dataset.",
    ],
}
RUN_CONFIG_FILE.write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
joblib.dump(run_config, OUTPUT_DIR / f"{RUN_NAME}_run_config.joblib")

print("Artefactos del modelo/tokenizador guardados en:", FINAL_MODEL_DIR)
print("Configuracion guardada en:", RUN_CONFIG_FILE)

# %% [markdown] cell 27
# ## Celda 12: Predicciones y metricas
#
# Se generan predicciones, metricas globales, metricas por etiqueta y matrices de confusion. La configuracion por defecto ejecuta `EVAL`, que es el conjunto solicitado para la evaluacion final.

# %% [code] cell 28

all_metrics = {}
all_label_metrics = []
all_prediction_files = []

for strategy in PROMPT_STRATEGIES:
    for split_name in INFERENCE_SPLITS:
        split_df = split_dfs[split_name]
        pred_df, metrics_payload, label_metrics_df = predict_dataframe(split_name, strategy, split_df)

        prefix = f"{RUN_NAME}_{strategy}_{split_name}"
        predictions_file = OUTPUT_DIR / f"{prefix}_predictions.csv"
        split_metrics_file = OUTPUT_DIR / f"{prefix}_metrics.json"
        label_metrics_file = OUTPUT_DIR / f"{prefix}_label_metrics.csv"
        confusion_png = OUTPUT_DIR / f"{prefix}_confusion_matrix.png"

        pred_df.to_csv(predictions_file, index=False, encoding="utf-8-sig")
        split_metrics_file.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        label_metrics_df.to_csv(label_metrics_file, index=False, encoding="utf-8-sig")
        save_confusion_matrix(f"{strategy} - {split_name}", pred_df, confusion_png)

        key = f"{strategy}_{split_name}"
        all_metrics[key] = metrics_payload
        label_metrics_df = label_metrics_df.copy()
        label_metrics_df.insert(0, "strategy", strategy)
        label_metrics_df.insert(1, "split", split_name)
        all_label_metrics.append(label_metrics_df)
        all_prediction_files.append(str(predictions_file))

        print(f"Predicciones {key}:", predictions_file)
        print(f"Metricas {key}:", split_metrics_file)
        display(pd.DataFrame([metrics_payload["summary"]]))
        display(label_metrics_df)

METRICS_FILE.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

summary_metrics_df = pd.DataFrame([
    payload["summary"] for payload in all_metrics.values()
])
summary_metrics_df.to_csv(SUMMARY_METRICS_CSV, index=False, encoding="utf-8-sig")

combined_label_metrics_df = pd.concat(all_label_metrics, ignore_index=True)
combined_label_metrics_df.to_csv(OUTPUT_DIR / f"{RUN_NAME}_all_label_metrics.csv", index=False, encoding="utf-8-sig")

print("Metricas consolidadas guardadas en:", METRICS_FILE)
display(summary_metrics_df)

# %% [markdown] cell 29
# ## Celda 13: Analisis de errores
#
# Se guardan los casos incorrectos y se resumen las confusiones mas frecuentes entre etiquetas. Esto permite revisar donde el modelo confunde funciones retoricas cercanas, por ejemplo `BACK` vs. `INTRO` o `DISC` vs. `CONC`.

# %% [code] cell 30

error_summaries = []

for strategy in PROMPT_STRATEGIES:
    for split_name in INFERENCE_SPLITS:
        pred_file = OUTPUT_DIR / f"{RUN_NAME}_{strategy}_{split_name}_predictions.csv"
        pred_df = pd.read_csv(pred_file)
        error_summary = build_error_analysis(split_name, strategy, pred_df)
        error_summaries.append(error_summary)

ERROR_ANALYSIS_FILE = OUTPUT_DIR / f"{RUN_NAME}_error_analysis_summary.json"
ERROR_ANALYSIS_FILE.write_text(json.dumps(error_summaries, indent=2, ensure_ascii=False), encoding="utf-8")

print("Resumen de analisis de errores:", ERROR_ANALYSIS_FILE)
for item in error_summaries:
    print("\nSplit:", item["split"], "| Strategy:", item["strategy"])
    print("Errores:", item["errors"], "| Error rate:", item["error_rate"])
    display(pd.DataFrame(item["top_confusion_pairs"]))
    display(pd.DataFrame(item["errors_by_true_label"]))

# %% [markdown] cell 31
# ## Celda 14: Cierre
#
# Al terminar, revisa la carpeta de salida. Debe contener configuracion del experimento, prompts, predicciones, metricas, matrices de confusion y analisis de errores. Si `SAVE_FULL_MODEL=True`, tambien se guardaran los pesos completos del modelo, lo cual puede requerir bastante espacio.

# %% [code] cell 32

print("Carpeta de salida:", OUTPUT_DIR)
for path in sorted(OUTPUT_DIR.glob("*")):
    print(path)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

# %% [code] cell 33

