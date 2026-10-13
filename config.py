import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / 'data'
DATA_ROOT = Path(os.environ.get('DDIB_DATA_DIR', DEFAULT_DATA_DIR)).expanduser()

DATA_DIR = {
    'CMUMOSI': str(DATA_ROOT / 'CMUMOSI'),
    'CMUMOSEI': str(DATA_ROOT / 'CMUMOSEI'),
    'IEMOCAPSIX': str(DATA_ROOT / 'IEMOCAP'),
    'IEMOCAPFOUR': str(DATA_ROOT / 'IEMOCAP'),
}

PATH_TO_RAW_AUDIO = {
    'IEMOCAPSIX': os.path.join(DATA_DIR['IEMOCAPSIX'], 'subaudio'),
    'IEMOCAPFOUR': os.path.join(DATA_DIR['IEMOCAPFOUR'], 'subaudio'),
}

PATH_TO_RAW_FACE = {
    'IEMOCAPSIX': os.path.join(DATA_DIR['IEMOCAPSIX'], 'subvideofaces'),
    'IEMOCAPFOUR': os.path.join(DATA_DIR['IEMOCAPFOUR'], 'subvideofaces'),
}

PATH_TO_TRANSCRIPTIONS = {
    'IEMOCAPSIX': os.path.join(DATA_DIR['IEMOCAPSIX'], 'transcription.csv'),
    'IEMOCAPFOUR': os.path.join(DATA_DIR['IEMOCAPFOUR'], 'transcription.csv'),
}

PATH_TO_FEATURES = {
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'features'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'features'),
    'IEMOCAPSIX': os.path.join(DATA_DIR['IEMOCAPSIX'], 'features'),
    'IEMOCAPFOUR': os.path.join(DATA_DIR['IEMOCAPFOUR'], 'features'),
}

PATH_TO_LABEL = {
    'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'CMUMOSI_features_raw_2way.pkl'),
    'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'CMUMOSEI_features_raw_2way.pkl'),
    'IEMOCAPSIX': os.path.join(DATA_DIR['IEMOCAPSIX'], 'IEMOCAP_features_raw_6way.pkl'),
    'IEMOCAPFOUR': os.path.join(DATA_DIR['IEMOCAPFOUR'], 'IEMOCAP_features_raw_4way.pkl'),
}
