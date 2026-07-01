import os

# ==============================================================================
# 1. DIRETÓRIOS E CAMINHOS (RESOLVIDOS DINAMICAMENTE)
# ==============================================================================
# Caminho base do projeto (raiz do repositório)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Pasta de dados e dataset
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_DIR = os.path.join(DATA_DIR, "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone")
CSV_PATH = os.path.join(DATA_DIR, "kidneyData.csv")

# Pasta para salvamento de modelos
MODELS_DIR = os.path.join(BASE_DIR, "modelos")

# Pasta de relatórios e gráficos
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# Garantir que as pastas de relatórios existam localmente
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ==============================================================================
# 2. HIPERPARÂMETROS DO MODELO E TREINAMENTO
# ==============================================================================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE_BINARY = 1e-4      # Usada no kidneymodel.py (Normal vs Tumor)
LEARNING_RATE_MULTICLASS = 1e-5    # Usada no training.py (Todas as classes)

# ==============================================================================
# 3. CONFIGURAÇÕES DO MLFLOW
# ==============================================================================
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_LOCAL_URI = f"file:{os.path.join(BASE_DIR, 'mlruns')}"
