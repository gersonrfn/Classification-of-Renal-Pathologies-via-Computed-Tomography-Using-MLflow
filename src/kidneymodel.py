import sys
import os

# Adiciona o diretório raiz do projeto ao sys.path para permitir imports absolutos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import socket
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
from sklearn.utils import class_weight
import kagglehub
import mlflow
import mlflow.tensorflow

# Importar configurações centralizadas
from config.settings import (
    DATASET_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE_BINARY,
    MLFLOW_TRACKING_URI,
    MLFLOW_LOCAL_URI
)

# ==========================================
# 1. VERIFICAÇÃO DE HARDWARE (M4 / APPLE SILICON)
# ==========================================
print("Verificando dispositivos disponíveis...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU (Metal) detectada: {gpus}")
    print("O treinamento utilizará a aceleração do chip M4.")
else:
    print("⚠️ Nenhuma GPU detectada. O treinamento usará apenas a CPU.")

# ==========================================
# 2. DOWNLOAD E CONFIGURAÇÃO DO DATASET
# ==========================================
if not os.path.exists(DATASET_DIR):
    print("\nBaixando o dataset via Kagglehub...")
    path = kagglehub.dataset_download("nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone")
    # O Kaggle extrai para uma pasta dupla dependendo do dataset
    DATASET_PATH = os.path.join(path, 'CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone', 'CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone')
else:
    DATASET_PATH = DATASET_DIR

print(f"Caminho do dataset local: {DATASET_PATH}")

# ==========================================
# 3. PRÉ-PROCESSAMENTO E AUGMENTATION
# ==========================================
print("\nConfigurando pré-processamento e data augmentation...")

train_datagen = ImageDataGenerator(
    rescale=1./255,             
    rotation_range=20,          
    zoom_range=0.2,             
    horizontal_flip=True,       
    validation_split=0.2        
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Carregamento focando apenas nas classes citadas no escopo (Normal vs Tumor)
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',        
    classes=['Normal', 'Tumor'], 
    subset='training'
)

val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    classes=['Normal', 'Tumor'], 
    subset='validation'
)

# ==========================================
# 4. TRATAMENTO DE DESBALANCEAMENTO
# ==========================================
labels = [k for k, v in train_generator.class_indices.items()]
print(f"\nClasses identificadas: {labels}")

# Resetar o gerador para alinhar os índices corretamente
train_generator.reset()

class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

class_weights_dict = dict(enumerate(class_weights))
print(f"Pesos das classes definidos para compensar desbalanceamento: {class_weights_dict}")

# ==========================================
# 5. CONSTRUÇÃO DO MODELO VGG16
# ==========================================
def build_model():
    print("\nCarregando o modelo VGG16 base...")
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # Congela a extração de características da VGG16
    for layer in base_model.layers:
        layer.trainable = False

    # Topo do modelo (Classificador Customizado)
    x = base_model.output
    x = Flatten()(x) 
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.6)(x) 
    predictions = Dense(1, activation='sigmoid')(x) # Saída Binária

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_BINARY),
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
    )
    return model

# ==========================================
# 6. CONFIGURAÇÃO DO MLFLOW
# ==========================================
print("\nConfigurando monitoramento com MLflow...")

def check_local_server(host="127.0.0.1", port=5000):
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False

# Evitamos a porta 5000 no macOS, pois ela é reservada para o AirPlay Receiver (Control Center).
# Se o servidor local estiver ativo na porta 5000, conectamos a ele. Caso contrário, salvamos localmente em './mlruns'.
if check_local_server():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"📡 Conectado ao servidor MLflow local em {MLFLOW_TRACKING_URI}.")
else:
    mlflow.set_tracking_uri(MLFLOW_LOCAL_URI)
    print(f"📁 Servidor MLflow não detectado. Salvando logs localmente em '{MLFLOW_LOCAL_URI}'.")

mlflow.set_experiment("Demo_ML_Pipeline")
# Ativa o autologging automático para o TensorFlow/Keras
mlflow.tensorflow.autolog(log_models=True)

# ==========================================
# 7. TREINAMENTO
# ==========================================
model = build_model()
model.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True
)

print("\nIniciando o treinamento na arquitetura Apple Silicon...")
with mlflow.start_run(run_name="vgg16_normal_vs_tumor") as run:
    # Registra parâmetros e hiperparâmetros adicionais customizados
    mlflow.log_params({
        "image_size": str(IMAGE_SIZE),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE_BINARY,
        "dataset_path": DATASET_PATH
    })
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[early_stop],
        class_weight=class_weights_dict
    )
    
    print(f"\nTreinamento finalizado! Run ID no MLflow: {run.info.run_id}")
