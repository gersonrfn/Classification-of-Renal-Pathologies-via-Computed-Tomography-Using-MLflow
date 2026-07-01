import sys
import os

# Adiciona o diretório raiz do projeto ao sys.path para permitir imports absolutos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import socket
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# Importar configurações centralizadas
from config.settings import (
    DATASET_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE_MULTICLASS,
    MLFLOW_TRACKING_URI,
    MLFLOW_LOCAL_URI,
    FIGURES_DIR,
    REPORTS_DIR
)

# 1. Configuração do MLflow
def check_local_server(host="127.0.0.1", port=5000):
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False

# Aponta para o servidor MLflow local se disponível; caso contrário, salva localmente
if check_local_server():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"📡 Conectado ao servidor MLflow local em {MLFLOW_TRACKING_URI}.")
else:
    mlflow.set_tracking_uri(MLFLOW_LOCAL_URI)
    print(f"📁 Servidor MLflow não detectado. Salvando logs localmente em '{MLFLOW_LOCAL_URI}'.")

# Define o nome do experimento no painel do MLflow
mlflow.set_experiment("Replica_Kidney_Disease_Classification")

# Ativa o autolog do TensorFlow (grava métricas por época, loss e parâmetros automaticamente)
mlflow.tensorflow.autolog()

# 2. Geradores de Dados com Data Augmentation e Normalização
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2 # 80% treino / 20% validação
)

# Garantir download se a pasta do dataset não estiver presente
if not os.path.exists(DATASET_DIR):
    print("\nDataset não encontrado localmente. Iniciando download via Kagglehub...")
    import kagglehub
    path = kagglehub.dataset_download("nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone")
    DATASET_PATH = os.path.join(path, 'CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone', 'CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone')
else:
    DATASET_PATH = DATASET_DIR

print(f"Usando dataset em: {DATASET_PATH}")

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

num_classes = train_generator.num_classes

# 3. Construção da VGG16 com Fine-Tuning
def construir_modelo(num_classes):
    # Carrega a VGG16 pré-treinada na ImageNet sem a camada de classificação do topo
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    
    # Congela os pesos das primeiras camadas da VGG16 (Transfer Learning)
    for layer in base_model.layers[:-4]:
        layer.trainable = False
        
    # Adiciona a nova cabeça de classificação (Custom Top Layers)
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.6)(x) # Dropout aumentado para maior regularização e mitigar overfitting
    output = Dense(num_classes, activation='softmax')(x)
    
    modelo_final = Model(inputs=base_model.input, outputs=output)
    return modelo_final

# 4. Execução do Experimento Rastreado
with mlflow.start_run(run_name="VGG16_FineTuning_Replica") as run:
    
    # Instanciando e compilando o modelo
    model = construir_modelo(num_classes)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_MULTICLASS),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n--- Iniciando o Treinamento da VGG16 ---")
    
    # Callback de Parada Antecipada (Early Stopping) para evitar overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True
    )
    
    # Treinamento do Modelo
    historico = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Log de parâmetros adicionais manualmente, se necessário
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("learning_rate", LEARNING_RATE_MULTICLASS)
    
    # Salvando o modelo final dentro do artefato do MLflow
    mlflow.keras.log_model(model, artifact_path="modelo_vgg16_rim")
    
    # 5. Geração e Registro da Matriz de Confusão e Relatório
    print("\n--- Gerando a Matriz de Confusão ---")
    
    # Criamos um gerador de validação sem embaralhamento para garantir alinhamento das previsões
    val_generator_eval = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Previsões do modelo
    previsoes = model.predict(val_generator_eval)
    y_pred = np.argmax(previsoes, axis=1)
    y_true = val_generator_eval.classes
    
    # Nomes das classes
    nomes_classes = list(val_generator_eval.class_indices.keys())
    
    # Computar a matriz de confusão
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Plotar a matriz de confusão com Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=nomes_classes, yticklabels=nomes_classes)
    plt.title('Matriz de Confusão - Classificação Renal VGG16')
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Prevista')
    plt.tight_layout()
    
    # Salvar em reports/figures/ e enviar para o MLflow
    caminho_cm = os.path.join(FIGURES_DIR, "matriz_confusao.png")
    plt.savefig(caminho_cm, dpi=300)
    mlflow.log_figure(plt.gcf(), "matriz_confusao.png")
    plt.close()
    
    # Salvar relatório completo de classificação em formato texto e enviar ao MLflow
    report = classification_report(y_true, y_pred, target_names=nomes_classes)
    caminho_report = os.path.join(REPORTS_DIR, "relatorio_classificacao.txt")
    with open(caminho_report, "w") as f:
        f.write(report)
    mlflow.log_artifact(caminho_report)
    
    print("\nRelatório de Classificação:\n", report)
    print("\n[Sucesso] Treinamento concluído, matriz de confusão e relatórios registrados no MLflow!")
