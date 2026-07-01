import sys
import os

# Adiciona o diretório raiz do projeto ao sys.path para permitir imports absolutos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Importar configurações centralizadas
from config.settings import DATASET_DIR, CSV_PATH, FIGURES_DIR, REPORTS_DIR

# Configuração de estilo para os gráficos
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18
})

# Paleta de cores premium
PALETTE = {
    'Normal': '#2ec4b6',  # Emerald/Teal
    'Cyst': '#3a86c8',    # Steel Blue
    'Stone': '#ff9f1c',   # Amber/Orange
    'Tumor': '#e71d36'    # Coral/Red
}

def analyze_dataset():
    print("🔍 Iniciando análise do dataset de TCs de rim...")
    
    # 1. Verificar subdiretórios das classes
    classes = ['Normal', 'Cyst', 'Stone', 'Tumor']
    class_counts = {}
    image_paths_by_class = {}
    
    for cls in classes:
        cls_dir = os.path.join(DATASET_DIR, cls)
        if not os.path.exists(cls_dir):
            print(f"⚠️ Diretório {cls_dir} não encontrado. Tentando localizar em subpastas...")
            # Algumas extrações do Kaggle criam pastas duplicadas
            search_pattern = os.path.join(DATASET_DIR, "**", cls)
            found_dirs = glob.glob(search_pattern, recursive=True)
            if found_dirs:
                cls_dir = found_dirs[0]
                print(f"✅ Encontrado em: {cls_dir}")
            else:
                raise FileNotFoundError(f"Não foi possível encontrar a pasta da classe: {cls}")
                
        # Encontrar todas as imagens (jpg, jpeg, png)
        img_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            img_files.extend(glob.glob(os.path.join(cls_dir, ext)))
            
        class_counts[cls] = len(img_files)
        image_paths_by_class[cls] = img_files
        
    print(f"📊 Contagem de imagens por classe: {class_counts}")
    
    # 2. Ler metadados do CSV (se existir)
    csv_exists = os.path.exists(CSV_PATH)
    csv_summary = ""
    if csv_exists:
        print(f"📄 CSV de metadados encontrado: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
        total_rows = len(df)
        csv_summary = f"- **Registros no CSV (`{os.path.basename(CSV_PATH)}`):** {total_rows} linhas\n"
        if 'Class' in df.columns:
            csv_counts = df['Class'].value_counts().to_dict()
            csv_summary += f"  - Distribuição no CSV: {csv_counts}\n"
    else:
        print(f"ℹ️ CSV de metadados não encontrado em {CSV_PATH}.")
        csv_summary = f"- **Registros no CSV:** Arquivo `{os.path.basename(CSV_PATH)}` não encontrado no diretório local.\n"

    # 3. Analisar resoluções e intensidade dos pixels (usando amostragem de 100 imagens por classe)
    resolutions = []
    channels = []
    pixel_values = []
    
    sample_images = {}
    
    print("📸 Analisando resoluções e intensidades de pixel...")
    for cls in classes:
        paths = image_paths_by_class[cls]
        if not paths:
            continue
        
        # Salva o primeiro para visualização posterior
        sample_images[cls] = paths[0]
        
        # Amostragem para estatísticas rápidas
        sample_size = min(len(paths), 100)
        np.random.seed(42)
        sample_paths = np.random.choice(paths, sample_size, replace=False)
        
        for p in sample_paths:
            # Ler imagem usando OpenCV
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is not None:
                resolutions.append(img.shape[:2])
                ch = img.shape[2] if len(img.shape) > 2 else 1
                channels.append(ch)
                pixel_values.extend(img.flatten())
                
    resolutions = np.array(resolutions)
    channels = np.array(channels)
    pixel_values = np.array(pixel_values)
    
    unique_resolutions = np.unique(resolutions, axis=0)
    unique_channels = np.unique(channels)
    
    pixel_mean = np.mean(pixel_values)
    pixel_std = np.std(pixel_values)
    pixel_min = np.min(pixel_values)
    pixel_max = np.max(pixel_values)
    
    print("📈 Estatísticas dos Pixels:")
    print(f"  - Média: {pixel_mean:.2f}")
    print(f"  - Desvio Padrão: {pixel_std:.2f}")
    print(f"  - Mínimo: {pixel_min}")
    print(f"  - Máximo: {pixel_max}")
    print(f"  - Resoluções encontradas: {unique_resolutions.tolist()}")
    print(f"  - Canais de cores: {unique_channels.tolist()}")

    # 4. Geração de Plots
    # Plot 1: Distribuição das Classes
    print("🎨 Gerando gráficos...")
    plt.figure(figsize=(10, 6))
    colors = [PALETTE[cls] for cls in classes]
    bars = plt.bar(class_counts.keys(), class_counts.values(), color=colors, edgecolor='none', width=0.6)
    
    # Adicionar contagens sobre as barras
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f'{yval:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
    plt.title('Distribuição de Classes no Dataset de TC Renal', pad=20, fontweight='bold')
    plt.xlabel('Classe')
    plt.ylabel('Quantidade de Imagens')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'class_distribution.png'), dpi=300)
    plt.close()
    
    # Plot 2: Exemplos de Imagens (Grid 2x2)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    for i, cls in enumerate(classes):
        img_path = sample_images.get(cls)
        if img_path:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # CT scans são essencialmente grayscale
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Classe: {cls}\n({img.shape[1]}x{img.shape[0]} px)", fontsize=14, pad=10, color=PALETTE[cls], fontweight='bold')
        axes[i].axis('off')
        
    plt.suptitle('Exemplos de Imagens por Classe (TC Renal)', fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'sample_images.png'), dpi=300)
    plt.close()
    
    # Plot 3: Distribuição de Intensidade dos Pixels
    plt.figure(figsize=(10, 6))
    # Selecionar uma amostra de 50.000 pixels para o histograma para ser rápido e preciso
    pixel_sample = np.random.choice(pixel_values, min(len(pixel_values), 50000), replace=False)
    
    sns.histplot(pixel_sample, kde=True, color='#4a5759', bins=50, edgecolor='none')
    plt.title('Distribuição da Intensidade de Pixels (Amostra)', pad=20, fontweight='bold')
    plt.xlabel('Valor do Pixel (0-255)')
    plt.ylabel('Densidade / Contagem')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'pixel_intensity_distribution.png'), dpi=300)
    plt.close()
    
    # 5. Salvar Relatório Markdown
    total_images = sum(class_counts.values())
    
    resolutions_str = ", ".join([f"{r[0]}x{r[1]}" for r in unique_resolutions])
    channels_str = ", ".join([str(c) for c in unique_channels])
    
    report_content = f"""# Relatório de Análise Exploratória do Dataset (EDA)

Este relatório contém os resultados da análise exploratória do dataset de Tomografias Computadorizadas (TC) de rim.

## 📊 Visão Geral do Dataset

- **Diretório dos Dados:** `{os.path.basename(DATASET_DIR)}`
- **Total de Imagens:** {total_images:,}
{csv_summary}

### Distribuição por Classe:
| Classe | Quantidade de Imagens | Proporção (%) |
| :--- | :---: | :---: |
"""
    for cls in classes:
        count = class_counts[cls]
        percentage = (count / total_images) * 100
        report_content += f"| **{cls}** | {count:,} | {percentage:.2f}% |\n"
        
    report_content += f"""| **Total** | **{total_images:,}** | **100.00%** |

---

## 📸 Propriedades Físicas das Imagens

- **Resoluções Identificadas (H x W):** `{resolutions_str}` pixels
- **Canais de Cor:** `{channels_str}` (onde 1 é tons de cinza e 3 é RGB/BGR)
- **Formato Predominante:** JPEG (extensão `.jpg`)

---

## 📈 Estatísticas de Intensidade dos Pixels (Amostra)

Estas estatísticas ajudam a entender a necessidade de etapas de normalização e padronização (como reescalar os pixels para o intervalo `[0, 1]` ou padronização de média zero).

- **Média Global:** `{pixel_mean:.3f}`
- **Desvio Padrão:** `{pixel_std:.3f}`
- **Valor Mínimo de Pixel:** `{pixel_min}`
- **Valor Máximo de Pixel:** `{pixel_max}`

---

## 🎨 Visualizações Geradas

Os seguintes gráficos foram gerados e salvos no diretório `reports/figures/`:

1. **[class_distribution.png](./figures/class_distribution.png)**: Gráfico de barras ilustrando o balanceamento do dataset.
2. **[sample_images.png](./figures/sample_images.png)**: Amostras visuais das tomografias para cada uma das classes.
3. **[pixel_intensity_distribution.png](./figures/pixel_intensity_distribution.png)**: Histograma exibindo a distribuição dos tons de cinza nas imagens.
"""

    report_path = os.path.join(REPORTS_DIR, "dataset_summary.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"✅ Relatório '{os.path.basename(report_path)}' criado com sucesso em {REPORTS_DIR}!")
    print("🎉 Análise concluída!")

if __name__ == "__main__":
    analyze_dataset()
