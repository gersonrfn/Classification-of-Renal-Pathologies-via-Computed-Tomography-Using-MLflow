# Relatório de Análise Exploratória do Dataset (EDA)

Este relatório contém os resultados da análise exploratória do dataset de Tomografias Computadorizadas (TC) de rim.

## 📊 Visão Geral do Dataset

- **Diretório dos Dados:** `CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone`
- **Total de Imagens:** 12,446
- **Registros no CSV (`kidneyData.csv`):** 12446 linhas
  - Distribuição no CSV: {'Normal': 5077, 'Cyst': 3709, 'Tumor': 2283, 'Stone': 1377}


### Distribuição por Classe:
| Classe | Quantidade de Imagens | Proporção (%) |
| :--- | :---: | :---: |
| **Normal** | 5,077 | 40.79% |
| **Cyst** | 3,709 | 29.80% |
| **Stone** | 1,377 | 11.06% |
| **Tumor** | 2,283 | 18.34% |
| **Total** | **12,446** | **100.00%** |

---

## 📸 Propriedades Físicas das Imagens

- **Resoluções Identificadas (H x W):** `504x622, 512x512, 534x660, 544x672, 545x675, 550x680, 552x682, 560x560, 564x697, 566x701, 569x705, 576x713, 591x731, 602x745, 604x746, 605x749, 611x754, 614x758, 619x764, 622x768, 623x771, 630x778, 631x781, 634x785, 636x787, 640x790, 640x791, 644x796, 646x798, 651x804, 655x809, 657x813, 658x813, 670x828, 674x832, 679x838, 680x840, 684x845, 684x847, 691x854, 696x861, 701x867, 702x869, 714x882, 719x888, 722x892, 737x910, 811x1003, 858x1060, 905x1118, 932x1152, 943x1167, 974x1203, 988x1221, 1001x1236, 1029x1271, 1110x1371` pixels
- **Canais de Cor:** `3` (onde 1 é tons de cinza e 3 é RGB/BGR)
- **Formato Predominante:** JPEG (extensão `.jpg`)

---

## 📈 Estatísticas de Intensidade dos Pixels (Amostra)

Estas estatísticas ajudam a entender a necessidade de etapas de normalização e padronização (como reescalar os pixels para o intervalo `[0, 1]` ou padronização de média zero).

- **Média Global:** `39.910`
- **Desvio Padrão:** `64.069`
- **Valor Mínimo de Pixel:** `0`
- **Valor Máximo de Pixel:** `255`

---

## 🎨 Visualizações Geradas

Os seguintes gráficos foram gerados e salvos no diretório `reports/figures/`:

1. **[class_distribution.png](./figures/class_distribution.png)**: Gráfico de barras ilustrando o balanceamento do dataset.
2. **[sample_images.png](./figures/sample_images.png)**: Amostras visuais das tomografias para cada uma das classes.
3. **[pixel_intensity_distribution.png](./figures/pixel_intensity_distribution.png)**: Histograma exibindo a distribuição dos tons de cinza nas imagens.
