# Classificação de Doenças Renais por Tomografia Computadorizada (TC)

Este projeto implementa uma pipeline de Deep Learning utilizando a arquitetura **VGG16** (com Transfer Learning e Fine-Tuning) para a classificação de doenças renais a partir de imagens de Tomografia Computadorizada (TC). Os experimentos e métricas são rastreados em tempo real utilizando o **MLflow**.

O dataset utilizado é o [CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone), que possui quatro classes principais:
- **Normal** (Rim Saudável)
- **Cyst** (Cisto Renal)
- **Stone** (Cálculo Renal)
- **Tumor** (Tumor Renal)

---

## 📁 Estrutura do Repositório

O projeto está organizado da seguinte forma para garantir modularidade e facilidade de deploy/compartilhamento no GitHub:

```
.
├── config/
│   └── settings.py          # Hiperparâmetros centrais e caminhos dinâmicos
├── data/
│   ├── .gitkeep             # Garante o rastreamento da pasta de dados no Git
│   └── CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/  # Dataset local (ignorado pelo git)
├── reports/
│   ├── dataset_summary.md   # Relatório de Análise Exploratória (EDA)
│   ├── relatorio_classificacao.txt # Métricas finais da classificação multiclasse
│   └── figures/             # Gráficos de distribuição, exemplos e matriz de confusão
├── src/
│   ├── dataset_analysis.py  # Análise exploratória de dados (EDA)
│   ├── kidneymodel.py       # Treinamento do classificador binário (Normal vs Tumor)
│   └── training.py          # Treinamento do classificador multiclasse (4 classes)
├── .gitignore               # Configurações de exclusão do Git (venv, data, mlflow)
├── README.md                # Documentação técnica do projeto
└── requirements.txt         # Lista de dependências do Python
```

---

## ⚙️ Instalação e Configuração

### 1. Clonar o repositório
```bash
git clone <url-do-seu-repositorio>
cd <nome-do-repositorio>
```

### 2. Configurar o Ambiente Virtual (Virtualenv)
Recomenda-se criar um ambiente virtual isolado para instalar as dependências:

No macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

No Windows (PowerShell):
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Instalar Dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configurar Credenciais do Kaggle (Opcional)
Se o dataset não estiver localmente na pasta `data/`, o script irá baixá-lo automaticamente via Kaggle. Para que isso funcione, garanta que você tenha o arquivo `kaggle.json` configurado em `~/.kaggle/kaggle.json`.

---

## 🚀 Como Executar o Projeto

Todos os scripts devem ser executados a partir do diretório raiz do projeto.

### 1. Análise Exploratória de Dados (EDA)
Para gerar relatórios e gráficos estatísticos sobre a distribuição e propriedades das imagens:
```bash
python src/dataset_analysis.py
```
*Os resultados serão salvos na pasta `reports/` (arquivo `dataset_summary.md` e gráficos em `reports/figures/`).*

### 2. Treinamento Binário (Normal vs Tumor)
Para treinar o modelo na tarefa de detecção binária de tumor renal:
```bash
python src/kidneymodel.py
```

### 3. Treinamento Completo (4 Classes)
Para treinar a rede VGG16 completa utilizando Fine-Tuning para as quatro categorias de imagens:
```bash
python src/training.py
```
*Ao final do treinamento, o script salvará a matriz de confusão e o relatório de classificação sob a pasta `reports/`.*

---

## 📊 Monitoramento de Experimentos com MLflow

O projeto está integrado ao **MLflow** para registrar hiperparâmetros, gráficos e métricas por época automaticamente.

### Servidor Local do MLflow
Se você quiser visualizar os experimentos no painel web, inicie o servidor MLflow local antes do treinamento:
```bash
mlflow server --host 127.0.0.1 --port 5000
```
Acesse no seu navegador: [http://127.0.0.1:5000](http://127.0.0.1:5000)

*Nota: Se o servidor local não for detectado na porta 5000, os scripts automaticamente salvarão os logs na pasta local `mlruns/` (que é ignorada pelo Git).*
