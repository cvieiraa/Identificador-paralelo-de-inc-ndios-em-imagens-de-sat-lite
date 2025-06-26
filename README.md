#  FireDetector - Processamento Paralelo de Imagens TIFF Gigantes

##  Visão Geral

O **FireDetector** é um sistema otimizado para **detecção de focos de incêndio** em imagens geoespaciais gigantescas no formato `.tiff`. O projeto foi construído com foco em **paralelização eficiente** e **uso moderado de RAM**, sendo capaz de analisar imagens com dezenas de gigabytes mesmo em ambientes com recursos limitados.O nosso código faz também a contagem da quantidade dos focos de incêndio

A detecção é feita através da análise de cores no espaço RGB, com aplicação de técnicas de segmentação, morfologia e filtragem de contornos. A execução pode ocorrer de forma **sequencial** ou **paralela**, utilizando **multiprocessing** para maior desempenho.

---

##  Funcionalidades

* Leitura parcial de imagens TIFF com memória mapeada (memmap)
* Detecção de pixels potencialmente vermelhos (focos de incêndio)
* Processamento **tile a tile** com possibilidade de inspeção local
* Modo **paralelo** com uso eficiente de múltiplos núcleos
* Geração de imagem final com marcação visual dos focos
* Exportação de tiles individuais para análise manual

---

##  Arquitetura do Projeto

### 1. Leitura por Tiles

A imagem é lida em blocos (`tiles`) de tamanho definido (padrão: 1024x1024). Isso garante que apenas uma pequena parte da imagem esteja em memória a qualquer momento.

### 2. Detecção RGB

Cada tile passa por uma segmentação por cor:

* O canal R (vermelho) precisa ser alto
* Os canais G e B devem ser baixos
* Após isso, aplica-se morfologia para reduzir ruído
* É feita a extração de contornos e cálculo dos centróides

### 3. Processamento Paralelo (multiprocessing)

O sistema distribui os tiles para diferentes processos, utilizando `multiprocessing.Pool` e leitura com `tiff.memmap`, evitando sobrecarga de RAM.

### 4. Renderização Final

Cada centro de incêndio detectado é desenhado como um círculo vermelho na imagem de saída. O resultado pode ser salvo como `.tiff` ou `.jpg`.

---

##  Execução

### Modo Paralelo

```bash
python fire_detector_parallel.py
```

* Ajuste `num_threads` conforme seu hardware (recomendado: metade ou mais dos núcleos lógicos)

### Modo Sequencial

```bash
python fire_detector_sequencial.py
```

* Para testes, pode-se reduzir o `tile_size` e aumentar o `delay_per_tile_seconds` para simulação.

### Exportar um Tile Específico

```bash
python extrair_tile.py
```

* Extrai um tile localizado em `(x, y)` e salva na pasta `tiles/`

---

## 🛠 Bibliotecas Utilizadas

* `numpy`
* `cv2` (OpenCV)
* `tifffile`
* `tqdm`
* `multiprocessing`
* `os`, `time`

---

##  Requisitos do Sistema

* CPU com 4+ núcleos (para paralelismo ideal)
* 8GB RAM (32GB recomendado)
* HD/SSD com espaço para arquivos TIFF e saída
* Python 3.8+

---

##  Estrutura do Projeto

```
fire_detector/
├── fire_detector_parallel.py       # Modo paralelo
├── fire_detector_sequencial.py     # Modo sequencial
├── extrair_tile.py                 # Ferramenta de inspeção
├── img_gigantes/                   # Imagens .tiff de entrada
├── resultados_paralelo/            # Saída do modo paralelo
├── resultados_sequencial/          # Saída do modo sequencial
└── tiles/                          # Tiles individuais extraídos
```

---

## Resultados de Performance

### Dados Detalhados

| Configuração | Tempo (s) | Tempo (min) | Tiles/s | Speedup | Eficiência (%) |
|--------------|-----------|-------------|---------|---------|----------------|
| **Sequencial (1)** | 1161.51 | 19.36 | 6.37 | 1.00x | 100.0% |
| **2 Threads** | 674.53 | 11.24 | 11.44 | 1.72x | 86.0% |
| **4 Threads** | 672.29 | 11.20 | 11.48 | 1.73x | 43.3% |
| **6 Threads** | 649.99 | 10.83 | 11.91 | 1.79x | 29.8% |
| **8 Threads** | 645.43 | 10.76 | 12.01 | 1.80x | 22.5% |
| **12 Threads** | 642.68 | 10.71 | 12.07 | 1.81x | 15.1% |

### Informações do Teste

- **Imagem processada**: 20GB (118320x61020 pixels)
- **Total de tiles**: 6,960
- **Focos detectados**: 18,060 (consistente em todos os testes)
- **Melhor configuração**: 12 threads (speedup de 1.81x)
- **Configuração recomendada**: 2-4 threads (melhor custo-benefício)

### Análise dos Resultados

✅ **Pontos Positivos:**
- Paralelização reduziu o tempo de processamento em ~45%
- Throughput quase dobrou (6.37 → 12.07 tiles/s)
- Resultados consistentes em todas as configurações

⚠️ **Observações:**
- Ganhos marginais após 2 threads (possível gargalo de I/O)
- Eficiência diminui com mais threads (Lei de Amdahl)
- Melhor custo-benefício entre 2-4 threads

##  Considerações Finais

O projeto demonstra como processar imagens geoespaciais extremamente grandes sem comprometer desempenho ou estabilidade. Com o uso de leitura incremental, paralelismo leve e segmentação baseada em cor, é possível identificar focos de incêndio com alta performance mesmo em ambientes desktop.

Para futuras melhorias, podem ser exploradas:

* Detecção por espectro infravermelho (caso disponível)
* Clusterização de focos
* Integração com modelos de deep learning

---

##  Autores

Guilherme Souza
Caio
Este projeto foi desenvolvido como exploração de técnicas de paralelismo em visão computacional para imagens de satélite.