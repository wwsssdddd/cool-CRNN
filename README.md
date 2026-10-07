# CRNN + CTC для распознавания текста переменной длины

Современная PyTorch-реализация текстовой части статьи **An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition**. Реализованы обучение, CTC, lexicon-free transcription, lexicon-based transcription, BK-tree и benchmark-оценка. OMR намеренно исключён.

Модель не имеет фиксированной длины слова. Изображение приводится к высоте 32 с сохранением пропорций; его ширина и число CTC-шагов зависят от конкретного примера.

## Как проходит один объект

```text
grayscale image (1, 32, W)
        ↓ CNN
feature map (512, 1, floor(W / 4) + 1)
        ↓ map-to-sequence
T признаков по 512 чисел
        ↓ BiLSTM(512 → 256) → BiLSTM(256 → 37)
CTC logits (T, 37)
```

В одном батче изображения дополняются справа белым фоном до максимальной ширины. Реальные длины передаются в `pack_padded_sequence` и `CTCLoss`, поэтому padding не участвует ни в BiLSTM, ни в функции потерь.

## Структура

```text
crnn-reproduction/
├── data/
├── models/
├── src/
│   ├── config.py       # параметры и выбор mps/cuda/cpu
│   ├── dataset.py      # lazy Dataset, resize, variable-length collate
│   ├── model.py        # CNN и два BiLSTM
│   ├── metrics.py      # CER, Levenshtein, word accuracy
│   ├── train.py        # CTC, train/eval loop
│   ├── pipeline.py     # полный запуск
│   ├── ctc_probability.py # точная CTC-вероятность Eq. (1)
│   ├── bktree.py       # поиск N_delta(l0)
│   ├── lexicon.py      # словарное декодирование Eq. (2)
│   ├── infer.py        # распознавание одного изображения
│   ├── benchmark.py    # IIIT5K/SVT/IC03/IC13 и Figure 4
│   └── analysis.py     # визуализация ошибок
├── configs/paper.env.example
├── experiments/paper_reference.json
└── tests/
```

## Данные

Есть два поддерживаемых формата.

### Вариант 1: ответ в имени файла

Сканирование рекурсивное; длины слов могут различаться:

```text
data/
├── cat.png
├── words/
│   ├── recognition.jpg
│   └── network7.png
└── verylongword42.png
```

### Вариант 2: TSV manifest

Используйте его, если имя файла не равно ответу:

```text
images/000001.jpg	cat
images/000002.jpg	recognition
images/000003.jpg	letter
```

Пути относительно `CRNN_DATA_PATH`. Запуск:

```bash
CRNN_DATA_PATH=/path/to/dataset \
CRNN_MANIFEST=/path/to/train.tsv \
python -m src.pipeline
```

Для серверного обучения предпочтительны отдельные manifests без случайного split:

```bash
CRNN_DATA_PATH=/path/to/dataset \
CRNN_TRAIN_MANIFEST=/path/to/train.tsv \
CRNN_VAL_MANIFEST=/path/to/val.tsv \
python -m src.pipeline
```

Текущий алфавит статьи: цифры `0-9` и строчные латинские буквы `a-z`. Регистр автоматически приводится к нижнему.

### Подготовка manifest

Из имён обычных файлов:

```bash
python -m src.prepare_manifest \
  --root /data/images \
  --format filename \
  --train-output /data/manifests/train.tsv \
  --val-output /data/manifests/val.tsv \
  --val-fraction 0.02
```

Для MJSynth, где слово находится во второй части имени `_word_`:

```bash
python -m src.prepare_manifest \
  --root /data/mjsynth \
  --annotations /data/mjsynth/annotation_train.txt \
  --format mjsynth \
  --train-output /data/manifests/synth_train.tsv \
  --val-output /data/manifests/synth_val.tsv \
  --val-fraction 0.02 \
  --skip-invalid
```

## Локальная установка на Mac

```bash
cd /path/to/crnn-reproduction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest -q
```

Проверка MPS:

```bash
python - <<'PY'
import torch
print(torch.backends.mps.is_available())
PY
```

## Smoke-тест обучения

```bash
CRNN_DATA_PATH=/path/to/small-dataset \
CRNN_DEVICE=mps \
CRNN_EPOCHS=1 \
CRNN_BATCH_SIZE=8 \
MPLBACKEND=Agg \
python -u -m src.pipeline
```

## Полное обучение

```bash
CRNN_DATA_PATH=/path/to/dataset \
CRNN_TRAIN_MANIFEST=/path/to/train.tsv \
CRNN_VAL_MANIFEST=/path/to/val.tsv \
CRNN_MODELS_PATH=./models/run-01 \
CRNN_DEVICE=cuda \
CRNN_EPOCHS=5 \
CRNN_BATCH_SIZE=64 \
MPLBACKEND=Agg \
python -u -m src.pipeline
```

Результаты:

```text
models/run-01/model
models/run-01/best.pt
models/run-01/last.pt
models/run-01/training_history.png
```

`last.pt` сохраняется после каждой эпохи и содержит модель, оптимизатор и историю.
После обрыва продолжите до общего числа `CRNN_EPOCHS`:

```bash
CRNN_RESUME=./models/run-01/last.pt \
CRNN_EPOCHS=5 \
python -u -m src.pipeline
```

Здесь `5` — общее конечное число эпох; подберите его по validation CER.

## Переменные окружения

| Переменная | Назначение | По умолчанию |
|---|---|---|
| `CRNN_DATA_PATH` | корень изображений | `./data` |
| `CRNN_MANIFEST` | необязательный TSV | не задан |
| `CRNN_TRAIN_MANIFEST` | отдельный train TSV | не задан |
| `CRNN_VAL_MANIFEST` | отдельный validation TSV | не задан |
| `CRNN_MODELS_PATH` | каталог результатов | `./models` |
| `CRNN_DEVICE` | `mps`, `cuda` или `cpu` | выбирается автоматически |
| `CRNN_MIN_IMAGE_WIDTH` | минимальная ширина после resize | `100` |
| `CRNN_TRAIN_IMAGE_WIDTH` | фиксированная train-ширина; `0` сохраняет пропорции | `0` |
| `CRNN_EPOCHS` | эпохи; сначала smoke, затем до сходимости | `1` |
| `CRNN_BATCH_SIZE` | размер батча | `64` |
| `CRNN_LR` | learning rate Adadelta | `1.0` |
| `CRNN_NUM_WORKERS` | процессы DataLoader | `0` |
| `CRNN_RESUME` | путь к `last.pt` | не задан |

## Практические ограничения

«Произвольная длина» означает отсутствие фиксированной константы длины слова. Физический предел всё равно определяется памятью: длинное слово создаёт широкое изображение и длинную последовательность. Код автоматически растягивает слишком узкое изображение так, чтобы CTC мог разместить все символы и необходимые blank между одинаковыми соседними буквами.

## Inference и словарь

Без словаря:

```bash
python -m src.infer --checkpoint models/run-01/model --image word.jpg
```

Полный перебор небольшого словаря:

```bash
python -m src.infer \
  --checkpoint models/run-01/model \
  --image word.jpg \
  --lexicon lexicon_50.txt
```

BK-tree для большого словаря, как в статье (`delta=3`):

```bash
python -m src.infer \
  --checkpoint models/run-01/model \
  --image word.jpg \
  --lexicon hunspell_50k.txt \
  --delta 3
```

## Benchmark-протоколы статьи

Для per-image словарей используйте JSONL:

```json
{"image":"images/0001.jpg","label":"hello","lexicons":{"50":"lexicons/0001_50.txt","1k":"lexicons/0001_1k.txt","50k":"lexicons/hunspell_50k.txt"}}
```

Lexicon-free:

```bash
python -m src.benchmark \
  --checkpoint models/run-01/model \
  --manifest benchmarks/iiit5k.jsonl \
  --data-root /data/iiit5k \
  --output-dir results/iiit5k_none
```

Точный перебор per-image 50-словного лексикона:

```bash
python -m src.benchmark \
  --checkpoint models/run-01/model \
  --manifest benchmarks/iiit5k.jsonl \
  --data-root /data/iiit5k \
  --lexicon-key 50 \
  --exhaustive \
  --output-dir results/iiit5k_50
```

Figure 4, IC03 с глобальным 50k-словарём:

```bash
python -m src.benchmark \
  --checkpoint models/run-01/model \
  --manifest benchmarks/ic03.jsonl \
  --data-root /data/ic03 \
  --global-lexicon /data/lexicons/hunspell_50k.txt \
  --deltas 0 1 2 3 4 5 \
  --output-dir results/ic03_figure4
```

Команда создаёт `summary.json`, предсказания каждого режима и `figure4_delta.png`.

Проверка Table 3:

```bash
python -m src.model_stats
```

Объединение результатов наборов в CSV/Markdown:

```bash
python -m src.summarize_benchmarks \
  IIIT5K=results/iiit5k_none/summary.json \
  SVT=results/svt_none/summary.json \
  IC03=results/ic03_none/summary.json \
  IC13=results/ic13_none/summary.json \
  --output results/table2.csv
```

Эталонные значения статьи записаны в `experiments/paper_reference.json`. Практические ограничения остаются физическими: очень длинное слово требует широкой картинки и больше памяти. Единственная намеренно исключённая часть статьи — OMR.
