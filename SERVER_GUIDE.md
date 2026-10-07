# Запуск CRNN в `/home/user1/16_team`

Проект размещается в `/home/user1/16_team/crnn-reproduction`. Существующая папка `/home/user1/16_team/coolenv` не изменяется.

## 1. Подключение

```bash
ssh user1@176.123.162.112
cd ~/16_team/crnn-reproduction
```

## 2. Окружение

Не используйте `coolenv`, пока неизвестны его зависимости. Создайте окружение проекта:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Установите CUDA-сборку PyTorch командой из https://pytorch.org/get-started/locally/, затем:

```bash
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pytest -q
```

## 3. Подключение датасета

Код не требует копировать датасет внутрь проекта. Пусть администратор сообщает три пути:

```text
DATA_ROOT=/путь/к/корню/изображений
TRAIN_MANIFEST=/путь/к/train.tsv
VAL_MANIFEST=/путь/к/val.tsv
```

Формат TSV:

```text
relative/path/image.jpg<TAB>correctword
```

Проверка до обучения:

```bash
export CRNN_DATA_PATH=/путь/к/датасету
export CRNN_TRAIN_MANIFEST=/путь/к/train.tsv
export CRNN_VAL_MANIFEST=/путь/к/val.tsv

python - <<'PY'
from src.config import TRAIN_MANIFEST_PATH, VAL_MANIFEST_PATH
from src.dataset import CaptchaDataset
train = CaptchaDataset(manifest_path=TRAIN_MANIFEST_PATH, split=False)
val = CaptchaDataset(manifest_path=VAL_MANIFEST_PATH, split=False)
print('train:', len(train), 'val:', len(val))
for ds in (train, val):
    image, target = ds[0]
    print(tuple(image.shape), len(target))
PY
```

Если manifests отсутствуют, создайте их через `python -m src.prepare_manifest`; примеры находятся в README.

## 4. Smoke-тест

```bash
export CRNN_DEVICE=cuda
export CRNN_MODELS_PATH=$HOME/16_team/crnn-reproduction/models/smoke
export CRNN_EPOCHS=1
export CRNN_BATCH_SIZE=8
export CRNN_NUM_WORKERS=2
export MPLBACKEND=Agg
python -u -m src.pipeline 2>&1 | tee smoke.log
```

## 5. Полное обучение

```bash
tmux new -s crnn
cd ~/16_team/crnn-reproduction
source .venv/bin/activate

export CRNN_DATA_PATH=/путь/к/датасету
export CRNN_TRAIN_MANIFEST=/путь/к/train.tsv
export CRNN_VAL_MANIFEST=/путь/к/val.tsv
export CRNN_DEVICE=cuda
export CRNN_MODELS_PATH=$HOME/16_team/crnn-reproduction/models/run-01
# Пример. В статье нет фиксированного числа эпох: выбирайте по validation CER.
export CRNN_EPOCHS=5
export CRNN_BATCH_SIZE=64
export CRNN_NUM_WORKERS=4
export CRNN_LR=1.0
export CRNN_TRAIN_IMAGE_WIDTH=100
export MPLBACKEND=Agg

python -u -m src.pipeline 2>&1 | tee train-run-01.log
```

Если слова не помещаются в 100 пикселей, установите `CRNN_TRAIN_IMAGE_WIDTH=0`: изображения будут масштабироваться пропорционально.

## 6. Продолжение после сбоя

```bash
export CRNN_RESUME=$HOME/16_team/crnn-reproduction/models/run-01/last.pt
python -u -m src.pipeline 2>&1 | tee -a train-run-01.log
```

`CRNN_EPOCHS` — общее конечное число эпох, а не число дополнительных эпох.

## 7. Результаты

```text
models/run-01/best.pt              лучшие веса
models/run-01/last.pt              полный checkpoint для resume
models/run-01/model                лучшие веса state_dict
models/run-01/training_history.png loss/CER
```
