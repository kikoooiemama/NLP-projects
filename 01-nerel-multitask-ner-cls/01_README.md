# Multi-task: NER + Event Classification (NEREL)

Multi-task модель для одновременного решения двух задач на русскоязычных
новостных текстах:

- **NER** (token-level, BIO-разметка) — извлечение именованных сущностей
  (персоны, организации, локации, даты и т. д.).
- **Event / Relation Classification** (document-level, multilabel) —
  определение набора событий и отношений, упомянутых в документе.

Идея: общий энкодер для двух задач должен использовать вычислительные ресурсы
эффективнее, чем два отдельных пайплайна, и потенциально улучшать качество за
счёт переноса признаков между задачами.

---

## Данные

[**NEREL**](https://huggingface.co/datasets/iluvvatar/NEREL) — публичный
русскоязычный датасет именованных сущностей и отношений на материале новостей.

**Формат:** JSONL — `train.jsonl`, `dev.jsonl`, `test.jsonl` плюс справочники
типов сущностей (`ent_types.jsonl`) и отношений (`rel_types.jsonl`).

**Объём:** ~750 train / ~190 dev / ~190 test документов.

**Особенности (по результатам EDA):**
- Сильный дисбаланс типов: `PERSON`, `PROFESSION`, `ORGANIZATION` встречаются
  в 5–8 тысяч раз, тогда как `LAW`, `AWARD`, `FACILITY` — в 10–15 раз реже.
- Длина текстов 500–2500 символов (медиана ~1500), есть выбросы до 12000+ —
  требуется truncation или sliding window.
- Высокая плотность разметки: ~40–80 сущностей на документ.

---

## Архитектура

```
ruBERT-base
    │
    ├─ last_hidden_state[:, :, :]  ──→  Dropout ──→  token_cls (Linear)  ──→  BIO logits
    │                                                                          │
    │                                                          CrossEntropy ↓ (ignore_index=-100)
    │                                                                  token_loss
    │
    └─ last_hidden_state[:, 0, :]  ──→  Dropout ──→  cls_cls   (Linear)  ──→  multilabel logits
                                                                               │
                                                            BCEWithLogitsLoss ↓
                                                                       cls_loss
```

**Backbone:** [`ai-forever/ruBert-base`](https://huggingface.co/ai-forever/ruBert-base) — предобученный BERT для русского языка.

**Две головы поверх энкодера:**
- Token-level — линейный слой на каждый токен → BIO-метка для NER.
- Document-level — линейный слой над `[CLS]` → multilabel-классификация
  топ-K событий и отношений (K = 30).

**Multi-task loss с uncertainty weighting (Kendall et al., 2018).** NER-loss
(CrossEntropy) и CLS-loss (BCE) имеют разные масштабы. Вместо ручного подбора
весов модель учит обучаемые параметры `log_sigma_token` и `log_sigma_cls`,
которые автоматически балансируют вклад каждой задачи:

```
loss = exp(-2·log_σ_token) · L_token + log_σ_token
     + exp(-2·log_σ_cls)   · L_cls   + log_σ_cls
```

В обучении видно, что параметры неопределённости медленно расходятся — модель
сама находит баланс, без ручной настройки весов.

---

## Эксперименты

Два прогона с разной длиной контекста — оценка влияния truncation на качество.

### Гиперпараметры (общие)

| | |
|---|---|
| Backbone | `ai-forever/ruBert-base` |
| Optimizer | AdamW (lr=2e-5, weight_decay=0.01) |
| Scheduler | linear warmup (10%) + linear decay |
| Batch size | 16 |
| Gradient clipping | 1.0 |
| Топ-K событий и отношений (CLS) | 30 |
| Multi-task loss | uncertainty weighting (Kendall et al.) |

### Результаты

| Эксперимент | max_length | epochs | Test Token F1 (macro) | Test CLS micro-F1 |
|---|---:|---:|---:|---:|
| Baseline | 256 | 10 | 0.524 | 0.809 |
| **Final** | **512** | **13** | **0.593** | **0.817** |

Увеличение контекста с 256 до 512 токенов даёт **+7 п.п. по Token F1** и
+1 п.п. по CLS F1. Эффект ожидаемый: новостные документы в датасете длиннее
256 токенов, при truncation теряется значимая часть сущностей.

CLS учится быстрее NER (плато с 6-й эпохи), NER требует больше эпох —
задача token-level сложнее по природе.

---

## Анализ ошибок

Типичные ошибки финальной модели на качественном анализе (10 примеров теста):

- **Глаголы → EVENT.** «нашли», «награждён» ошибочно помечаются как `B-EVENT`.
  Модель ассоциирует действия с событиями.
- **Редкие типы пропускаются.** `LAW`, `DISEASE` — мало примеров в трейне,
  модель просто не успевает их выучить.
- **Конфликт пересекающихся типов.** «американский военный» →
  `B-NATIONALITY B-PROFESSION` вместо корректной мульти-метки.
- **Разрыв многословных сущностей** при переходе через границу типа в
  соседнем токене.

CLS-задача стабильнее: основные отношения определяются с уверенностью
0.7–0.96, ошибки в основном на границе порога 0.5.

---

## Квантизация

К финальной модели применена **dynamic post-training quantization** (PyTorch,
INT8 для линейных слоёв) — чтобы оценить пригодность для CPU-инференса.

| Метрика | FP32 baseline | INT8 dynamic | Δ |
|---|---:|---:|---|
| Token F1 (macro) | 0.593 | 0.275 | **−54%** |
| CLS micro-F1 | 0.817 | 0.786 | −4% |
| Размер на диске | 680 MB | 436 MB | −36% |
| Время инференса (CPU, batch=16) | 6850 ms | 3800 ms | **1.8× быстрее** |

**Вывод:** dynamic PTQ применима для CLS-головы (потеря 4% — приемлемо), но
**ломает NER** — token-level задачи накапливают ошибку по всей
последовательности и чувствительнее к снижению точности весов. Для
production-NER нужно Static PTQ с калибровкой или QAT.

---

## Стек

PyTorch · transformers (HuggingFace) · ruBERT · scikit-learn · matplotlib · tqdm

---

## Структура папки

```
01-nerel-multitask-ner-cls/
├── README.md                        — этот файл
├── 01_nerel_multitask_256.ipynb     — baseline эксперимент (max_length=256)
└── 02_nerel_multitask_512.ipynb     — финальный эксперимент (max_length=512) + квантизация
```

Данные NEREL (`train.jsonl`, `dev.jsonl`, `test.jsonl`, `ent_types.jsonl`,
`rel_types.jsonl`).

---

## Возможные улучшения

- **CRF-слой** поверх NER-головы для учёта зависимостей между соседними
  BIO-метками — типичный буст для span-based F1.
- **Балансировка редких типов** — взвешенная CrossEntropy или аугментация
  данных для `LAW`, `DISEASE`, `AWARD`.
- **Static PTQ / QAT** для сохранения качества NER при квантизации.
- **Sliding window** для документов длиннее 512 токенов вместо truncation.
