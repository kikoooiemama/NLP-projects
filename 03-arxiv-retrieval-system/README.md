# arXiv Retrieval System: Bi-Encoder + LLM Reranker

Двухэтапная retrieval-система для поиска научных статей arXiv по
естественно-языковым запросам. Цель проекта — построить production-grade
поиск с целевой метрикой **MRR@5 > 0.91** и провести профилирование узких мест.

**Достигнуто:**

| Метрика | Значение |
|---|---:|
| **MRR@5** | **0.973** |
| **Hits@1** | 0.956 |
| **Hits@5** | 0.994 |

В 95.6% случаев правильная статья оказывается на 1-й позиции, в 99.4% —
попадает в top-5.

---

## Данные

**База:** `arxiv-metadata-s.json` — 98 213 статей arXiv (`id`, `title`, `abstract`).
**Тест:** `test_sample.csv` — 1000 запросов, у каждого один правильный
`id`-ответ. Покрытие 100% — все целевые ID присутствуют в базе.

**Статистика по корпусу (EDA):**
- Заголовки: 7–381 символ, медиана 72.
- Аннотации: 18–3871 символов, медиана 955, 75-й перцентиль 1291.
- Запросы: 40–238 символов, медиана 124.

`max_length=512` токенов покрывает большую часть title+abstract без потерь.

---

## Архитектура

Классический **two-stage retrieval pipeline**:

```
        Query
          │
          ▼
   ┌──────────────┐
   │  Bi-Encoder  │  ──────►  query embedding (4096-d, normalized)
   └──────────────┘
          │
          ▼
   ┌──────────────┐
   │ FAISS Index  │  ──────►  top-100 кандидатов (cosine similarity)
   │ (IndexFlatIP)│           по 98K документам
   └──────────────┘
          │
          ▼
   ┌──────────────┐
   │ Cross-Encoder│  ──────►  relevance scores (yes/no logits → softmax)
   │   Reranker   │           для 100 пар (query, doc)
   └──────────────┘
          │
          ▼
       top-5
```

### Stage 1: Bi-Encoder + FAISS (быстрый отбор)

**Модель:** [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
— одна из топовых на MTEB-бенчмарке среди моделей сопоставимого размера.
Поддерживает instruction-prefixed запросы:

```
"Instruct: Given a scientific search query, retrieve the most relevant research paper
Query: <user query>"
```

**Pooling:** last-token (как и положено decoder-only моделям), L2-нормализация.

**Индекс:** `faiss.IndexFlatIP` — точный поиск по inner product (= cosine
для нормализованных векторов). Для 98K документов это ~71 мс на запрос — точный
перебор укладывается в реальное время.

### Stage 2: Cross-Encoder Reranker (точное переранжирование)

**Модель:** [`Qwen/Qwen3-Reranker-0.6B`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
— generative reranker из того же семейства. Задача формулируется как
бинарная классификация: «релевантен ли документ запросу — yes/no».

**Формат входа:**
```
<|im_start|>system
Judge whether the Document meets the requirements based on the Query
and the Instruct provided. Note that the answer can only be "yes" or "no".
<|im_end|>
<|im_start|>user
<Instruct>: ...
<Query>:    ...
<Document>: ...
<|im_end|>
<|im_start|>assistant
<think>

</think>
```

**Извлечение score:** берутся логиты последней позиции для токенов `yes` и `no`,
применяется softmax, на выход — вероятность `yes`. Эта вероятность и есть
итоговый relevance score, по которому переранжируются 100 кандидатов.

**Почему пара Embedder + Reranker из одного семейства:** минимизирует
distribution mismatch между этапами — обе модели обучены согласованно на
схожих данных и инструкциях.

---

## Параметры

| | |
|---|---|
| Embedder | `Qwen/Qwen3-Embedding-0.6B` |
| Reranker | `Qwen/Qwen3-Reranker-0.6B` |
| `max_length` | 512 |
| `top_k_retrieval` (кандидатов на reranker) | 100 |
| `top_k_final` | 5 |
| Embedding batch size | 32 |
| Rerank batch size | 8 |
| Документ для индексации | `title + abstract` |
| FAISS index | `IndexFlatIP` (exact search) |

---

## Профилирование

Время на запрос разделено между этапами очень неравномерно:

| Компонент | Время | Доля |
|---|---:|---:|
| Embedding + FAISS search | 0.071 ± 0.011 с | **1.1%** |
| Reranking (100 кандидатов) | 6.39 ± 0.67 с | **98.9%** |
| **Total (1 запрос)** | **6.46 с** | 100% |

Полная оценка на 1000 запросов заняла ~1ч 48мин.

**Узкое место — reranker.** Bi-encoder + FAISS отрабатывают за десятки
миллисекунд, всё время съедает cross-encoder, который обрабатывает
100 пар (query, document) последовательно.

---

## Возможные улучшения по производительности

В порядке убывания эффекта:

1. **Сократить `top_k_retrieval` до 50.** Reranker линеен по числу
   кандидатов — двукратное ускорение «бесплатно». При MRR@5=0.97 запас
   по качеству большой; правильный документ почти всегда попадает в
   top-50 после bi-encoder'а.
2. **Увеличить `rerank_batch_size` 8 → 16/32.** GPU эффективнее на крупных
   батчах. Текущий батч 8 выбран из-за ограничений VRAM на локальной машине;
   на сервере с большей памятью это лёгкий буст 20–30%.
3. **ONNX Runtime / TensorRT для reranker.** Граф-оптимизация и kernel
   fusion дают существенное ускорение без потери качества — главный
   target для production.
4. **Квантизация reranker (INT8 / INT4)** через bitsandbytes или GPTQ.
   Для модели 0.6B эффект скромнее, чем для 7B+, но снижает VRAM и
   ускоряет I/O.
5. **Приближённые индексы FAISS** (IVFFlat, HNSW) — критично при
   масштабировании до миллионов документов; на 98K разница минимальна.
6. **Батчирование запросов** — повышает throughput при пакетной обработке,
   не latency одного запроса.

---

## Стек

PyTorch · transformers · FAISS · sentence-transformers · pandas · matplotlib · tqdm

Полный список зависимостей — `requirements.txt`.

---

## Структура папки

```
03-arxiv-retrieval-system/
├── README.md                      — этот файл
├── 01_retrieval_system.ipynb      — полный пайплайн: EDA, индексация, поиск, оценка
└── requirements.txt               — закреплённые версии зависимостей
```

**Данные** не коммитятся в репозиторий — `arxiv-metadata-s.json` (~220 МБ) и
`test_sample.csv` скачиваются по [ссылке](https://disk.yandex.ru/d/_hlESjxRVivrdg)
и кладутся рядом с ноутбуком.

**Артефакты индекса** (`faiss_index.bin`, `doc_store.json`) генерируются при
первом запуске и кешируются на диск — повторные запуски используют готовый
индекс.
