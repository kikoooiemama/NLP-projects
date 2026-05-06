# NLP-projects

Практические проекты по обработке естественного языка, выполненные в рамках
курса «Обработка естественного языка», 2025–2026.

---

## Содержание

| # | Проект | Задача | Ключевой результат |
|---|---|---|---|
| 01 | [Multi-task NER + Event Classification](./01-nerel-multitask-ner-cls/) | Multi-task: token-level NER + document-level multilabel classification | Token F1 = 0.59 · CLS F1 = 0.82 |
| 02 | [Training My Own LLM (Pretrain + SFT)](./02-train-llm-pretrain-sft/) | Pretrain decoder-only трансформера с нуля + SFT Qwen2.5-0.5B | Связная генерация в стиле классики, переход к инструктивному формату |
| 03 | [arXiv Retrieval System](./03-arxiv-retrieval-system/) | Двухэтапный поиск научных статей: bi-encoder + cross-encoder reranker | MRR@5 = 0.973 · Hits@5 = 0.994 |
| 04 | [Fashion Product Search](./04-fashion-product-search/) | Fine-tuning CLIP на каталоге одежды + multi-modal поиск | Val CLIP score = 30.57 · поиск за миллисекунды на 39K товаров |

---

## 01 — Multi-task NER + Event Classification

Multi-task модель для одновременного решения двух задач на русскоязычных
новостных текстах: token-level NER (BIO-разметка) и document-level
multilabel-классификация событий и отношений. Один общий энкодер (ruBERT) с
двумя головами и **uncertainty weighting** (Kendall et al.) для автоматической
балансировки двух loss-функций без ручного подбора весов.

Дополнительно — **dynamic post-training quantization** с честным сравнением
качества и скорости: показано, что INT8 пригоден для CLS-головы (потеря 4%),
но ломает NER (потеря 54%) — token-level задачи требуют Static PTQ или QAT.

**Датасет:** [NEREL](https://huggingface.co/datasets/iluvvatar/NEREL)

**Результаты:** Token F1 (macro) = 0.593 · CLS micro-F1 = 0.817 · 1.8× speed-up при квантизации

→ [Подробное описание](./01-nerel-multitask-ner-cls/)

---

## 02 — Training My Own LLM: Pretrain + SFT

End-to-end pipeline обучения языковой модели в двух стадиях:

1. **Pretrain** — собственный decoder-only трансформер ~150M параметров
   (LlamaConfig, GQA) с кастомным BPE-токенизатором (~3K vocab), обученный
   с нуля на корпусе [RussianNovels](https://github.com/JoannaBy/RussianNovels)
   (Толстой, Достоевский, Гоголь и др.). Модель учится **структуре языка**.
2. **SFT** — дообучение Qwen2.5-0.5B на инструктивном датасете
   [alpaca-cleaned-ru](https://huggingface.co/datasets/d0rj/alpaca-cleaned-ru)
   в диалоговом формате `system / user / assistant`.

Проект демонстрирует обе типичные стадии обучения LLM — pretrain и post-train —
на упрощённом, но методически корректном масштабе.

**Датасеты:** RussianNovels (pretrain) · alpaca-cleaned-ru (SFT)

**Результаты:** связная генерация в стиле русской классики после pretrain;
переход от base-модели к инструктивному формату после SFT

→ [Подробное описание](./02-train-llm-pretrain-sft/)

---

## 03 — arXiv Retrieval System

Production-grade поиск по научным статьям arXiv: пользовательский запрос на
естественном языке → top-5 наиболее релевантных статей из базы 98K документов.

Классическая **two-stage retrieval**: bi-encoder
([Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)) +
FAISS для быстрого отбора 100 кандидатов → cross-encoder reranker
([Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)) для
точного переранжирования через generative yes/no классификацию.

Полное профилирование показало: reranker занимает **98.9%** времени —
это и есть главный target для оптимизации (предложены 6 конкретных стратегий
ускорения).

**Датасет:** arXiv metadata (98K статей, 1000 тестовых запросов)

**Результаты:** MRR@5 = 0.973 (целевая > 0.91) · Hits@1 = 0.956 · Hits@5 = 0.994

→ [Подробное описание](./03-arxiv-retrieval-system/)

---

## 04 — Fashion Product Search

Multimodal-поиск по каталогу одежды и аксессуаров: пользователь вводит запрос
на английском («red skirt», «black leather boots», «mickey mouse») —
система возвращает релевантные изображения товаров.

Fine-tuning [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32)
на ~39K парах (image, description) с симметричным contrastive loss
(InfoNCE). Поиск работает по **предвычисленным эмбеддингам** через одно
матричное умножение — миллисекунды на каталог из 39K товаров.

**Датасет:** [Fashion Product Images](https://www.kaggle.com/datasets/nirmalsankalana/fashion-product-text-images-dataset) (Kaggle, ~44K)

**Результаты:** val CLIP score = 30.57 (целевая > 30) · корректная выдача на
тестовых запросах разных типов: цвет+категория, материал, абстрактные
(«mickey mouse» → футболки с принтами Disney)

→ [Подробное описание](./04-fashion-product-search/)
