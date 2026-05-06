# Fashion Product Search: Fine-tuned CLIP

Поисковая система по каталогу одежды, аксессуаров и обуви: пользователь
вводит текстовый запрос на английском («red skirt», «blue sunglasses»,
«black leather boots») — система возвращает наиболее релевантные изображения
товаров из базы.

В основе — предобученная **CLIP**, дообученная на парах
`(изображение товара, описание)` для адаптации к домену fashion-каталога.

**Достигнуто:**

| Метрика | Значение |
|---|---:|
| Целевой средний CLIP score | **> 30** |
| Best validation CLIP score | **30.57** (epoch 3) |

Поиск по предвычисленным эмбеддингам отрабатывает за доли секунды на
~39K товаров — один матричный продукт `text_emb @ image_embs.T`.

---

## Данные

[**Fashion Product Images and Text Dataset**](https://www.kaggle.com/datasets/nirmalsankalana/fashion-product-text-images-dataset)
— ~44K товаров с Kaggle: изображение, описание, категория, display name.

**EDA:**
- 44 441 строка, 142 категории. Топ: T-shirts (7K), Shirts (3.2K),
  Casual Shoes (2.8K), Watches (2.5K).
- Все картинки одинакового разрешения 1080×1440, белый/нейтральный фон.
- Описания подробные, средняя длина 556 символов (CLIP processor обрежет
  до 77 токенов — ключевая информация обычно в начале).

**Предобработка:**
- Удаление пропусков в `description` / `image` → 44 158.
- Удаление дубликатов по `image` → 44 158 (нет дубликатов изображений).
- **Удаление дубликатов по `description` → 38 854.** Это важный шаг:
  ~5300 пар имели одинаковый текст для разных картинок. Для contrastive
  learning это шум — модель получает противоречивый сигнал, когда один
  и тот же текст должен быть «близко» к разным картинкам в одном батче.
- Train/test split 90/10 → 34 968 / 3 886.

---

## Fine-tuning

### Модель

[`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32)
— ViT-B/32 vision encoder + text Transformer, обученный на 400M пар
изображение-текст из интернета. Дообучается целиком (не frozen).

### Loss — стандартный симметричный CLIP loss (InfoNCE)

В каждом батче строится матрица сходств `B×B` между всеми парами
(image, text). На диагонали — правильные пары, всё остальное — негативные.
Loss применяется **симметрично** в обе стороны:

```python
def clip_loss(logits_per_image, logits_per_text):
    labels = torch.arange(batch_size)
    loss_i2t = cross_entropy(logits_per_image, labels)  # image → text
    loss_t2i = cross_entropy(logits_per_text, labels)   # text → image
    return (loss_i2t + loss_t2i) / 2
```

Эффект: эмбеддинги правильных пар сближаются, неправильные — отталкиваются.

### Гиперпараметры

| | |
|---|---|
| Backbone | `openai/clip-vit-base-patch32` |
| Optimizer | AdamW (lr=5e-6, weight_decay=0.01) |
| Batch size | 32 |
| Epochs | 5 |
| Train batches per epoch | 1093 |
| Время эпохи | ~16 минут (GPU, локально) |
| Loss | symmetric InfoNCE (CLIP loss) |

### Динамика обучения

| Epoch | train_loss | train_clip_score | val_clip_score |
|---:|---:|---:|---:|
| 1 | 0.265 | 29.78 | 30.36 |
| 2 | 0.149 | 30.63 | 30.16 |
| **3** | **0.108** | **30.99** | **30.57** ← best |
| 4 | 0.089 | 31.38 | 30.32 |
| 5 | 0.076 | 31.44 | 30.54 |

После 3-й эпохи **train продолжает расти, validation стагнирует** — начало
переобучения. Для системы поиска используется чекпоинт **epoch 3**.

---

## Поисковая система

### Предвычисление эмбеддингов

Чтобы поиск работал быстро, эмбеддинги всех 38 854 изображений каталога
вычисляются один раз (`get_image_features`), нормализуются (L2) и сохраняются
на диск (`image_embeddings.pt`). При запросе пересчитывается только эмбеддинг
текста.

### Поиск

```python
def search_products(model, processor, image_embeddings, valid_indices,
                    full_df, text_query, top_k=5, device='cuda'):
    # 1. Эмбеддинг текста (один проход через text encoder)
    text_features = model.get_text_features(...)
    text_features = F.normalize(text_features, dim=-1)

    # 2. Cosine similarity со всеми изображениями (одно матричное умножение)
    similarities = image_embeddings @ text_features.T

    # 3. Top-K
    top_k_values, top_k_indices = similarities.topk(top_k)
    return [...]
```

Поиск по всему каталогу — это **одна операция матричного умножения**
`(N, D) × (D, 1) → (N, 1)`. На 39K векторов размерности 512 — миллисекунды.

### Тестовые запросы

Система проверена на 15 разнообразных запросах:

| Тип | Примеры |
|---|---|
| Цвет + категория | `red skirt`, `blue sunglasses`, `purple tie`, `beige bag` |
| Материал + категория | `black leather boots`, `white sports shoes` |
| Композитные | `pretty dress for girls`, `navy blue T-shirt`, `men's yellow pants` |
| Абстрактные/доменные | `mickey mouse` → футболки с принтами Disney |
| Косметика | `crimson lipstick` |
| Бельё | `red bra`, `blue socks` |

Качественный анализ показал релевантную выдачу по всем категориям. Особенно
интересно срабатывание абстрактных запросов: «mickey mouse» возвращает
футболки с принтами персонажа, что демонстрирует сохранённое от базового
CLIP мульти-доменное понимание текста.

---

## Стек

PyTorch · transformers (CLIP) · scikit-learn · pandas · matplotlib · seaborn · PIL · tqdm

Полный список зависимостей — `requirements.txt`.

---

## Структура папки

```
04-fashion-product-search/
├── README.md                       — этот файл
├── 01_fashion_clip_search.ipynb    — полный пайплайн: fine-tuning + поисковая система
└── requirements.txt                — закреплённые версии зависимостей
```

**Данные** не коммитятся — ~44K изображений (несколько ГБ) скачиваются с
Kaggle по ссылке в первой ячейке.

**Артефакты** (`checkpoints/clip_epoch_*.pt`, `image_embeddings.pt`)
генерируются при запуске и кешируются на диск.

---

## Возможные улучшения

- **Расширенная аугментация изображений** (random crop, color jitter) для
  большей устойчивости к вариациям представления товара.
- **Сильный backbone** — `openai/clip-vit-large-patch14` или
  `laion/CLIP-ViT-H-14-laion2B` для более качественных эмбеддингов.
- **Hard negative mining** — целенаправленный отбор сложных негативных
  примеров (товары той же категории, но разного цвета/стиля).
- **FAISS-индекс** вместо `topk` по полной матрице — критично при
  масштабировании до миллионов товаров.
- **Двуязычный поиск** — fine-tuning на multilingual CLIP
  (`M-CLIP`, `XLM-Roberta-Large-Vit-B-32`) для поддержки русскоязычных
  запросов.
- **Reranker второго уровня** — после top-50 от CLIP применять более
  тяжёлую vision-language модель (BLIP-2, LLaVA) для уточнения порядка.
