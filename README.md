# NLP-projects

Pet-проекты по обработке естественного языка, выполненные в рамках курса
**«Обработка естественного языка»**, 2025–2026.

> **Статус:** репозиторий формируется. Проекты переносятся из закрытого учебного репозитория и добавлением подробных README по каждому проекту.

---

## Содержание

| # | Проект | Тематика | Статус |
|---|--------|----------|--------|
| 1 | [BERT и Multi-Head Attention](./01-bert-attention/) | Реализация и применение BERT, механизм многоглавного внимания | переносится |
| 2 | [Seq2Seq и машинный перевод](./02-seq2seq-translation/) | Encoder-Decoder архитектуры, машинный перевод | переносится |
| 3 | [NER — извлечение именованных сущностей](./03-ner/) | Распознавание именованных сущностей, BIO-разметка, fine-tuning BERT | переносится |
| 4 | [LLM и итоговый проект](./04-llm-final/) | Большие языковые модели, prompt engineering, fine-tuning | переносится |

---

## 1. BERT и Multi-Head Attention

**Задача:** разобраться с архитектурой Transformer и BERT — реализовать механизм
multi-head self-attention с нуля, изучить позиционные эмбеддинги и токенизацию,
применить предобученный BERT к задачам классификации текста.

**Стек:** PyTorch, transformers, HuggingFace Datasets, scikit-learn.

---

## 2. Seq2Seq и машинный перевод

**Задача:** построить модель машинного перевода на базе encoder-decoder
архитектуры. Реализованы вариант на RNN/LSTM с attention и вариант на
Transformer, сделано сравнение качества по BLEU.

**Стек:** PyTorch, sacreBLEU, sentencepiece, HuggingFace Datasets.

---

## 3. NER — извлечение именованных сущностей

**Задача:** распознавание именованных сущностей (персоны, организации, локации,
даты) на тексте с помощью fine-tuning предобученного BERT. Работа с
BIO-разметкой, метриками F1 на уровне сущностей, обработкой подтокенов.

**Стек:** PyTorch, transformers (BertForTokenClassification), seqeval,
HuggingFace Datasets.

---

## 4. LLM и итоговый проект

**Задача:** работа с большими языковыми моделями — prompt engineering,
few-shot learning, базовый fine-tuning через PEFT/LoRA, оценка качества
генерации.

**Стек:** transformers, peft, accelerate, OpenAI API, Anthropic Claude API.

---

## Автор

**Николай Пахомов** — ML / NLP / LLM Engineer.
- Telegram: [@kikoooiemama](https://t.me/kikoooiemama)
- Email: nikolay.pakhomov.ds@gmail.com
- GitHub: [kikoooiemama](https://github.com/kikoooiemama)
