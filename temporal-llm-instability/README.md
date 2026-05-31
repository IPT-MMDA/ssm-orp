## Дослідження temporal instability у LLM (Qwen2.5-1.5B-Instruct)

### Мета

Експериментально перевірити, чи існує **статистично значущий зв’язок** між внутрішньою динамічною нестабільністю прихованих станів трансформера та **помилками** під час авторегресивної генерації відповіді на арифметичні задачі.

Основний артефакт: ноутбук [`Qwen.ipynb`](Qwen.ipynb).

### Чому обрано Qwen2.5-1.5B-Instruct

- Модель достатньо мала для роботи в умовах **обмежених ресурсів** (CPU або одна GPU).
- Підтримує доступ до **прихованих станів** усіх шарів під час генерації — це потрібно для обчислення метрик instability.

### Структура експерименту (`Qwen.ipynb`)

1. **Конфігурація** — модель, `N_EXAMPLES = 300`, `MAX_NEW_TOKENS = 16`, seed.
2. **Синтетичний датасет** — 6 рівнів складності (прості дії, комбінації, word problems).
3. **Інференс** — greedy generation (`do_sample=False`), вилучення hidden states на кожному згенерованому токені.
4. **Метрики на рівні токена**, зокрема:
   - temporal hidden-state divergence (`temporal_delta_mean`, `temporal_cos_mean`, …);
   - layer-wise divergence (`layer_div_mean`, `layer_roughness`, …);
   - pseudo-Lyapunov instability;
   - variance норм hidden states;
   - entropy логітів (`logit_entropy`).
5. **Агрегація на рівні прикладу** — зокрема `instability_peak` (максимум z-score по метриках).
6. **Статистика** — point-biserial correlation, Mann–Whitney U, ROC-AUC.
7. **Візуалізації** — гістограми, boxplot, ROC-криві, trajectories instability.
8. **Висновки** — у фінальній markdown-ячейці ноутбука.

### Встановлення та запуск

#### 1. Середовище

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS / WSL
```

#### 2. Залежності

```bash
pip install torch transformers datasets pandas numpy scipy scikit-learn matplotlib tqdm jupyter
```

Потрібен доступ до Hugging Face Hub для завантаження `Qwen/Qwen2.5-1.5B-Instruct` (перший запуск завантажить ваги моделі).

#### 3. Запуск

```bash
jupyter notebook Qwen.ipynb
```

Або відкрийте `Qwen.ipynb` у VS Code / Cursor і виконуйте ячейки зверху вниз.

**Примітка:** повний прогін на 300 прикладах може зайняти **десятки хвилин на CPU** (~5 с на приклад у збереженому запускі). Для швидкої перевірки зменшіть `N_EXAMPLES` у конфігурації.

### Вихідні файли

Після блоку **SAVE** у ноутбуку зберігаються (у робочій директорії):

| Файл | Опис |
|------|------|
| `token_metrics.csv` | Метрики instability для кожного згенерованого токена |
| `sample_metrics.csv` | Агреговані метрики, правильність відповіді, текст генерації |

### Основні результати (збережений запуск)

| Показник | Значення |
|----------|----------|
| Точність на синтетичних задачах | ~73.3% |
| `instability_peak` vs помилка (point-biserial) | r ≈ 0.26, p < 0.00001 |
| ROC-AUC (`instability_peak` → error) | ≈ 0.62 |
| ROC-AUC (`mean_logit_entropy` → error) | ≈ 0.76 |
| ROC-AUC (`max_logit_entropy` → error) | ≈ 0.71 |

Неправильні відповіді в середньому супроводжуються **вищою instability**, більшою дисперсією внутрішньої динаміки та **більшою ентропією** вихідного розподілу.

### Обмеження

- Модель 1.5B параметрів; експерименти на **синтетичній** арифметиці (не GSM8K).
- Використано спрощені proxy-метрики instability, без повного аналізу динамічних систем.
- Attention maps та Jacobian-based sensitivity не аналізувались.

### Перспективи

Масштабування на більші моделі, складніші reasoning-benchmarks, аналіз attention instability та causal зв’язків між instability і hallucinations.
