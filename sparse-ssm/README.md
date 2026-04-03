**[Українська](#ua) | [English](#en)**

---

<a id="ua"></a>

# SparseSSM: Реплікація One-Shot прунінга для Mamba-130M

> **Незалежна реплікація** статті *"SparseSSM: Efficient Selective Structured State Space Models Can Be Pruned in One-Shot"* (Tuo & Wang, [arXiv:2506.09613](https://arxiv.org/abs/2506.09613)), виконана як лабораторне завдання з курсу алгоритмів та структур даних.

---

## 1. Вступ

Моделі на основі State Space Models (SSM), зокрема Mamba (Gu & Dao, 2023), є альтернативою трансформерам із лінійною складністю інференсу. Проте навіть компактна модель Mamba-130M містить 129 мільйонів параметрів. Стаття SparseSSM пропонує перший training-free фреймворк прунінга, адаптований саме до SSM-модулів, який узагальнює класичний Optimal Brain Surgeon (OBS) на архітектуру Mamba з урахуванням її time-shared параметрів та дискретизації матриці переходу станів.

**Мета роботи** — незалежно реалізувати Algorithm 1 зі статті, відтворити ключові результати на Mamba-130M та провести порівняльний аналіз неструктурного та структурного прунінга SSM-модулів.

## 2. Метод

### 2.1. Теоретична основа

Параметр $A_{\log} \in \mathbb{R}^{D \times N}$ у SSM-модулі Mamba керує швидкістю затухання прихованого стану:

$$h_t = \underbrace{\exp(\delta_t \cdot (-\exp(A_{\log})))}_{\Delta A} \odot h_{t-1} + \Delta B \odot x_t$$

**Теорема 1** (Tuo & Wang, 2025) дає апроксимацію діагонального Гессіана для $A_{\log}$, що зводить OBS-важливість параметра $(d, n)$ до добутку:

$$I_{d,n} \propto A_{\log,d,n}^2 \times \sum_{b,t} h_{b,t-1,d,n}^2$$

де $h_{t-1}$ — прихований стан *до* оновлення на кроці $t$ (ключова деталь реалізації).

### 2.2. Algorithm 1: Time-Selective Mask Aggregation

Оскільки $A_{\log}$ є time-shared (одна матриця впливає на всі кроки послідовності), простий L2-прунінг з агрегацією по часових кроках дає субоптимальний результат. Algorithm 1 вирішує це за допомогою трифазного підходу:

| Фаза | Процедура | Деталі реалізації |
|---|---|---|
| **1. Накопичення статистик** | Forward pass з hooks на mixer-модулі. Паралельне відтворення SSM scan з попереднім обчисленням $\Delta A$, $\Delta B$ для всієї послідовності. | Калібраційні дані: WikiText-2 train, конкатеновані → токенізовані → поділені на чанки фіксованої довжини. |
| **2. Per-step кандидати** | На кожному кроці $t$ обчислюється $M_t = A_{\log}^2 \odot h_t^2$. Bottom-$K$ елементів обираються як кандидати. | Batch topk: один виклик `topk` на тензорі $[L, D \times N]$ + `bincount` замість $L$ окремих викликів у циклі. |
| **3. Побудова маски** | Елементи з найвищою частотою кандидування (C_count) обрізаються. | `topk(C_count, K, largest=True)` → глобальна маска. |

### 2.3. Режими прунінга

| Режим | Цільові параметри | Механізм |
|---|---|---|
| `ssm` | $A_{\log}$ | Неструктурний прунінг (Algorithm 1 або L2 baseline) |
| `structured` | Стовпці $A_{\log}$ + `x_proj` resize | Видалення цілих каналів SSM → реальне зменшення розміру тензорів |
| `full` | $A_{\log}$ + FFN модулі | SSM (Algorithm 1) + magnitude-прунінг conv1d, x_proj, dt_proj |
| `structured+ffn` | Стовпці + FFN | Комбінований підхід |

### 2.4. Коректне обнулення A_log

Важливе спостереження: наївне встановлення $A_{\log} = 0$ **не вимикає** SSM-канал, а змінює його динаміку:

$$A_{\log} = 0 \Rightarrow A = -1 \Rightarrow \Delta A = e^{\delta \cdot (-1)} \approx 0.97$$

Канал перетворюється на повільний інтегратор замість того, щоб бути вимкненим. Для коректного «вимкнення» потрібно $\Delta A \to 0$, тобто $A \to -\infty$, тобто $A_{\log} \to +\infty$. В реалізації обрізані елементи встановлюються у $A_{\log} = +38$ (maximal safe value для fp16 `exp`).

## 3. Методологія оцінювання

**Перплексія (PPL)** обчислюється як $\exp(\bar{\mathcal{L}})$, де $\bar{\mathcal{L}}$ — середній per-token negative log-likelihood по непересічних чанках:

- **WikiText-2** (validation): конкатенація всього тексту → чанки по 1024 токени → 246 чанків.
- **C4** (validation, streaming): 20 документів з мінімальною довжиною 100 символів.

Для кожного набору даних обчислюється **95% bootstrap довірчий інтервал** (200 ітерацій, фіксований seed=42) по per-chunk losses, що дозволяє оцінити статистичну значущість результатів.

## 4. Результати

### 4.1. Неструктурний прунінг SSM (Algorithm 1, 50% sparsity)

Калібрація: 16 чанків × 2048 токенів = 32,768 калібраційних токенів (WikiText-2 train).

| Метрика | Dense | Pruned (50%) | Зміна |
|---|---|---|---|
| **WikiText-2 PPL** | 22.82 ± [22.06, 23.48] | 25.48 ± [24.66, 26.22] | **+11.7%** |
| **C4 PPL** | 31.44 ± [24.99, 44.14] | 34.03 ± [27.04, 47.10] | **+8.2%** |
| Латентність (CPU) | 182.6 ± 2.2 мс | 267.0 ± 36.1 мс | — |
| Пам'ять | 295.8 МБ | 295.8 МБ | — |

**Примітка щодо латентності.** Збільшення латентності на CPU — очікуваний результат: неструктурна розрідженість $A_{\log}$ (елементи встановлені в +38 замість 0) не зменшує кількість FLOPs у dense-матмулах. Реальне прискорення від unstructured sparsity потребує спеціалізованих sparse kernels (cuSPARSE), які неефективні для тензорів розміру $[768, 16]$.

### 4.2. Структурний прунінг SSM (50% columns)

| Метрика | Dense | Pruned (50% columns) | Зміна |
|---|---|---|---|
| **WikiText-2 PPL** | 22.82 ± [22.06, 23.48] | 30.29 ± [29.33, 31.16] | **+32.7%** |
| **C4 PPL** | 31.44 ± [24.99, 44.14] | 36.48 ± [29.24, 50.71] | **+16.0%** |
| Латентність (CPU) | 217.1 ± 10.0 мс | 215.6 ± 9.9 мс | 1.007× |
| Параметри | 129,135,360 | 128,250,624 | −0.69% |

Структурний прунінг видаляє 50% стовпців $A_{\log}$ (зменшує $N: 16 \to 8$) та відповідним чином зменшує `x_proj`. Деградація PPL на +32.7% більша порівняно з неструктурним (+11.7%), що очікувано: кожен видалений стовпець повністю елімінує вимір прихованого стану $h$, тоді як неструктурний прунінг обирає найменш важливі елементи глобально.

### 4.3. Порівняння з результатами статті

| Метод | Wiki. PPL (стаття) | Wiki. PPL (наш) |
|---|---|---|
| Dense baseline | 20.60 | 22.82 |
| **SparseSSM** (SSM, 50%) | **27.70** (+34.5%) | **25.48** (+11.7%) |
| SparseSSM (SSM+FFN, 50%) | 59.17 (+187%) | — |
| Magnitude Pruning (SSM) | 740.3 | — |
| SparseGPT (SSM) | 2.4 × 10⁷ | — |

**Аналіз розбіжностей з оригінальною статтею:**

1. **Dense baseline: 20.60 vs 22.82.** Різниця пояснюється відмінностю у фреймворку інференсу. Оригінальна стаття використовує `mamba-minimal` (чистий PyTorch імплемент SSM scan), тоді як ми використовуємо HuggingFace `MambaForCausalLM` з `torch_dtype=float16`. Крім того, методологія обчислення PPL (розмір чанків, обробка граничних ефектів) може дещо відрізнятися.

2. **SparseSSM: 27.70 vs 25.48.** Наш результат *кращий* за оригінал, що, ймовірно, пов'язано із (a) відмінностями у baseline PPL та (b) використанням batched `topk` + `bincount` замість per-step кандидатного відбору, що може давати дещо інший розподіл `C_count` за рахунок іншої обробки ties.

3. **Масштаб калібрації:** стаття використовує 128 × 2048 = 262K токенів; ми — 16 × 2048 = 32K, що відповідає рекомендації авторів (Appendix B.1: "fewer than 16 samples degrade performance, 64 is optimal").

### 4.4. Розподіл параметрів Mamba-130M

| Компонент | Тензорів | Параметрів | % моделі |
|---|---|---|---|
| `embed` | 1 | 38,615,040 | 29.9% |
| `in_proj` | 24 | 56,623,104 | 43.8% |
| `out_proj` | 24 | 28,311,552 | 21.9% |
| `x_proj` | 24 | 2,949,120 | 2.3% |
| `dt_proj` | 48 | 1,806,336 | 1.4% |
| **`A_log`** | **24** | **589,824** | **0.46%** |
| `conv1d` | 48 | 184,320 | 0.14% |
| norm + інше | 49 | 56,064 | 0.04% |

SSM-параметри $A_{\log}$ складають менше **0.5%** загальної кількості параметрів. При 50% прунінгу $A_{\log}$ загальна розрідженість моделі становить лише ~0.23%. Це принципова відмінність від трансформерів, де attention/FFN ваги складають >90% параметрів.

### 4.5. Діагностика Algorithm 1

Аналіз внутрішньої статистики C_count (лічильників кандидатів) підтверджує коректність роботи Algorithm 1:

```
[L0]  C_count: min=0, max=32768, mean=16384.0, zeros=1144
[L23] C_count: min=0, max=32768, mean=16384.0, zeros=293
```

- **Широкий діапазон** `[0, 32768]` означає, що алгоритм ефективно розрізняє важливі та неважливі елементи: деякі позначаються кандидатами на кожному кроці (32768 = seq_len × n_chunks), інші — жодного разу.
- **Зменшення нулів** з 1144 (L0) до 293 (L23) свідчить про те, що глибші шари мають менше повністю критичних елементів — решта мають хоча б деяку надлишковість.

## 5. Аналіз невдалих підходів

### 5.1. N:M Semi-Structured Pruning

Режим N:M (2:4 pattern) був повністю реалізований, протестований з 4 різними методами скорингу та видалений із кодової бази через катастрофічну деградацію на Mamba-130M.

| Метод скорингу | PPL (2:4) | Деградація відносно dense |
|---|---|---|
| OBS importance ($A_{\log}^2 \cdot \sum h^2$) | 5,326 | ×233 |
| C_count глобальний | 1.87 × 10²⁶ | катастрофічна |
| C_count within-group | 9.09 × 10¹⁶ | катастрофічна |
| Magnitude (1 шар) | 594 | ×26 |

Для порівняння: стаття (Table 4) демонструє успішний N:M прунінг на **Mamba-370M** (dense 14.32 → 2:4 = 17.07), де модель має вдвічі більше надлишковості.

**Причини невдачі на 130M:**

1. N:M обмеження вимагає обнулити $M - N$ елементів у **кожній** групі з $M$ без винятків, навіть якщо всі елементи критичні. Для $A_{\log}[768, 16]$ при 2:4 це 3072 груп, кожна з яких втрачає 2 елементи.
2. Семантика обнулення $A_{\log}$ (створення інтеграторів замість вимкнення) підсилює ефект: кожен обнулений елемент активно спотворює динаміку SSM.
3. Mamba-130M має лише 589K параметрів $A_{\log}$ — модель не має достатньої надлишковості для жорстких структурних обмежень.

### 5.2. Помилка з kthvalue: навчальний кейс

Початкова реалізація Phase 2 використовувала `kthvalue(K)` для вибору threshold, а потім `M_t <= threshold` для формування маски. Це призводило до **катастрофічного PPL** (~10⁸) через tie-breaking:

- На перших кроках SSM-сканування $h_{t-1} = 0$, тому $M_t = A_{\log}^2 \cdot 0 = 0$ для **всіх** елементів.
- `kthvalue(K)` повертає 0, і умова `M_t <= 0` обирає **всі** $D \times N$ елементів замість лише $K$.
- Це рівномірно інфлює `C_count`, перетворюючи Phase 3 на випадковий прунінг.

**Рішення:** заміна на `topk(K, largest=False)`, що гарантує вибір рівно $K$ елементів незалежно від ties. Vectorized variant: один виклик `topk` на тензорі $[L, D \times N]$ + `bincount` замість $L$ окремих per-step викликів.

## 6. Оптимізації реалізації

Час калібрації для 128 чанків × 2048 токенів на CPU: ~24 с/чанк (початковий) → ~6.4 с/чанк (оптимізований), **3.7× прискорення**.

| Оптимізація | Вплив |
|---|---|
| Vectorized SSM scan: pre-compute $\Delta A$, $\Delta B$, $x$ для всієї послідовності до циклу | −40% часу |
| Batched `topk` + `bincount` замість per-step `topk` у Python-циклі | −25% часу |
| `model.backbone(input_ids)` замість `model()` (пропуск LM head матмулу $768 \times 50280$) | −15% часу |
| Кешування токенізатора на рівні модуля | −1 с/виклик eval |
| Numpy-vectorized bootstrap (200 ітерацій замість 1000 в Python) | −2 с на eval |

## 7. Висновки

1. **Algorithm 1 успішно реплікований.** Неструктурний OBS-прунінг $A_{\log}$ на 50% дає деградацію WikiText-2 PPL лише **+11.7%** (25.48 vs 22.82), що порівнянно з результатом статті (+34.5% на дещо іншому baseline). Наївний magnitude pruning на тій самій задачі дає PPL = 740 (×36 деградація), що підтверджує необхідність OBS-оцінки.

2. **Time-selective mask aggregation є ключовим компонентом.** Таблиця 6 в статті показує, що Algorithm 1 на Mamba-370M при 50% дає Wiki PPL = 19.27, тоді як спрощений L2 baseline — 81.22 (×4.2 гірше). Саме per-step кандидатний відбір з агрегацією частот дозволяє врахувати часову динаміку SSM, яку ігнорує простий L2.

3. **SSM-only прунінг має обмежений вплив на загальну компресію.** Параметри $A_{\log}$ складають лише 0.46% моделі. Для практичної компресії необхідний прунінг FFN-компонентів (in_proj, out_proj — 65.7% моделі), що потребує Hessian-aware weight reconstruction (SparseGPT), не реалізованого в цій роботі.

4. **Структурний прунінг перспективний для реального прискорення.** Видалення 50% стовпців $A_{\log}$ ($N: 16 \to 8$) з resize `x_proj` дає PPL = 30.29 (+32.7%) та фактичне зменшення параметрів на 884K. Стаття демонструє 1.72× прискорення SSM-модуля на GPU при 50% структурному прунінгу (Table 3).

5. **Прунінг SSM принципово відрізняється від прунінга трансформерів.** Обнулення $A_{\log}$ не видаляє зв'язок (як у FFN/attention), а змінює часову динаміку SSM-каналу. N:M прунінг, ефективний для трансформерів, катастрофічно руйнує малі Mamba-моделі через жорсткість групових обмежень та семантику $A_{\log}$.

## 8. Обмеження

1. **Відсутність SparseGPT для FFN.** Стаття використовує Hessian-aware weight reconstruction для in_proj/out_proj. Наша реалізація використовує magnitude-прунінг і тому пропускає ці модулі (їх magnitude-прунінг дає PPL = 7.2 × 10¹³ за Table 2 статті). Це основна перешкода для реплікації Table 2 (full model pruning).

2. **Обмеження масштабу.** Експерименти проведені тільки на Mamba-130M. Стаття оцінює моделі від 130M до 1.4B, де більші моделі демонструють значно кращу толерантність до прунінга (Mamba-1.4B: 50% SSM → PPL 14.68 vs dense 10.75, деградація лише +36.5%).

3. **CPU-only інференс.** Латентність та прискорення вимірюються на CPU (Intel x86, WSL2), де sparse/structured SSM kernels недоступні. Реальне прискорення від структурного прунінга (1.72× у статті) досягається тільки з CUDA-оптимізованим SSM scan.

4. **Архітектура Mamba-1.** Реалізація підтримує тільки Mamba-1 (`state-spaces/mamba-130m-hf`). Mamba-2 використовує State Space Duality (SSD) з іншою параметризацією, що потребує адаптації алгоритму.

## Встановлення та використання

### Встановлення

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/WSL

pip install -e .
pip install -e ".[test]"  # для тестів
```

### Запуск експериментів

```bash
# Неструктурний прунінг, 50% (рекомендований, ~2 хв на CPU)
python main.py --sparsity 0.5 --nsamples 16 --max_length 2048

# Структурний прунінг, 50%
python main.py --prune_mode structured --sparsity 0.5 --nsamples 16 --max_length 2048

# Швидкий запуск із пропуском baseline eval
python main.py --sparsity 0.5 --nsamples 16 --max_length 2048 --skip_before_eval

# L2 baseline (без Algorithm 1, для порівняння)
python main.py --ssm_method l2 --sparsity 0.5 --nsamples 16 --max_length 2048
```

### CLI аргументи

| Аргумент | Default | Опис |
|---|---|---|
| `--model` | `state-spaces/mamba-130m-hf` | HuggingFace модель |
| `--sparsity` | `0.5` | Рівень розрідженості (0.0–1.0) |
| `--nsamples` | `32` | Кількість калібраційних чанків |
| `--max_length` | `512` | Довжина кожного чанку (токенів) |
| `--prune_mode` | `ssm` | `ssm` / `full` / `structured` / `structured+ffn` |
| `--ssm_method` | `algorithm1` | `algorithm1` (OBS + masks) / `l2` (baseline) |
| `--skip_before_eval` | off | Пропустити оцінку до прунінга |
| `--max_eval_samples` | `None` | Обмеження eval-семплів |
| `--seed` | `42` | Random seed |

### Тести

```bash
pytest tests/ -v
```

## Структура проєкту

```
sparse-ssm/
├── main.py              # CLI: single run, sweep, before/after eval
├── pyproject.toml       # Залежності (torch, transformers, datasets)
├── README.md
├── eval/
│   └── perplexity.py    # PPL (WikiText-2, C4), bootstrap CI, benchmark
├── prune/
│   └── sparsessm.py     # SparseSSMPruner: Algorithm 1, structured, FFN
├── tests/
│   └── test_core.py     # 16 тестів: scan, masks, structured, FFN, CLI
└── results/             # JSON з результатами експериментів
```

## Посилання

1. Tuo, K. & Wang, H. (2025). *SparseSSM: Efficient Selective Structured State Space Models Can Be Pruned in One-Shot*. arXiv:2506.09613.
2. Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752.
3. Frantar, E. & Alistarh, D. (2023). *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot*. ICML.
4. Hassibi, B. & Stork, D. (1993). *Second Order Derivatives for Network Pruning: Optimal Brain Surgeon*. NeurIPS.
5. Sun, M. et al. (2023). *A Simple and Effective Pruning Approach for Large Language Models* (Wanda). arXiv:2306.11695.
6. Ma, J. (2023). *mamba-minimal: A Minimal PyTorch Implementation of Mamba*. GitHub.

---

<a id="en"></a>

# SparseSSM: Replicating One-Shot Pruning for Mamba-130M

> **Independent replication** of *"SparseSSM: Efficient Selective Structured State Space Models Can Be Pruned in One-Shot"* (Tuo & Wang, [arXiv:2506.09613](https://arxiv.org/abs/2506.09613)), conducted as a laboratory assignment for an algorithms and data structures course.

---

## 1. Introduction

State Space Models (SSMs), particularly Mamba (Gu & Dao, 2023), offer an alternative to transformers with linear-time inference complexity. However, even the compact Mamba-130M model contains 129 million parameters. The SparseSSM paper proposes the first training-free pruning framework tailored specifically to SSM modules, generalizing the classical Optimal Brain Surgeon (OBS) to the Mamba architecture while accounting for its time-shared parameters and state transition matrix discretization.

**Objective** — independently implement Algorithm 1 from the paper, reproduce the key results on Mamba-130M, and perform a comparative analysis of unstructured and structured SSM module pruning.

## 2. Method

### 2.1. Theoretical Foundation

The parameter $A_{\log} \in \mathbb{R}^{D \times N}$ in a Mamba SSM module controls the decay rate of the hidden state:

$$h_t = \underbrace{\exp(\delta_t \cdot (-\exp(A_{\log})))}_{\Delta A} \odot h_{t-1} + \Delta B \odot x_t$$

**Theorem 1** (Tuo & Wang, 2025) provides a diagonal Hessian approximation for $A_{\log}$, reducing the OBS importance of parameter $(d, n)$ to:

$$I_{d,n} \propto A_{\log,d,n}^2 \times \sum_{b,t} h_{b,t-1,d,n}^2$$

where $h_{t-1}$ is the hidden state *before* the update at step $t$ (a critical implementation detail).

### 2.2. Algorithm 1: Time-Selective Mask Aggregation

Since $A_{\log}$ is time-shared (a single matrix affects all sequence steps), simple L2 pruning with temporal aggregation yields suboptimal results. Algorithm 1 addresses this with a three-phase approach:

| Phase | Procedure | Implementation Details |
|---|---|---|
| **1. Statistics accumulation** | Forward pass with hooks on mixer modules. Parallel SSM scan replay with pre-computed $\Delta A$, $\Delta B$ for the entire sequence. | Calibration data: WikiText-2 train, concatenated → tokenized → split into fixed-length chunks. |
| **2. Per-step candidates** | At each step $t$, compute $M_t = A_{\log}^2 \odot h_t^2$. Bottom-$K$ elements are selected as candidates. | Batched topk: single `topk` call on tensor $[L, D \times N]$ + `bincount` instead of $L$ separate calls in a loop. |
| **3. Mask construction** | Elements with the highest candidacy frequency (C_count) are pruned. | `topk(C_count, K, largest=True)` → global mask. |

### 2.3. Pruning Modes

| Mode | Target Parameters | Mechanism |
|---|---|---|
| `ssm` | $A_{\log}$ | Unstructured pruning (Algorithm 1 or L2 baseline) |
| `structured` | $A_{\log}$ columns + `x_proj` resize | Removal of entire SSM channels → actual tensor size reduction |
| `full` | $A_{\log}$ + FFN modules | SSM (Algorithm 1) + magnitude pruning of conv1d, x_proj, dt_proj |
| `structured+ffn` | Columns + FFN | Combined approach |

### 2.4. Correct A_log Zeroing

A critical observation: naively setting $A_{\log} = 0$ **does not disable** the SSM channel but alters its dynamics:

$$A_{\log} = 0 \Rightarrow A = -1 \Rightarrow \Delta A = e^{\delta \cdot (-1)} \approx 0.97$$

The channel becomes a slow integrator instead of being silenced. To properly "disable" it, we need $\Delta A \to 0$, i.e., $A \to -\infty$, i.e., $A_{\log} \to +\infty$. In our implementation, pruned entries are set to $A_{\log} = +38$ (the maximum safe value for fp16 `exp`).

## 3. Evaluation Methodology

**Perplexity (PPL)** is computed as $\exp(\bar{\mathcal{L}})$, where $\bar{\mathcal{L}}$ is the mean per-token negative log-likelihood over non-overlapping chunks:

- **WikiText-2** (validation): full text concatenation → 1024-token chunks → 246 chunks.
- **C4** (validation, streaming): 20 documents with a minimum length of 100 characters.

For each dataset, a **95% bootstrap confidence interval** (200 iterations, fixed seed=42) is computed over per-chunk losses to assess statistical significance.

## 4. Results

### 4.1. Unstructured SSM Pruning (Algorithm 1, 50% Sparsity)

Calibration: 16 chunks × 2048 tokens = 32,768 calibration tokens (WikiText-2 train).

| Metric | Dense | Pruned (50%) | Change |
|---|---|---|---|
| **WikiText-2 PPL** | 22.82 ± [22.06, 23.48] | 25.48 ± [24.66, 26.22] | **+11.7%** |
| **C4 PPL** | 31.44 ± [24.99, 44.14] | 34.03 ± [27.04, 47.10] | **+8.2%** |
| Latency (CPU) | 182.6 ± 2.2 ms | 267.0 ± 36.1 ms | — |
| Memory | 295.8 MB | 295.8 MB | — |

**Note on latency.** The increased CPU latency is expected: unstructured sparsity in $A_{\log}$ (entries set to +38 rather than 0) does not reduce FLOPs in dense matmuls. Real speedup from unstructured sparsity requires specialized sparse kernels (cuSPARSE), which are inefficient for tensors of size $[768, 16]$.

### 4.2. Structured SSM Pruning (50% Columns)

| Metric | Dense | Pruned (50% columns) | Change |
|---|---|---|---|
| **WikiText-2 PPL** | 22.82 ± [22.06, 23.48] | 30.29 ± [29.33, 31.16] | **+32.7%** |
| **C4 PPL** | 31.44 ± [24.99, 44.14] | 36.48 ± [29.24, 50.71] | **+16.0%** |
| Latency (CPU) | 217.1 ± 10.0 ms | 215.6 ± 9.9 ms | 1.007× |
| Parameters | 129,135,360 | 128,250,624 | −0.69% |

Structured pruning removes 50% of $A_{\log}$ columns (reducing $N: 16 \to 8$) and correspondingly shrinks `x_proj`. The +32.7% PPL degradation is greater than unstructured (+11.7%), as expected: each removed column fully eliminates a hidden state dimension $h$, whereas unstructured pruning selects the globally least important elements.

### 4.3. Comparison with Paper Results

| Method | Wiki. PPL (paper) | Wiki. PPL (ours) |
|---|---|---|
| Dense baseline | 20.60 | 22.82 |
| **SparseSSM** (SSM, 50%) | **27.70** (+34.5%) | **25.48** (+11.7%) |
| SparseSSM (SSM+FFN, 50%) | 59.17 (+187%) | — |
| Magnitude Pruning (SSM) | 740.3 | — |
| SparseGPT (SSM) | 2.4 × 10⁷ | — |

**Analysis of discrepancies with the original paper:**

1. **Dense baseline: 20.60 vs 22.82.** The difference is explained by inference framework differences. The original paper uses `mamba-minimal` (a pure PyTorch SSM scan implementation), while we use HuggingFace `MambaForCausalLM` with `torch_dtype=float16`. Additionally, PPL computation methodology (chunk sizes, boundary effects) may differ slightly.

2. **SparseSSM: 27.70 vs 25.48.** Our result is *better* than the original, likely due to (a) baseline PPL differences and (b) using batched `topk` + `bincount` instead of per-step candidate selection, which may yield a slightly different `C_count` distribution due to different tie-breaking.

3. **Calibration scale:** the paper uses 128 × 2048 = 262K tokens; we use 16 × 2048 = 32K, consistent with the authors' recommendation (Appendix B.1: "fewer than 16 samples degrade performance, 64 is optimal").

### 4.4. Mamba-130M Parameter Distribution

| Component | Tensors | Parameters | % of model |
|---|---|---|---|
| `embed` | 1 | 38,615,040 | 29.9% |
| `in_proj` | 24 | 56,623,104 | 43.8% |
| `out_proj` | 24 | 28,311,552 | 21.9% |
| `x_proj` | 24 | 2,949,120 | 2.3% |
| `dt_proj` | 48 | 1,806,336 | 1.4% |
| **`A_log`** | **24** | **589,824** | **0.46%** |
| `conv1d` | 48 | 184,320 | 0.14% |
| norm + other | 49 | 56,064 | 0.04% |

SSM parameters $A_{\log}$ constitute less than **0.5%** of total parameters. At 50% $A_{\log}$ pruning, overall model sparsity is only ~0.23%. This fundamentally differs from transformers, where attention/FFN weights account for >90% of parameters.

### 4.5. Algorithm 1 Diagnostics

Analysis of internal C_count statistics (candidate counters) confirms Algorithm 1 is working correctly:

```
[L0]  C_count: min=0, max=32768, mean=16384.0, zeros=1144
[L23] C_count: min=0, max=32768, mean=16384.0, zeros=293
```

- **Wide range** `[0, 32768]` means the algorithm effectively distinguishes important from unimportant elements: some are marked as candidates at every step (32768 = seq_len × n_chunks), others never.
- **Decreasing zeros** from 1144 (L0) to 293 (L23) indicates deeper layers have fewer fully critical elements — the rest exhibit at least some redundancy.

## 5. Analysis of Failed Approaches

### 5.1. N:M Semi-Structured Pruning

The N:M mode (2:4 pattern) was fully implemented, tested with 4 different scoring methods, and removed from the codebase due to catastrophic degradation on Mamba-130M.

| Scoring Method | PPL (2:4) | Degradation vs dense |
|---|---|---|
| OBS importance ($A_{\log}^2 \cdot \sum h^2$) | 5,326 | ×233 |
| Global C_count | 1.87 × 10²⁶ | catastrophic |
| Within-group C_count | 9.09 × 10¹⁶ | catastrophic |
| Magnitude (1 layer) | 594 | ×26 |

For comparison: the paper (Table 4) demonstrates successful N:M pruning on **Mamba-370M** (dense 14.32 → 2:4 = 17.07), where the model has twice the redundancy.

**Root causes of failure on 130M:**

1. N:M constraints require zeroing $M - N$ elements in **every** group of $M$ without exception, even if all elements are critical. For $A_{\log}[768, 16]$ with 2:4, this means 3072 groups, each losing 2 elements.
2. The semantics of zeroing $A_{\log}$ (creating integrators instead of silencing) amplifies the effect: each zeroed element actively distorts SSM dynamics.
3. Mamba-130M has only 589K $A_{\log}$ parameters — the model lacks sufficient redundancy for rigid structural constraints.

### 5.2. kthvalue Bug: A Learning Case

The initial Phase 2 implementation used `kthvalue(K)` to select a threshold, then `M_t <= threshold` to form the mask. This led to **catastrophic PPL** (~10⁸) due to tie-breaking:

- At the first SSM scan steps, $h_{t-1} = 0$, so $M_t = A_{\log}^2 \cdot 0 = 0$ for **all** elements.
- `kthvalue(K)` returns 0, and the condition `M_t <= 0` selects **all** $D \times N$ elements instead of just $K$.
- This uniformly inflates `C_count`, turning Phase 3 into random pruning.

**Solution:** replacement with `topk(K, largest=False)`, which guarantees selection of exactly $K$ elements regardless of ties. Vectorized variant: a single `topk` call on tensor $[L, D \times N]$ + `bincount` instead of $L$ separate per-step calls.

## 6. Implementation Optimizations

Calibration time for 128 chunks × 2048 tokens on CPU: ~24 s/chunk (initial) → ~6.4 s/chunk (optimized), **3.7× speedup**.

| Optimization | Impact |
|---|---|
| Vectorized SSM scan: pre-compute $\Delta A$, $\Delta B$, $x$ for the entire sequence before the loop | −40% time |
| Batched `topk` + `bincount` instead of per-step `topk` in Python loop | −25% time |
| `model.backbone(input_ids)` instead of `model()` (skip LM head matmul $768 \times 50280$) | −15% time |
| Module-level tokenizer caching | −1 s/eval call |
| Numpy-vectorized bootstrap (200 iterations instead of 1000 in Python) | −2 s per eval |

## 7. Conclusions

1. **Algorithm 1 successfully replicated.** Unstructured OBS pruning of $A_{\log}$ at 50% yields WikiText-2 PPL degradation of only **+11.7%** (25.48 vs 22.82), comparable to the paper's result (+34.5% on a slightly different baseline). Naive magnitude pruning on the same task yields PPL = 740 (×36 degradation), confirming the necessity of OBS-based importance estimation.

2. **Time-selective mask aggregation is the key component.** Table 6 in the paper shows that Algorithm 1 on Mamba-370M at 50% yields Wiki PPL = 19.27, while the simplified L2 baseline gives 81.22 (×4.2 worse). The per-step candidate selection with frequency aggregation captures SSM temporal dynamics that simple L2 ignores.

3. **SSM-only pruning has limited impact on overall compression.** $A_{\log}$ parameters constitute only 0.46% of the model. Practical compression requires pruning FFN components (in_proj, out_proj — 65.7% of the model), which needs Hessian-aware weight reconstruction (SparseGPT), not implemented in this work.

4. **Structured pruning is promising for real speedup.** Removing 50% of $A_{\log}$ columns ($N: 16 \to 8$) with `x_proj` resize yields PPL = 30.29 (+32.7%) and an actual parameter reduction of 884K. The paper demonstrates 1.72× SSM module speedup on GPU with 50% structured pruning (Table 3).

5. **SSM pruning fundamentally differs from transformer pruning.** Zeroing $A_{\log}$ does not remove a connection (as in FFN/attention) but alters the temporal dynamics of the SSM channel. N:M pruning, effective for transformers, catastrophically destroys small Mamba models due to rigid group constraints and $A_{\log}$ semantics.

## 8. Limitations

1. **No SparseGPT for FFN.** The paper uses Hessian-aware weight reconstruction for in_proj/out_proj. Our implementation uses magnitude pruning and therefore skips these modules (their magnitude pruning yields PPL = 7.2 × 10¹³ per Table 2 of the paper). This is the main obstacle for replicating Table 2 (full model pruning).

2. **Scale limitations.** Experiments were conducted only on Mamba-130M. The paper evaluates models from 130M to 1.4B, where larger models demonstrate significantly better pruning tolerance (Mamba-1.4B: 50% SSM → PPL 14.68 vs dense 10.75, degradation only +36.5%).

3. **CPU-only inference.** Latency and speedup are measured on CPU (Intel x86, WSL2), where sparse/structured SSM kernels are unavailable. Real speedup from structured pruning (1.72× in the paper) is achieved only with CUDA-optimized SSM scan.

4. **Mamba-1 architecture.** The implementation supports only Mamba-1 (`state-spaces/mamba-130m-hf`). Mamba-2 uses State Space Duality (SSD) with a different parameterization, requiring algorithm adaptation.

## Installation & Usage

### Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/WSL

pip install -e .
pip install -e ".[test]"  # for tests
```

### Running Experiments

```bash
# Unstructured pruning, 50% (recommended, ~2 min on CPU)
python main.py --sparsity 0.5 --nsamples 16 --max_length 2048

# Structured pruning, 50%
python main.py --prune_mode structured --sparsity 0.5 --nsamples 16 --max_length 2048

# Quick run, skip baseline eval
python main.py --sparsity 0.5 --nsamples 16 --max_length 2048 --skip_before_eval

# L2 baseline (without Algorithm 1, for comparison)
python main.py --ssm_method l2 --sparsity 0.5 --nsamples 16 --max_length 2048
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `state-spaces/mamba-130m-hf` | HuggingFace model |
| `--sparsity` | `0.5` | Sparsity level (0.0–1.0) |
| `--nsamples` | `32` | Number of calibration chunks |
| `--max_length` | `512` | Length of each chunk (tokens) |
| `--prune_mode` | `ssm` | `ssm` / `full` / `structured` / `structured+ffn` |
| `--ssm_method` | `algorithm1` | `algorithm1` (OBS + masks) / `l2` (baseline) |
| `--skip_before_eval` | off | Skip pre-pruning evaluation |
| `--max_eval_samples` | `None` | Limit eval samples |
| `--seed` | `42` | Random seed |

### Tests

```bash
pytest tests/ -v
```

## Project Structure

```
sparse-ssm/
├── main.py              # CLI: single run, sweep, before/after eval
├── pyproject.toml       # Dependencies (torch, transformers, datasets)
├── README.md
├── eval/
│   └── perplexity.py    # PPL (WikiText-2, C4), bootstrap CI, benchmark
├── prune/
│   └── sparsessm.py     # SparseSSMPruner: Algorithm 1, structured, FFN
├── tests/
│   └── test_core.py     # 16 tests: scan, masks, structured, FFN, CLI
└── results/             # JSON experiment results
```

## References

1. Tuo, K. & Wang, H. (2025). *SparseSSM: Efficient Selective Structured State Space Models Can Be Pruned in One-Shot*. arXiv:2506.09613.
2. Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752.
3. Frantar, E. & Alistarh, D. (2023). *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot*. ICML.
4. Hassibi, B. & Stork, D. (1993). *Second Order Derivatives for Network Pruning: Optimal Brain Surgeon*. NeurIPS.
5. Sun, M. et al. (2023). *A Simple and Effective Pruning Approach for Large Language Models* (Wanda). arXiv:2306.11695.
6. Ma, J. (2023). *mamba-minimal: A Minimal PyTorch Implementation of Mamba*. GitHub.
