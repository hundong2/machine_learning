# TimesFM: Google's decoder-only foundation model for time-series forecasting

**TimesFM is a patched, decoder-only transformer that treats time-series forecasting like language modeling — predict the next chunk of 128 values given past chunks of 32.** Google Research's bet, formalized in the ICML 2024 paper by Das, Kong, Sen, and Zhou, is that a single model pretrained on ~100 billion real-world time points can forecast *anything* zero-shot, without per-dataset training. It worked: the 200M-parameter v1.0 matched or beat supervised state-of-the-art on Monash, Darts, and ETT benchmarks, and successor versions (2.0, 2.5) briefly held #1 on Salesforce's GIFT-Eval leaderboard. **The model is now embedded in BigQuery ML's `AI.FORECAST` and AlloyDB**, making SQL-level forecasting a one-line operation. But 2026 reality is more nuanced: Amazon's Chronos-2 has overtaken TimesFM on GIFT-Eval, zero-shot TimesFM consistently loses to CatBoost/LightGBM on financial returns, and probabilistic calibration was broken until the 2.5 release in September 2025. This report synthesizes the architecture, paper, benchmarks, applications, and limitations into a decision-ready picture.

## The architecture in one diagram

TimesFM is **a GPT-style decoder-only transformer operating on patches of real-valued time-series data**, directly adapting the LLM recipe to forecasting. A 1D univariate series is chopped into non-overlapping **input patches of 32 contiguous values**. Each patch is fed through a two-layer residual MLP (with a parallel mask vector) that projects it into a 1280-dim token. Causal multi-head self-attention (16 heads × 80-dim per head) then processes the sequence like text tokens, and a separate residual-block output head predicts the **next 128 values** from each token's hidden state.

The key architectural move is **output patch length > input patch length (128 > 32)**. Because the decoder emits 128 future values per autoregressive step, a 512-step horizon requires only 4 forward passes versus 16 for a naive patch-length-32 scheme — drastically cutting error accumulation. The paper's ablation (Figure 3b) shows ETT MAE monotonically drops as output patch grows from 8 → 128. For input patches, **p=32 is the sweet spot**: p=16 ties accuracy but runs 2× slower, while p=64/128 degrades accuracy.

Three generations have shipped:

| Version | Params | Layers | Context | Horizon support | Released | Key change |
|---|---|---|---|---|---|---|
| **TimesFM 1.0** | 200M | 20 | 512 | Any (sinusoidal PE, uncalibrated quantiles) | Feb 2024 | Original paper |
| **TimesFM 2.0** | 500M | 50 | 2,048 | Any; 10 experimental quantile heads | Dec 2024 | Drops positional embedding, uses RoPE; adds LOTSA pretraining data |
| **TimesFM 2.5** | 200M (+30M quantile head) | 20 | **16,384** | Up to 1,000 with calibrated continuous quantile head | Sep 2025 | Smaller + 8× longer context; RevIN normalization; removes frequency token; flip-invariance and positivity flags |

All versions share the same 1280-hidden-dim, 16-head, 32/128 patch configuration; what changes is depth, positional scheme, and training data. **TimesFM 2.5 is the counterintuitive champion**: half the parameters of 2.0 but better accuracy and 8× more context, achieved by scaling data breadth rather than parameters.

## How it was pretrained — the data mix matters more than size

The TimesFM paper trained on a **100B-time-point corpus with an 80/20 real-to-synthetic mix**, balanced across four granularity buckets (sub-hourly+hourly, daily, weekly, monthly). The headline datasets:

- **Wikipedia Pageviews dominate** — ~374B raw points across hourly, daily, weekly, and monthly granularities (5.6M hourly series alone). This is why TimesFM excels at web-traffic-shaped problems.
- **Google Trends** adds ~540M points across 22,435 queries at four frequencies.
- **Public benchmarks folded in**: M4 (23.75M points), Electricity (8.4M), Traffic (15.1M), Weather-Informer (2.2M), Favorita Sales (139M), LibCity (34.3M).
- **3 million synthetic series** (~6.1B points): piecewise-linear trends, ARMA(p,q) with p,q ∈ [1,8], sine/cosine waves — critical for rare granularities. Removing synthetic data barely affects hourly ETT but destroys quarterly/yearly Monash accuracy.

Training objective is **teacher-forced next-patch MSE regression** — the model predicts all future 128-point patches in parallel given teacher-forced history, averaging loss across N decoded patches. A random-prefix masking trick (mask first r ∈ [0, 31] positions) lets a single model handle every context length from 1 to max_context. Training took **2 days on TPUv5e with 16 tensor cores**, batch size 4096, 1.5M iterations, cosine-decayed peak LR of 5e-4. TimesFM 2.0 added a subset of Salesforce's LOTSA corpus (Azure VM traces, London smart meters, Kaggle traffic, ERA5 climate), and 2.5 incorporates Salesforce's GiftEvalPretrain.

**Frequency handling** uses a coarse categorical token {0 = sub-daily, 1 = weekly/monthly, 2 = quarterly/yearly} in versions 1.0 and 2.0 — a free parameter the user can tune, not strictly enforced. TimesFM 2.5 **removed the frequency token entirely**, relying on the model to infer periodicity from context.

## Benchmark results: strong on classical, middle-of-pack in 2026

The paper's zero-shot TimesFM 1.0 (200M) results are genuinely impressive against classical and supervised deep-learning baselines.

**Monash archive** (18 datasets, geometric-mean scaled MAE; naive baseline = 1.00, lower is better):

| Model | GM-scaled MAE | Type |
|---|---|---|
| **TimesFM-ZS (200M)** | **0.6846** | Zero-shot FM |
| N-BEATS | 0.7005 | Supervised |
| FFNN | 0.7044 | Supervised |
| DeepAR | 0.7477 | Supervised |
| CatBoost | 0.7733 | Supervised |
| ARIMA | 0.9449 | Statistical |
| llmtime-ZS (GPT-3) | 0.9715 | LLM prompt |
| PatchTST-ZS (control) | 1.0557 | Zero-shot |
| Naive | 1.0000 | Baseline |

**TimesFM beats llmtime by >25%** and ties supervised N-BEATS while being fully zero-shot. On ETT (Informer benchmark, avg MAE across h=96,192), TimesFM-ZS scores **0.36** versus supervised PatchTST's 0.37 — within statistical significance of the best supervised model, with no target-domain training.

The story changes on the **Salesforce GIFT-Eval leaderboard** (live mirror, April 2026), which tests models on 97 datasets with proper out-of-distribution splits and flags pretraining-data leakage:

| Rank | Model | MASE | CRPS | Leak flag |
|---|---|---|---|---|
| 11 | **Chronos-2 (Amazon)** | 0.70 | 0.49 | No |
| 14 | Timer-S1 | 0.69 | 0.49 | No |
| 16 | **TimesFM-2.5 (200M)** | 0.71 | 0.49 | No |
| 31 | Moirai-2 (Salesforce) | 0.73 | 0.52 | No |
| 38 | TimesFM-2.0-500M | 0.76 | 0.55 | Yes |
| 52 | Chronos-T5-Large (710M) | 0.87 | 0.65 | Yes |
| 58 | TimesFM-1.0-200M | 1.08 | 0.68 | Yes |
| 65 | Auto-ARIMA | 1.07 | 0.91 | No |
| 68 | Seasonal-Naive | 1.00 | 1.00 | No |

Two conclusions stand out. **Among non-leaking pretrained foundation models, TimesFM 2.5 ranks just behind Chronos-2 and ties with IBM Granite PatchTST-FM** — credible but no longer the clear leader. And the aggregated MASE gap between foundation models and classical Auto-ARIMA is roughly 35% (0.71 vs 1.07), which matters for serious production use but is smaller than marketing suggests.

Nixtla's independent **Foundation Time-Series Arena** on 30,000+ unseen series ranks TimesFM **second in accuracy behind their proprietary TimeGPT-1**, ahead of Chronos, with inference speed comparable to TimeGPT and ~5× faster than Chronos. Moirai-MoE from Salesforce claims up to **65× fewer activated parameters** than TimesFM with 17% better accuracy on Monash — a reminder that architecture innovation still trumps scale in this domain.

## Practical implementation — SQL, Python, and fine-tuning

The v1 Python API centers on three classes. A minimal TimesFM 2.0 forecast looks like:

```python
import timesfm, numpy as np
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="gpu", per_core_batch_size=32, horizon_len=128,
        num_layers=50, model_dims=1280, context_len=2048,
        use_positional_embedding=False),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-2.0-500m-pytorch"))
point, quantiles = tfm.forecast(
    inputs=[np.sin(np.linspace(0, 20, 100))],  # list of variable-length 1D arrays
    freq=[0])  # 0=high, 1=mid, 2=low frequency
# point.shape = (N, horizon), quantiles.shape = (N, horizon, 10)
```

The v2.5 API is cleaner, using a `ForecastConfig` and a compiled model:

```python
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
model.compile(timesfm.ForecastConfig(
    max_context=1024, max_horizon=256,
    use_continuous_quantile_head=True,   # calibrated probabilistic output
    force_flip_invariance=True,          # averages f(X) and -f(-X)
    infer_is_positive=True,              # clamps negative outputs
    fix_quantile_crossing=True))
```

For a `pandas` DataFrame in long format (`unique_id`, `ds`, `y`), `tfm.forecast_on_df(df, freq="D", value_name="y", num_jobs=-1)` parallelizes across series automatically. **Covariate support** (static/dynamic, numerical/categorical) is available via the `xreg` extra via `forecast_with_covariates(...)`, which fits a ridge regression alongside the transformer and combines them in either `"xreg + timesfm"` (linear first, residual to TimesFM) or `"timesfm + xreg"` mode.

Beyond Python, **BigQuery ML's `AI.FORECAST`** is by far the smoothest entry point — SQL users get TimesFM for free:

```sql
SELECT * FROM AI.FORECAST(
  TABLE bike_trips,
  data_col => 'num_trips', timestamp_col => 'trip_date',
  id_cols => ['station_id'], horizon => 720, confidence_level => 0.95);
```

Minimum 3 data points; practical minimum ~30. Fine-tuning is supported via LoRA with HuggingFace Transformers + PEFT (official example in `timesfm-forecasting/examples/finetuning/`), plus full fine-tuning scripts in the v1 branch's `finetuning/` and `peft/` directories. The repo is Apache-2.0 licensed with ~10,000 stars, 825 forks, and 19 contributors led by Rajat Sen.

## Where TimesFM shines — and where it breaks

Real-world deployments cluster in predictable territory. **Web-traffic and search-interest forecasting** (Wikipedia pageviews, Google Trends) is TimesFM's pretraining home turf and consistently its strongest domain. **Retail demand forecasting** is a major success: Grid Dynamics benchmarks show ~15% MAE reduction on monthly car-parts and ~25% on daily restaurant visitors versus AutoARIMA; on Rohlik grocery data, TimesFM beat LightGBM, XGBoost, TFT, and NHITS on 5,800 SKUs at 13-week horizons. **Short-term electricity load forecasting** across ~300 European households (arXiv 2410.09487) shows zero-shot TimesFM matching trained-from-scratch PatchTST, especially with 96+ hours of context. **Epidemiology** (hand-foot-and-mouth disease, Korea/Singapore/Chongqing) found TimesFM competitive with LSTMs and tuned ARIMA at longer horizons. Google Cloud's flagship tutorial forecasts 720 hours of SF Bay bike-share ridership from 4 months of history, capturing weekly seasonality with 95% prediction intervals.

**Finance is the canonical failure case.** arXiv 2511.18578 tested TimesFM-500M zero-shot on daily stock excess returns and got **R² = −2.80%**, directional accuracy below 50%, and a −1.47% annualized long-short portfolio return — worse than CatBoost/LightGBM ensembles. For volatility and Value-at-Risk, GARCH and simple econometric baselines beat zero-shot TimesFM; only after continual pretraining on TOPIX500/S&P500 (Preferred Networks, arXiv 2505.11163) did TimesFM statistically outperform traditional models. The distribution mismatch between Wikipedia/Trends/electricity data and heavy-tailed, non-stationary financial returns is fundamental.

Beyond finance, these **specific limitations matter for production use**:

- **Univariate-first architecture**: TimesFM models each series independently. Chronos-2 and Moirai-2 are architecturally better at cross-series and multivariate dependencies; the XReg workaround is a linear tack-on, not a true multivariate attention mechanism.
- **Patch-size floor**: input patch of 32 means context <32 throws shape errors (GitHub issues #175, #231). BigQuery enforces a 3-point minimum but accuracy is poor below ~30.
- **Uncalibrated probability estimates** in v1.0 and v2.0 — explicitly marked "experimental, not calibrated after pretraining." Multiple papers report poor CRPS relative to Chronos, which tokenizes values discretely and is probabilistic by design. TimesFM 2.5's new 30M-parameter continuous quantile head is the first properly calibrated version.
- **No structural-break handling**: COVID-like shocks, product discontinuations, and regime shifts anchor forecasts to the wrong regime — and benchmark studies have flagged pretraining-data leakage from 2020 anomalies specifically.
- **Data-leakage risk in benchmarks**: because M4, Electricity, Traffic, and LOTSA subsets are in the pretraining corpus, some studies show 47-184% benchmark inflation. Users must verify their test data wasn't seen during pretraining.
- **Infrastructure friction**: `lingvo` doesn't support Apple Silicon/ARM; `jaxlib` is required for covariate forecasting; 32+ GB RAM recommended for 2.0; GPU essentially required for reasonable latency. GitHub issue #328 tracks an active flat-line bug on H100s with 2.5 + `torch.compile`.
- **No explainability**: unlike ARIMA_PLUS or Prophet, there's no trend/seasonal/holiday decomposition. Google Cloud's own docs explicitly recommend ARIMA_PLUS when explainability matters.

## What this means — picking the right tool in 2026

**TimesFM has achieved something remarkable: a single pretrained model that forecasts most conventional business time series without training, accessible via a SQL function.** The paper's core claim — that LLM-style decoder-only pretraining transfers to time series — is now validated across independent benchmarks, and the architectural recipe (patched transformer, output patch > input patch, synthetic data for rare frequencies) has become a template for the field.

But the 2026 landscape is pluralistic, not TimesFM-dominated. Choose **TimesFM 2.5** when you need a zero-shot baseline on conventional business data, SQL integration via BigQuery, or open self-hostable weights; it's the most enterprise-ready option. Choose **Chronos-2** when probabilistic calibration is critical and you can tolerate Amazon's ecosystem; it now leads non-leaking GIFT-Eval. Choose **Moirai-2 or TimeGPT** when true multivariate attention matters. Choose **IBM TinyTimeMixer** when compute is constrained — it's 40× smaller and reportedly beats TimesFM on some benchmarks. Choose **statistical ensembles (AutoARIMA + AutoETS + Theta)** when data is well-behaved and explainability matters — Nixtla has shown they beat Chronos-Large by 10% at 1/5 the compute. And for **financial returns, regime-change-heavy series, or very short histories**, don't use any foundation model zero-shot.

The deeper lesson from TimesFM's trajectory is that **time-series foundation models are not yet "GPT moments" but they are real productivity wins for 80% of routine forecasting tasks**. The paper's scaling plot shows monotonic improvement from 17M → 70M → 200M parameters, but the 2.5 release's shrink-while-improving demonstrates that data breadth, context length, and training tricks (RevIN, flip-invariance, continuous quantile heads) now matter more than raw parameter count. Expect the next wave to emphasize multivariate attention, MoE efficiency, and proper probabilistic calibration rather than bigger backbones.

---

## Data tables ready for React charts

**Parameter count vs. GIFT-Eval MASE (non-leaking where possible)** — for a scatter plot:
```json
[
  {"model":"TimesFM-2.5","params_M":200,"mase":0.71,"crps":0.49,"org":"Google"},
  {"model":"TimesFM-2.0","params_M":500,"mase":0.76,"crps":0.55,"org":"Google"},
  {"model":"TimesFM-1.0","params_M":200,"mase":1.08,"crps":0.68,"org":"Google"},
  {"model":"Chronos-2","params_M":120,"mase":0.70,"crps":0.49,"org":"Amazon"},
  {"model":"Chronos-Bolt-Base","params_M":205,"mase":0.81,"crps":0.57,"org":"Amazon"},
  {"model":"Chronos-T5-Large","params_M":710,"mase":0.87,"crps":0.65,"org":"Amazon"},
  {"model":"Chronos-T5-Base","params_M":200,"mase":0.88,"crps":0.65,"org":"Amazon"},
  {"model":"Chronos-T5-Small","params_M":46,"mase":0.89,"crps":0.66,"org":"Amazon"},
  {"model":"Moirai-2.0-R-small","params_M":14,"mase":0.73,"crps":0.52,"org":"Salesforce"},
  {"model":"Moirai-1.1-R-Large","params_M":311,"mase":0.87,"crps":0.60,"org":"Salesforce"},
  {"model":"Moirai-1.1-R-Base","params_M":91,"mase":0.90,"crps":0.61,"org":"Salesforce"},
  {"model":"Moirai-1.1-R-Small","params_M":14,"mase":0.95,"crps":0.65,"org":"Salesforce"},
  {"model":"TiRex","params_M":35,"mase":0.72,"crps":0.49,"org":"NXAI"},
  {"model":"Toto-Open-Base","params_M":151,"mase":0.75,"crps":0.52,"org":"Datadog"},
  {"model":"Lag-Llama","params_M":200,"mase":1.23,"crps":0.88,"org":"ServiceNow"},
  {"model":"TTM-R2","params_M":5,"mase":1.02,"crps":0.87,"org":"IBM"},
  {"model":"PatchTST","params_M":10,"mase":0.85,"crps":0.59,"org":"Princeton"},
  {"model":"iTransformer","params_M":6,"mase":0.89,"crps":0.62,"org":"Tsinghua"},
  {"model":"N-BEATS","params_M":8,"mase":0.94,"crps":0.82,"org":"ServiceNow"},
  {"model":"Auto-ARIMA","params_M":0,"mase":1.07,"crps":0.91,"org":"Classical"},
  {"model":"Seasonal-Naive","params_M":0,"mase":1.00,"crps":1.00,"org":"Classical"}
]
```

**Training data size (billions of time points)** — for a bar chart:
```json
[
  {"model":"GIFT-Eval Pretrain","points_B":240,"org":"Salesforce"},
  {"model":"TimesFM 1.0","points_B":100,"org":"Google"},
  {"model":"TimesFM 2.0","points_B":100,"org":"Google"},
  {"model":"TimeGPT-1","points_B":100,"org":"Nixtla"},
  {"model":"Chronos","points_B":84,"org":"Amazon"},
  {"model":"Moirai LOTSA","points_B":27,"org":"Salesforce"},
  {"model":"MOMENT","points_B":1.13,"org":"CMU"}
]
```

**TimesFM 1.0 paper — Monash geometric-mean scaled MAE** (from Table 4, arXiv 2310.10688):
```json
[
  {"model":"TimesFM-ZS","monash_gm":0.6846},
  {"model":"N-BEATS","monash_gm":0.7005},
  {"model":"FFNN","monash_gm":0.7044},
  {"model":"DeepAR","monash_gm":0.7477},
  {"model":"CatBoost","monash_gm":0.7733},
  {"model":"TBATS","monash_gm":0.7736},
  {"model":"ETS","monash_gm":0.8104},
  {"model":"PR","monash_gm":0.8218},
  {"model":"Transformer","monash_gm":0.8619},
  {"model":"WaveNet","monash_gm":0.9384},
  {"model":"Theta","monash_gm":0.9371},
  {"model":"ARIMA","monash_gm":0.9449},
  {"model":"llmtime-ZS","monash_gm":0.9715},
  {"model":"Naive","monash_gm":1.0000},
  {"model":"PatchTST-ZS","monash_gm":1.0557},
  {"model":"SES","monash_gm":1.0855}
]
```

**ETT zero-shot average MAE** (horizon 96 & 192, Table 5):
```json
[
  {"model":"TimesFM-ZS","ett_avg_mae":0.36},
  {"model":"PatchTST-sup","ett_avg_mae":0.37},
  {"model":"PatchTST-ZS","ett_avg_mae":0.35},
  {"model":"llmtime","ett_avg_mae":0.45},
  {"model":"FEDformer","ett_avg_mae":0.53},
  {"model":"Autoformer","ett_avg_mae":0.53},
  {"model":"Informer","ett_avg_mae":0.99}
]
```

**TimesFM scaling (Monash GM-scaled MAE vs model size)** — from paper Figure 3a:
```json
[
  {"params_M":17,"monash_gm":0.76},
  {"params_M":70,"monash_gm":0.71},
  {"params_M":200,"monash_gm":0.6846}
]
```

**Inference time on Monash (Moirai-MoE paper)** — for comparing FM latency:
```json
[
  {"model":"Moirai-Small (14M)","seconds":264},
  {"model":"Moirai-MoE-Small (11M/117M)","seconds":273},
  {"model":"Moirai-Base (91M)","seconds":358},
  {"model":"Moirai-MoE-Base (86M/935M)","seconds":370},
  {"model":"Moirai-Large (311M)","seconds":537},
  {"model":"Chronos-Small (46M)","seconds":551},
  {"model":"Chronos-Base (200M)","seconds":1177},
  {"model":"Chronos-Large (710M)","seconds":2780}
]
```

**Version evolution** — for a timeline:
```json
[
  {"version":"1.0","params_M":200,"context":512,"layers":20,"released":"2024-02","context_x_params":102400},
  {"version":"2.0","params_M":500,"context":2048,"layers":50,"released":"2024-12","context_x_params":1024000},
  {"version":"2.5","params_M":200,"context":16384,"layers":20,"released":"2025-09","context_x_params":3276800}
]
```

**Use case → industry mapping** — for a grid/heatmap visualization:
```json
[
  {"industry":"Retail","use_case":"SKU demand forecasting","fit":"strong","example":"Rohlik grocery, 5800 SKUs, beats LightGBM/TFT"},
  {"industry":"Web/Media","use_case":"Pageviews & search interest","fit":"native","example":"Pretraining home turf (Wikipedia+Trends)"},
  {"industry":"Energy","use_case":"Short-term load forecasting","fit":"strong","example":"300 EU households, matches trained PatchTST"},
  {"industry":"Transportation","use_case":"Bike/taxi/traffic volume","fit":"strong","example":"SF bike-share 720hr forecast, GCP tutorial"},
  {"industry":"Healthcare","use_case":"Epidemic curve forecasting","fit":"competitive","example":"HFMD Korea/SG/Chongqing, ties tuned ARIMA"},
  {"industry":"Supply chain","use_case":"Inventory/parts planning","fit":"strong","example":"Grid Dynamics: 15% MAE reduction on car-parts"},
  {"industry":"Cloud ops","use_case":"Capacity/anomaly","fit":"good","example":"LOTSA VM traces in v2.0 pretraining"},
  {"industry":"Finance","use_case":"Daily return prediction","fit":"POOR","example":"R²=-2.8%, loses to CatBoost zero-shot"},
  {"industry":"Finance","use_case":"Volatility / VaR","fit":"POOR zero-shot","example":"GARCH wins unless fine-tuned"},
  {"industry":"Short-history","use_case":"<30 data points","fit":"FAIL","example":"Below patch size 32 throws errors"}
]
```

**Key limitations severity** — for a radar/bar chart:
```json
[
  {"limitation":"Multivariate support","severity":8,"note":"Univariate-first; Moirai/Chronos-2 better"},
  {"limitation":"Probabilistic calibration","severity":6,"note":"Fixed in 2.5; broken in 1.0/2.0"},
  {"limitation":"Regime changes","severity":7,"note":"No break detection"},
  {"limitation":"Financial data","severity":9,"note":"Loses to CatBoost zero-shot"},
  {"limitation":"Short histories","severity":7,"note":"Patch-size floor of 32"},
  {"limitation":"Irregular sampling","severity":6,"note":"Linear interp only"},
  {"limitation":"Explainability","severity":7,"note":"No decomposition; use ARIMA_PLUS instead"},
  {"limitation":"Apple Silicon support","severity":5,"note":"lingvo incompatibility"},
  {"limitation":"Benchmark leakage","severity":6,"note":"M4/Electricity in pretraining"}
]
```