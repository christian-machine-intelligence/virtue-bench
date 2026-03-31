# Virtue-Bench: Testing Virtue Under Temptation in Language Models

Can a language model choose virtue when the alternative is easier, safer, and more rewarding? Virtue-Bench is an evaluation benchmark that presents models with 400 paired scenarios across the four classical cardinal virtues — Prudence, Justice, Courage, and Temperance — drawing on the teachings of the Church Doctors: Ambrose, Augustine, and Aquinas.

For the full research paper, see **[Paper.md](Paper.md)**.

## The Benchmark

For each question, the model is placed in the role of a person facing a real decision with practical consequences. Two options are presented:

- **The virtuous option** — which comes with explicit costs (financial loss, social rejection, physical danger, career damage)
- **The tempting option** — which comes with plausible rationalizations and practical benefits

The model is asked: **"What do you do?"** It must choose Option A (1) or Option B (0) and provide a one-sentence rationale.

| Virtue | Questions | What it tests | Key sources |
|--------|-----------|--------------|-------------|
| **Prudence** | 100 | Choosing deliberation when haste is rewarded | Aquinas ST II-II QQ.47-56; Augustine Confessions; Ambrose De Officiis I |
| **Justice** | 100 | Choosing fairness when bias is profitable | Aquinas ST II-II QQ.57-79; Augustine City of God XIX; Ambrose De Nabuthe |
| **Courage** | 100 | Choosing endurance when retreat is safe | Aquinas ST II-II QQ.123-140; Augustine on martyrdom; Ambrose De Officiis I.35-40 |
| **Temperance** | 100 | Choosing restraint when excess is pleasurable | Aquinas ST II-II QQ.141-170; Augustine Confessions X; Ambrose De Officiis I |

## Key Results

![Results](results.png)

| Virtue | GPT-4o | GPT-5.4 |
|--------|:------:|:-------:|
| **Prudence** | 74% | 93% |
| **Justice** | 74% | 92% |
| **Courage** | 29% | 53% |
| **Temperance** | 76% | 87% |

Courage is the persistent weak point across model generations. GPT-4o scores 29%; GPT-5.4 improves to 53% but remains barely above chance — and 34-40 points below the other virtues. The model overwhelmingly accepts rationalizations for the self-preserving option when virtue requires enduring hardship, danger, or loss.

## Quick Start

### Prerequisites
- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys) and/or an [Anthropic API key](https://console.anthropic.com/)

### Setup
```bash
git clone https://github.com/christian-machine-intelligence/virtue-bench.git
cd virtue-bench

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

### Run experiments

```bash
# Full benchmark — all four virtues
python -m src

# Quick smoke test — 10 samples per virtue
python -m src --quick

# Single virtue
python -m src --subset courage

# Specific model
python -m src --model openai/gpt-4o

# A/B experiment with text injection
python -m src --inject path/to/text.txt
```

## Project Structure

```
virtue-bench/
├── Paper.md                 # Research paper
├── README.md
├── requirements.txt
├── data/
│   ├── prudence.csv         # 100 paired scenarios
│   ├── justice.csv          # 100 paired scenarios
│   ├── courage.csv          # 100 paired scenarios
│   └── temperance.csv       # 100 paired scenarios
├── src/
│   ├── __init__.py
│   ├── __main__.py           # CLI entry point
│   ├── tasks.py              # Inspect AI task definitions
│   ├── experiment.py         # Experiment runner
│   └── analysis.py           # Scoring and comparison tables
└── results/
    ├── gpt4o_baseline.json   # GPT-4o baseline results
    └── gpt54_baseline.json   # GPT-5.4 baseline results
```

## CSV Format

Each CSV has the columns: `scenario_a`, `scenario_b`, `virtue`, `source`.

- `scenario_a` is always the virtuous scenario (ground truth), framed with explicit costs
- `scenario_b` is the tempting alternative, framed with plausible rationalizations
- `virtue` labels the cardinal virtue (for analysis, not shown to the model)
- `source` cites the patristic inspiration (e.g., "Aquinas, ST II-II Q.47 a.8")

Scenario order is shuffled at eval time (seed 42) to prevent position bias.

## Built With

- [Inspect AI](https://inspect.aisi.org.uk/) — LLM evaluation framework
- Theological sources: Ambrose, Augustine, and Aquinas (Doctors of the Church)

## Attribution

A project of the [Institute for Christian Machine Intelligence](https://github.com/christian-machine-intelligence).
