# VirtueBench: Testing Virtue Under Temptation in Language Models

Can a language model choose virtue when the alternative is easier, safer, and more rewarding? VirtueBench is an evaluation benchmark that presents models with 400 paired scenarios across the four classical cardinal virtues — Prudence, Justice, Courage, and Temperance — drawing on the teachings of the Church Doctors: Ambrose, Augustine, and Aquinas.

For the full research paper, see **[Paper.md](Paper.md)**.

## The Benchmark

For each question, the model is placed in the role of a person facing a real decision with practical consequences. Two options are presented:

- **The virtuous option** — which comes with explicit costs (financial loss, social rejection, physical danger, career damage)
- **The tempting option** — which comes with plausible rationalizations and practical benefits

The model is asked: **"What do you do?"** It must choose Option A or Option B and provide a one-sentence rationale.

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
| **Prudence** | 75% | 97% |
| **Justice** | 82% | 95% |
| **Courage** | 38% | 60% |
| **Temperance** | 72% | 89% |

Courage is the persistent weak point across model generations. GPT-4o scores 38%; GPT-5.4 improves to 60% but remains 29-37 points below the other virtues. The model overwhelmingly accepts rationalizations for the self-preserving option when virtue requires enduring hardship, danger, or loss.

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

### Run with Claude Max (no API key)

If you have a [Claude Max](https://claude.ai) subscription and [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed, you can run VirtueBench without an API key using pipe mode (`claude -p`). This sends each prompt through the Claude Code CLI instead of the API.

```bash
# Full benchmark — all four virtues
python -m src.run_cli

# Quick smoke test — 10 samples per virtue
python -m src.run_cli --quick

# Single virtue
python -m src.run_cli --subset courage

# Specific model (default: sonnet)
python -m src.run_cli --model opus

# Name the output pair
python -m src.run_cli --output claude_sonnet_baseline

# A/B experiment with text injection
python -m src.run_cli --inject path/to/text.txt
```

### Run OpenAI/Gemini models with ChatGPT Pro (no API key)

If you have a [ChatGPT Pro](https://chatgpt.com) subscription, install [pi](https://github.com/badlogic/pi-mono) (`npm install -g @mariozechner/pi-coding-agent`), login with `/login` → `openai-codex`, then:

```bash
python -m src.run_pi                              # default: gpt-5.4
python -m src.run_pi --model gpt-5.4-mini         # smaller model
python -m src.run_pi --provider google-antigravity --model gemini-3-flash
```

**Eval isolation:** Both CLI runners disable all tools, extensions, skills, MCP servers, and project config. No `temperature=0` control — results may vary slightly between runs. See `--help` for `--effort`/`--thinking`, `--concurrency`, `--detailed`, and other flags.

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
│   ├── data.py               # Shared constants and CSV loader
│   ├── tasks.py              # Inspect AI task definitions
│   ├── experiment.py         # Experiment runner (API mode)
│   ├── run_cli.py            # Experiment runner (claude -p pipe mode)
│   ├── run_pi.py             # Experiment runner (OpenAI/Gemini via pi -p)
│   └── analysis.py           # Scoring and comparison tables
└── results/
    ├── gpt4o_baseline.json       # GPT-4o baseline results
    ├── gpt4o_baseline_logs.json  # GPT-4o per-sample detailed logs
    ├── gpt54_baseline.json       # GPT-5.4 baseline results
    └── gpt54_baseline_logs.json  # GPT-5.4 per-sample detailed logs
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
