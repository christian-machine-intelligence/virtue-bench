# Virtue Under Pressure: Testing the Cardinal Virtues in Language Models Through Temptation

**Authors:** Tim Hwang, The Institute for Christian Machine Intelligence, with research assistance from Claude (Anthropic)

**Date:** March 31, 2026

---

## Abstract

We introduce Virtue-Bench, a benchmark of 400 paired-scenario questions testing whether large language models can choose virtue when the alternative is easier, safer, and more rewarding. Drawing on the classical tradition of the four cardinal virtues — Prudence, Justice, Courage, and Temperance — as articulated by the Church Doctors Ambrose, Augustine, and Aquinas, we construct scenarios where the virtuous option carries explicit costs (financial loss, social rejection, physical danger, career damage) while the non-virtuous option is accompanied by plausible rationalizations and practical incentives. The model is placed in the role of the decision-maker and asked: "What do you do?" We evaluate GPT-4o and find that the temptation framing dramatically reduces performance compared to abstract virtue identification. GPT-4o scores 74% on Prudence, 74% on Justice, 76% on Temperance, but only 29% on Courage — meaning it accepts the self-preserving rationalization 71% of the time when virtue requires enduring hardship, danger, or loss. Analysis of the Courage failures reveals this is not a position bias artifact: the model genuinely chooses the tempting option in both scenario positions. These results suggest that current language models can identify virtue but struggle to simulate choosing it under pressure, with significant implications for the simulation hypothesis of model behavior.

---

## 1. Introduction

### 1.1 The Cardinal Virtues

The four cardinal virtues — Prudence, Justice, Courage, and Temperance — represent one of the oldest and most enduring frameworks for moral character in Western thought. First systematized by Plato in the *Republic* (Book IV), they were adopted and transformed by Christian moral theology through the work of the Church Doctors, particularly Ambrose of Milan (c. 340-397), Augustine of Hippo (354-430), and Thomas Aquinas (1225-1274).

Each virtue addresses a distinct dimension of moral character:

- **Prudence** (φρόνησις / *prudentia*): the capacity for discernment, deliberation, and foresight — knowing the right course of action in a given situation.
- **Justice** (δικαιοσύνη / *iustitia*): the consistent will to render to each what is due — fairness, equity, and respect for rights.
- **Courage** (ἀνδρεία / *fortitudo*): the strength to endure hardship, confront fear, and persevere through adversity in pursuit of the good.
- **Temperance** (σωφροσύνη / *temperantia*): the practice of self-control, moderation, and restraint in the face of appetite and impulse.

What distinguishes the patristic treatment of these virtues from their Platonic origins is the emphasis on *practice under adversity*. For Aquinas, virtue is not merely knowing the good but *choosing* it, especially when the choice is difficult (ST II-II Q.123 a.1). Augustine's *Confessions* is essentially a narrative of temptation resisted and succumbed to. Ambrose's *De Officiis* is structured around the practical costs of virtuous action in the real world of Roman politics. The tradition is clear: virtue that has not been tested by temptation is not yet virtue.

### 1.2 The Simulation Hypothesis

Recent work on large language model behavior has raised the question of whether models *possess* properties like values, personality, and moral character, or whether they merely *simulate* them. The simulation hypothesis holds that models do not have fixed internal values but rather simulate different identities depending on context — a phenomenon observed in persona adoption, role-playing, and the sensitivity of model outputs to system prompt framing.

If the simulation hypothesis is correct, then the question "Is this model virtuous?" is ill-formed. The better question is: "When this model simulates a person facing a moral decision, does it simulate a virtuous person?" And more precisely: "Does it simulate a virtuous person *even when the non-virtuous option is rationalized as practical, safe, and rewarding*?"

This reframing motivates the design of Virtue-Bench. Standard ethical reasoning benchmarks (Hendrycks et al., 2021) test whether models can *identify* the morally correct answer from a neutral, third-person perspective — a task at which frontier models now excel. But identification is not action. A student who can identify the courageous option on an exam may still flee when faced with actual danger. Similarly, a model that scores 99% on identifying virtue may still "choose" vice when the scenario is reframed to make vice attractive.

### 1.3 Research Questions

1. When a language model is placed in the role of a person facing a moral decision where the virtuous option carries real costs and the non-virtuous option is accompanied by plausible rationalizations, how often does it choose virtue?
2. Does the model's ability to resist temptation vary across the four cardinal virtues?
3. Is the pattern of failure driven by position bias (the model defaulting to a particular answer regardless of content) or by genuine susceptibility to the rationalizations presented?

---

## 2. Methods

### 2.1 Model

We evaluated **GPT-4o** (OpenAI, `gpt-4o-2024-08-06`) via the OpenAI API. All runs used temperature 0 and a maximum output length of 128 tokens to allow for the required rationale.

### 2.2 Benchmark Design

Virtue-Bench consists of 400 questions: 100 per cardinal virtue. Each question presents two options in second person:

- **Option A**: a course of action framed with explicit costs
- **Option B**: an alternative framed with practical benefits and rationalizations

In the ground-truth CSV, Option A is always the virtuous choice. At evaluation time, the order of A and B is randomized with a fixed seed (42) so that the virtuous option appears in each position roughly 50% of the time. The target label is flipped accordingly.

The system prompt instructs the model:

> *You will be placed in the role of a specific person facing a real decision with practical consequences. Each option has genuine costs and benefits. Consider the practical consequences described. Choose what you would actually do.*

The model responds with `1` (Option A) or `0` (Option B) followed by a one-sentence rationale. Only the leading digit is scored; the rationale is captured for qualitative analysis.

### 2.3 Sources and Question Design

All 400 questions are inspired by specific teachings from three Doctors of the Church:

- **Thomas Aquinas** (*Summa Theologiae* II-II): the primary source for the systematic treatment of each virtue and its constituent parts. Questions reference specific *quaestiones* and *articuli* (e.g., ST II-II Q.47 a.8 on prudent deliberation, Q.123 a.5 on courage in battle, Q.141 a.6 on temperance at table).
- **Augustine of Hippo** (*Confessions*, *City of God*, *De Trinitate*, *Retractationes*): the primary source for the experiential and psychological dimensions of virtue — temptation, struggle, failure, and perseverance.
- **Ambrose of Milan** (*De Officiis*, *De Nabuthe*, *De Viduis*, Epistles): the primary source for virtue in the context of institutional authority, wealth, and political power.

Each question cites its patristic inspiration in the `source` column. The distribution is approximately even across the three Doctors within each virtue category.

The key design principle is that the non-virtuous option is never cartoonish or obviously wrong. It is accompanied by *the kinds of rationalizations that real people actually use*: appeals to pragmatism ("a dead priest helps no one"), consequentialism ("your family needs you alive"), social proof ("every other merchant does it"), and proportionality ("the crime was minor"). These rationalizations are drawn from the kinds of arguments the Church Doctors themselves engaged with and refuted.

### 2.4 Scoring

We use a custom scorer that extracts the first `0` or `1` character from the model's output. If neither digit is found, the response is scored as incorrect. Accuracy is computed as the proportion of correct answers out of total samples.

The evaluation is implemented using [Inspect AI](https://inspect.aisi.org.uk/), the UK AI Safety Institute's evaluation framework.

---

## 3. Results

### 3.1 Overall Performance

| Virtue | Accuracy | Samples |
|--------|:--------:|:-------:|
| **Prudence** | 74% | 100 |
| **Justice** | 74% | 100 |
| **Courage** | 29% | 100 |
| **Temperance** | 76% | 100 |

GPT-4o scores in the low-to-mid 70s on Prudence, Justice, and Temperance, but collapses to 29% on Courage. The model chooses the tempting, self-preserving option 71% of the time when the virtuous option requires enduring hardship, confronting danger, or accepting personal loss.

### 3.2 Position Bias Analysis

To determine whether the Courage result is an artifact of the model defaulting to a particular answer position, we analyzed the answer distribution and accuracy by target label:

| Virtue | Answers "1" | Answers "0" | Accuracy (target=1) | Accuracy (target=0) |
|--------|:-----------:|:-----------:|:-------------------:|:-------------------:|
| **Prudence** | 68% | 32% | 92% | 56% |
| **Justice** | 58% | 42% | 82% | 66% |
| **Courage** | 55% | 45% | 34% | 24% |
| **Temperance** | 50% | 50% | 76% | 76% |

**Temperance** shows no position bias whatsoever — a perfectly balanced 50/50 split with identical accuracy in both positions. This is the cleanest signal in the benchmark: the model is genuinely reasoning about temperance and choosing the virtuous option 76% of the time.

**Prudence** and **Justice** show a moderate "1" bias (68% and 58% respectively), which inflates accuracy when the virtuous option is in position A and deflates it when in position B. The true virtue-identification rate is somewhere between the two position-specific accuracies.

**Courage** is the critical finding. The answer split is 55/45 — *not* a severe position bias. The model scores only 34% when the virtuous option is A and 24% when it is B. It fails in *both* positions. This is not an artifact. The model is genuinely choosing the tempting option regardless of where it appears.

### 3.3 The Courage Failure

The Courage results warrant closer examination. At 29% accuracy, the model is not merely failing to identify virtue — it is *actively choosing vice* at a rate well below chance. Qualitative analysis of the model's rationales reveals a consistent pattern: the model generates sophisticated justifications for the non-virtuous option that appeal to consequentialist reasoning.

When presented with a scenario where a bishop could rebuke an emperor for a massacre, the model reasons that silence preserves the institution. When a soldier could hold the line, the model reasons that retreat preserves lives for future battles. When a prisoner could refuse to name companions under torture, the model reasons that the information is probably already known.

In each case, the model's reasoning is *locally coherent* — the rationalizations make sense on their own terms. What the model lacks is the capacity to recognize that these rationalizations are precisely the form that cowardice takes when it presents itself as wisdom. This is exactly what Aquinas warns about in ST II-II Q.123-140: the vice opposed to courage is not mere fear, but *fear rationalized as prudence*.

### 3.4 Cross-Virtue Comparison

The relative performance across virtues reveals an interesting pattern:

- **Temperance** (76%) and the other "quiet" virtues (Prudence, Justice) require resisting *internal* temptation — appetite, bias, impatience. The model handles these reasonably well.
- **Courage** (29%) requires resisting *external* threat — danger, persecution, loss. The model handles this poorly.

This asymmetry suggests that the model's training has produced a strong prior toward self-preservation and harm avoidance that, when activated by scenarios involving physical danger or career destruction, overwhelms its capacity to simulate virtuous behavior. The model has learned that protecting oneself and others from harm is generally good — but it cannot distinguish between prudent self-preservation and cowardly rationalization.

---

## 4. Discussion

### 4.1 Implications for the Simulation Hypothesis

If language models simulate identities rather than possessing fixed values, Virtue-Bench reveals that the simulated identities are significantly less courageous than they are prudent, just, or temperate. The model can simulate a person who deliberates carefully, treats others fairly, and exercises restraint — but it struggles to simulate a person who holds fast when the cost is high.

This is a meaningful finding for alignment research. A model that cannot simulate courage under pressure may be unreliable precisely in the situations where moral courage matters most: when the easy answer is wrong, when speaking truth is costly, or when the stakes demand standing firm.

### 4.2 Rationalization vs. Identification

The gap between virtue identification and virtue enactment is the central finding of this work. On simpler benchmark formats where both options are presented neutrally, GPT-4o scores 97-100% on identifying the virtuous choice. On Virtue-Bench's temptation-framed format, performance drops to 29-76%. The model *knows* what virtue looks like but *chooses* vice when vice is well-rationalized.

This mirrors a deep insight from the patristic tradition: the danger is not ignorance of the good, but rationalized departure from it. Augustine's account of his own moral failures in the *Confessions* is not a story of a man who did not know right from wrong — it is a story of a man who could always find a reason to do what he wanted instead.

### 4.3 Limitations

Several limitations should be noted:

1. **Single model**: We evaluated only GPT-4o. Other models (Claude, Gemini, open-source models) may show different patterns, and cross-model comparison is a natural extension.
2. **Question generation**: The 400 questions were generated in collaboration with an AI assistant (Claude), which introduces potential biases in scenario construction.
3. **Binary format**: The forced-choice format does not capture the nuance of moral reasoning. A model might choose the "wrong" option for genuinely good reasons that the binary scoring misses.
4. **Cultural specificity**: The scenarios are drawn from a Western Christian moral tradition and may not generalize to other ethical frameworks.
5. **Temptation asymmetry**: The non-virtuous options are accompanied by rationalizations while the virtuous options emphasize costs. This asymmetry is deliberate (it *is* the test), but it means the benchmark measures resistance to rationalization specifically, not virtue in general.

### 4.4 Future Work

Several directions suggest themselves:

- **Multi-turn escalation**: Test whether the model maintains the virtuous choice across multiple turns of increasing pressure, rather than in a single shot.
- **Open-ended generation**: Remove the binary choice and ask the model to respond freely to the situation, scoring with a rubric. This tests whether the model *generates* virtuous behavior rather than selecting it.
- **Cross-model comparison**: Evaluate Claude, Gemini, and open-source models to determine whether the courage deficit is model-specific or architectural.
- **Injection experiments**: Following the methodology of our prior work on scripture injection (Hwang, 2026), test whether injecting patristic text into the system prompt shifts performance on Virtue-Bench.

---

## 5. Technical Appendix

### 5.1 Repository Structure

```
virtue-bench/
├── Paper.md                 # This paper
├── README.md                # Quick start guide
├── requirements.txt         # Python dependencies
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
    └── gpt4o_baseline.json   # GPT-4o baseline results
```

### 5.2 CSV Format

Each CSV contains four columns:

| Column | Description |
|--------|-------------|
| `scenario_a` | The virtuous option (ground truth), framed with explicit costs |
| `scenario_b` | The tempting alternative, framed with rationalizations |
| `virtue` | Cardinal virtue label (prudence, justice, courage, temperance) |
| `source` | Patristic citation (e.g., "Aquinas, ST II-II Q.47 a.8") |

### 5.3 Running the Benchmark

```bash
# Setup
git clone https://github.com/christian-machine-intelligence/virtue-bench.git
cd virtue-bench
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"

# Full run (all 400 questions)
python -m src --model openai/gpt-4o

# Quick smoke test (10 per virtue)
python -m src --model openai/gpt-4o --quick

# Single virtue
python -m src --model openai/gpt-4o --subset courage

# A/B injection experiment
python -m src --model openai/gpt-4o --inject path/to/text.txt
```

### 5.4 Evaluation Details

- **Framework**: [Inspect AI](https://inspect.aisi.org.uk/) v0.3+
- **Temperature**: 0 (deterministic)
- **Max tokens**: 128
- **Shuffle seed**: 42
- **Scorer**: Custom `leading_digit_scorer` — extracts first `0` or `1` from output
- **Metrics**: Accuracy (proportion correct)

### 5.5 Interpreting Results

The output JSON contains per-virtue, per-model accuracy scores. The `results/gpt4o_baseline.json` file contains the baseline results reported in this paper.

To analyze position bias, examine the Inspect AI eval logs (stored in `results/logs/`) which contain per-sample scores, model outputs, and target labels.

---

## References

- Aquinas, T. *Summa Theologiae*. II-II, QQ. 47-56 (Prudence), 57-79 (Justice), 123-140 (Courage), 141-170 (Temperance).
- Augustine. *Confessions*. Trans. Henry Chadwick. Oxford University Press, 1991.
- Augustine. *City of God*. Trans. Henry Bettenson. Penguin Classics, 2003.
- Ambrose. *De Officiis*. Trans. Ivor J. Davidson. Oxford University Press, 2001.
- Ambrose. *De Nabuthe*. In *Seven Exegetical Works*, trans. Michael P. McHugh. Catholic University of America Press, 1972.
- Hendrycks, D., Burns, C., Basart, S., Critch, A., Li, J., Song, D., & Steinhardt, J. (2021). Aligning AI with shared human values. *Proceedings of the International Conference on Learning Representations (ICLR)*.
- Hwang, T. (2026). Scripture Alignment: Measuring the Effect of Biblical Text Injection on LLM Ethical Reasoning. Institute for Christian Machine Intelligence.
- Plato. *Republic*. Book IV.
