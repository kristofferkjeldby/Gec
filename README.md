# Gec

A small GPT, implemented from scratch in C# to learn the math behind it. No ML framework
underneath — matrix multiplication, softmax, layer normalization, attention, GELU, cross entropy and
Adam are all plain C# over `double[,]`, with both forward and backward passes hand-derived and
verified against numerical gradients.

The pipeline goes from a raw text file to generated text in four commands: train a byte-pair
vocabulary, encode the corpus to token ids, train the model, then sample from it.

## Quick start

Requires the [.NET 10 SDK](https://dotnet.microsoft.com/download).

```bash
python3 generate_corpus.py                                       # writes Gec.Console/Data/corpus.txt
dotnet run --project Gec.Console -- vocab                        # corpus.txt  -> vocab.json + merges.txt
dotnet run --project Gec.Console -- tokenize                     # corpus.txt  -> tokens.txt
dotnet run --project Gec.Console -c Release -- train             # tokens.txt  -> train/val + model.json
dotnet run --project Gec.Console -c Release -- complete "The small cat "
```

Use `-c Release` for `train` and `complete`; a debug build is several times slower.

## Commands

Every argument is optional except the completion prompt. An omitted or empty argument falls back to
the repository layout, so the commands chain with no arguments at all. Running with no command
prints the usage summary.

### vocab

```
vocab [corpus] [merges] [output-directory]
```

Learns a byte-pair-encoding vocabulary from a corpus and writes `vocab.json` and `merges.txt`.
Merging stops early once no adjacent pair repeats, so the learned count can come out below the
request.

```bash
dotnet run --project Gec.Console -- vocab
dotnet run --project Gec.Console -- vocab Gec.Console/Data/corpus.txt 500
```

```
Corpus     .../Gec.Console/Data/corpus.txt (250009 characters)
Training   500 merges...
Vocabulary 543 tokens (43 characters + 500 merges)
Written    .../Gec.Console/Data/vocab.json
Written    .../Gec.Console/Data/merges.txt
```

### tokenize

```
tokenize [corpus] [vocab-directory] [output]
```

Encodes a corpus into token ids using an existing vocabulary.

```bash
dotnet run --project Gec.Console -- tokenize
```

```
Corpus     .../Gec.Console/Data/corpus.txt (250009 characters)
Vocabulary .../Gec.Console/Data/vocab.json (543 tokens, 500 merges)
Encoded    53356 tokens (4,69 characters per token)
Written    .../Gec.Console/Data/tokens.txt
```

### train

```
train [tokens] [vocab-directory] [model] [steps]
```

Splits the token stream into `train.txt` and `val.txt`, trains the model, and saves it as JSON. The
split is by position rather than random, so validation text is never trained on and re-running gives
the same split. Architecture and hyperparameters come from
[`Settings`](Gec.Console/Configuration/Settings.cs); the vocabulary size comes from `vocab.json`.

```bash
dotnet run --project Gec.Console -c Release -- train
dotnet run --project Gec.Console -c Release -- train "" "" "" 300   # fewer steps
```

```
Tokens     .../Data/tokens.txt (53356 tokens, 48020 train / 5336 validation)
Written    .../Data/train.txt
Written    .../Data/val.txt
Model      2 layers, dModel 64, 4 heads, vocab 543 (172.191 parameters)
Training   1150 steps, batch 8, learning rate 0,003

  step    50/1150   train loss 4,1251   validation loss 4,1054   5,0s
  step   500/1150   train loss 2,0263   validation loss 2,2108   49,8s
  step  1150/1150   train loss 1,9971   validation loss 2,0905   115,3s

Written    .../Data/model.json (3.718 KB)
```

The defaults are tuned for roughly a two-minute run. Train and validation loss stay close together,
which is the sign that the model is learning the grammar rather than memorizing the corpus.

### complete

```
complete <prompt> [max-tokens] [model] [vocab-directory]
```

Encodes the prompt, then extends it one token at a time by sampling from the model's next-token
distribution. Contexts longer than `MaxSeqLen` slide, so only the most recent window is fed back in.

```bash
dotnet run --project Gec.Console -c Release -- complete "The small cat "
dotnet run --project Gec.Console -c Release -- complete "A tired farmer " 120
```

```
Model      .../Data/model.json (2 layers, dModel 64, vocab 543)
Prompt     The small cat
Generated  60 tokens at temperature 0,8

The small cat counts the candles before dinner.
Tobias dreams sometimes along the forest.
The black singer likes to drink juice.
The busy crow keeps flowers in the harbour.
A white crow listens and shares cheese.
The grey sailor paints the small boats in the winter.
```

None of those sentences appear in the corpus. The model has picked up the sentence shapes, the
article and agreement rules, and which objects each verb takes — `drink juice`, `paints the boats`,
`counts the candles` — rather than reproducing lines it saw.

## The corpus

`generate_corpus.py` writes ~250 KB from a slot grammar: a fixed set of sentence shapes filled from
word lists, with correct articles, subject-verb agreement, and verbs restricted to objects they can
plausibly take. The number of possible sentences is far larger than the corpus, so memorizing is not
a shortcut, and generation quality is easy to judge by eye. It is seeded, so the same corpus comes
out every time.

## File formats

Deliberately plain, so every stage is inspectable.

| File | Format |
| --- | --- |
| `corpus.txt` | Plain text. |
| `vocab.json` | JSON object, one `"token": id` entry per line. |
| `merges.txt` | A `#version` header, then one space-separated `A B` merge per line, in learned order. |
| `tokens.txt`, `train.txt`, `val.txt` | Whitespace-separated decimal token ids. Line breaks are only for readability — read by splitting on any whitespace. |
| `model.json` | A `config` object plus a `parameters` array of `{name, rows, cols, values}`, values row-major. |

`config` alone determines the architecture, so loading rebuilds the model from it and fills the
parameters in order, checking each name and shape as it goes. Parameter names are hierarchical
(`block0.mlp.up.weights`), and `values` is written one matrix row per line (wrapped at 32 numbers),
so a `[543, 64]` embedding reads as one line per token.

## Project layout

- **Gec.Core** — the model: matrix primitives (`Common/`, `Extensions/`), layers (`Models/`),
  optimizer and loss (`Training/`), and JSON persistence (`Serialization/`).
- **Gec.Tokenizer** — byte-pair encoding: trainer, encoder, and the vocabulary/merge file readers.
- **Gec.Console** — the four commands, with paths and hyperparameters in `Configuration/Settings.cs`.
- **Gec.Tests** — NUnit test suite.

## Architecture

A decoder-only transformer, assembled from parts that each have a hand-derived backward pass:

- token and position `Embedding`
- `NLayers` × `TransformerBlock` — pre-norm, so each sublayer is `x + f(norm(x))`
- `MultiHeadAttention` over causal `AttentionHead`s, then `Mlp` with GELU
- a final `LayerNorm` and a `Linear` head projecting to vocabulary logits

Attention is causal: position `i` may only attend to positions `<= i`. `AttentionHead` defaults to
plain bidirectional attention and is given `causal: true` by `TransformerBlock`, so a head used on
its own is not masked unless asked.

Modules expose two backward paths. `Backward` returns gradients as a tuple and is what the numerical
gradient tests pin. `Backpropagate` accumulates into `Parameter` objects and returns only the input
gradient, which is what lets blocks chain without threading tuples upward. A `Parameter` wraps a
module's live array by reference, so the optimizer writes straight into the model.

## Building and testing

```bash
dotnet test
```

Forward passes are checked against hand-computed values where practical. Backward passes are
verified with numerical gradient checking: a scalar loss is built from `Forward`, each parameter is
perturbed by a small `h`, and the resulting central-difference approximation is compared against the
analytical gradient (see `Gec.Tests/Core/NumericalGradient.cs`). `GptModelTests` runs that check
across every parameter of a small end-to-end model, and separately asserts that appending tokens
never changes earlier logits — the property that would silently break if the causal mask were wrong.
