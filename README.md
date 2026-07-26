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
# put any plain-text corpus at Gec.Console/Data/corpus.txt
dotnet run --project Gec.Console -- vocab                        # corpus.txt  -> vocab.json + merges.txt
dotnet run --project Gec.Console -- tokenize                     # corpus.txt  -> tokens.txt
dotnet run --project Gec.Console -c Release -- train             # tokens.txt  -> train/val + model.json
dotnet run --project Gec.Console -c Release -- complete "Once upon a time "
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
Corpus     .../Gec.Console/Data/corpus.txt (8028411 characters)
Training   500 merges...
Vocabulary 565 tokens (65 characters + 500 merges)
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
Corpus     .../Gec.Console/Data/corpus.txt (8028411 characters)
Vocabulary .../Gec.Console/Data/vocab.json (565 tokens, 500 merges)
Encoded    2791025 tokens (2,88 characters per token)
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

Each batch item is backpropagated through its own `GptModel` replica, up to `Workers` replicas running
at once via [`ParallelTrainer`](Gec.Core/Training/ParallelTrainer.cs). Gradients from every replica are
summed into replica 0 before the optimizer step, then the updated weights are broadcast back out —
mathematically identical to a sequential loop over the batch, just parallelized across cores.

```bash
dotnet run --project Gec.Console -c Release -- train
dotnet run --project Gec.Console -c Release -- train "" "" "" 300   # fewer steps
```

```
Tokens     .../Data/tokens.txt (2791025 tokens, 2511922 train / 279103 validation)
Written    .../Data/train.txt
Written    .../Data/val.txt
Model      4 layers, dModel 96, 6 heads, vocab 565 (581.173 parameters)
Training   3300 steps, batch 10, learning rate 0,003

  step   100/3300   train loss 5,2546   validation loss 5,2656   37,2s
  step  1000/3300   train loss 3,5246   validation loss 3,6404   502,3s
  step  2000/3300   train loss 2,8693   validation loss 2,8781   1155,7s
  step  3300/3300   train loss 2,4189   validation loss 2,4217   1952,9s

Written    .../Data/model.json (12.351 KB)
```

At the current defaults, a full run takes on the order of half an hour. Train and validation loss stay
close together, which is the sign that the model is learning the grammar rather than memorizing the
corpus.

### complete

```
complete <prompt> [max-tokens] [model] [vocab-directory]
```

Encodes the prompt, then extends it one token at a time by sampling from the model's next-token
distribution. Contexts longer than `MaxSeqLen` slide, so only the most recent window is fed back in.

```bash
dotnet run --project Gec.Console -c Release -- complete "Once upon a time "
dotnet run --project Gec.Console -c Release -- complete "The little fox " 120
```

```
Model      .../Data/model.json (4 layers, dModel 96, vocab 565)
Prompt     Once upon a time
Generated  80 tokens at temperature 0,8

Once upon a time to eatch together.

"Will you come back?" asked Ellie.

"Yes!" said Pip.

Pip Maya sat down. He kept out her own her wings. She saw the ducks song of tree. He looked very quiet.

Nextraw a soft shell. "Is this your familyear
```

The model has picked up story structure it never saw verbatim — titles, quoted dialogue with `said`/`asked`
tags, character names, paragraph breaks — well before the underlying grammar and spelling are reliable.
At 581K parameters and a half-hour of training this is expected; the shapes are right, the fine-grained
correctness is not yet.

## The corpus

`corpus.txt` is a plain UTF-8 text file — drop in any prose and the rest of the pipeline adapts to it,
since vocabulary size, token count, and model size all derive from whatever is in this file. The
default corpus is currently ~8 MB of short, simple children's stories, generated up front as training
data: simple sentences and everyday vocabulary keep the language uniform enough for a small model to
learn from, while still being real, varied English rather than a fixed grammar.

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
so a `[565, 96]` embedding reads as one line per token.

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
