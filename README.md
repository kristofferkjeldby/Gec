# Gec

A simple transformer, implemented from scratch in C# to learn the math behind it. No ML framework
underneath — matrix multiplication, softmax, layer normalization, attention, and GELU are all
plain C# over `double[,]`, with both forward and backward passes hand-derived and verified.

## Status

Implemented, with forward and backward passes:

- `Matmul`, `Softmax`, `Gelu`, and general matrix ops (`Matrix`, `MatrixExtensions`)
- `Linear`
- `LayerNorm`
- `AttentionHead` / `MultiHeadAttention`
- `Mlp`

`TransformerBlock` (wiring the above into a full block with residual connections) is currently
disabled — see [`Gec.Core/Models/TransformerBlock.cs`](Gec.Core/Models/TransformerBlock.cs) — while
it's updated for `LayerNorm`'s current matrix-based API. No training loop / optimizer yet.

## Project layout

- **Gec.Core** — the library: matrix primitives (`Common/`, `Extensions/`) and model layers (`Models/`).
- **Gec.Tests** — NUnit test suite.

## Building and testing

Requires the [.NET 10 SDK](https://dotnet.microsoft.com/download).

```bash
dotnet test
```

## Testing approach

Forward passes are checked against hand-computed values where practical. Backward passes are
verified with numerical gradient checking: a scalar loss is built from `Forward`, each parameter is
perturbed by a small `h`, and the resulting central-difference approximation is compared against
the analytical gradient from `Backward` (see `Gec.Tests/NumericalGradient.cs`).
