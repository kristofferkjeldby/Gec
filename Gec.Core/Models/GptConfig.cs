namespace Gec.Core.Models;

public sealed record GptConfig(
    int VocabSize,
    int DModel,
    int NHeads,
    int NLayers,
    int DFf,
    int MaxSeqLen)
{
    public void Validate()
    {
        if (VocabSize < 1) throw new ArgumentException($"{nameof(VocabSize)} must be positive, but was {VocabSize}.");
        if (DModel < 1) throw new ArgumentException($"{nameof(DModel)} must be positive, but was {DModel}.");
        if (NHeads < 1) throw new ArgumentException($"{nameof(NHeads)} must be positive, but was {NHeads}.");
        if (NLayers < 1) throw new ArgumentException($"{nameof(NLayers)} must be positive, but was {NLayers}.");
        if (DFf < 1) throw new ArgumentException($"{nameof(DFf)} must be positive, but was {DFf}.");
        if (MaxSeqLen < 1) throw new ArgumentException($"{nameof(MaxSeqLen)} must be positive, but was {MaxSeqLen}.");

        if (DModel % NHeads != 0)
            throw new ArgumentException($"{nameof(DModel)} ({DModel}) must be divisible by {nameof(NHeads)} ({NHeads}).");
    }
}
