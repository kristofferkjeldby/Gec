using Gec.Core.Common;

namespace Gec.Core.Models;

public class MultiHeadAttention
{
    private readonly AttentionHead[] _heads;
    private readonly Linear _outputProjection;

    public MultiHeadAttention(int dModel, int nHeads)
    {
        if (dModel % nHeads != 0)
            throw new Exception("dModel must be divisible by nHeads");

        var dHead = dModel / nHeads;
        _heads = new AttentionHead[nHeads];
        
        for (int i = 0; i < nHeads; i++)
        {
            _heads[i] = new AttentionHead(dModel, dHead);
        }

        _outputProjection = new Linear(dModel, dModel);
    }

    public double[,] Forward(double[,] input) // input shape: [seqLen, dModel]
    {
        var results = _heads.Select(h => h.Forward(input)).ToArray();
        var concatenated = Matrix.ConcatColumns(results);

        return _outputProjection.Forward(concatenated);
    }
}