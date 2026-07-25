using Gec.Core.Common;
using Gec.Core.Extensions;
using Gec.Core.Training;

namespace Gec.Core.Models;

public class MultiHeadAttention
{
    private readonly AttentionHead[] _heads;
    private readonly Linear _outputProjection;
    private readonly int _dHead;

    public MultiHeadAttention(int dModel, int nHeads, bool causal = false, Random? random = null, string name = "attention")
    {
        if (dModel % nHeads != 0)
            throw new Exception("dModel must be divisible by nHeads");

        _dHead = dModel / nHeads;
        _heads = new AttentionHead[nHeads];

        for (int i = 0; i < nHeads; i++)
        {
            _heads[i] = new AttentionHead(dModel, _dHead, causal, random, $"{name}.head{i}");
        }

        _outputProjection = new Linear(dModel, dModel, random, $"{name}.projection");
    }

    public IEnumerable<Parameter> Parameters() =>
        _heads.SelectMany(head => head.Parameters()).Concat(_outputProjection.Parameters());

    public double[,] Forward(double[,] input) // input shape: [seqLen, dModel]
    {
        var results = _heads.Select(h => h.Forward(input)).ToArray();
        var concatenated = Matrix.ConcatColumns(results);

        return _outputProjection.Forward(concatenated);
    }

    public double[,] Backpropagate(double[,] gradOutput)
    {
        var gradConcatenated = _outputProjection.Backpropagate(gradOutput);

        double[,]? gradInput = null;

        for (var h = 0; h < _heads.Length; h++)
        {
            var gradHead = ColumnBlock(gradConcatenated, h * _dHead, _dHead);
            var gradHeadInput = _heads[h].Backpropagate(gradHead);

            gradInput = gradInput is null ? gradHeadInput : Matrix.Matadd(gradInput, gradHeadInput);
        }

        return gradInput!;
    }

    private static double[,] ColumnBlock(double[,] matrix, int firstColumn, int width)
    {
        var rows = matrix.Rows();
        var block = new double[rows, width];

        for (var i = 0; i < rows; i++)
        for (var j = 0; j < width; j++)
            block[i, j] = matrix[i, firstColumn + j];

        return block;
    }
}
