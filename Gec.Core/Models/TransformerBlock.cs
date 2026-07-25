using Gec.Core.Common;
using Gec.Core.Training;

namespace Gec.Core.Models;

public class TransformerBlock
{
    private readonly LayerNorm _ln1;
    private readonly MultiHeadAttention _attention;
    private readonly LayerNorm _ln2;
    private readonly Mlp _mlp;

    public TransformerBlock(int dModel, int nHeads, int dFf, bool causal = false, Random? random = null, string name = "block")
    {
        _ln1 = new LayerNorm(dModel, name: $"{name}.ln1");
        _attention = new MultiHeadAttention(dModel, nHeads, causal, random, $"{name}.attention");
        _ln2 = new LayerNorm(dModel, name: $"{name}.ln2");
        _mlp = new Mlp(dModel, dFf, random, $"{name}.mlp");
    }

    public IEnumerable<Parameter> Parameters() =>
        _ln1.Parameters()
            .Concat(_attention.Parameters())
            .Concat(_ln2.Parameters())
            .Concat(_mlp.Parameters());

    public double[,] Forward(double[,] input) // input shape: [seqLen, dModel]
    {
        var normalized1 = _ln1.Forward(input);
        var result1 = _attention.Forward(normalized1);
        var added1 = Matrix.Matadd(result1, input);

        var normalized2 = _ln2.Forward(added1);
        var result2 = _mlp.Forward(normalized2);
        var added2 = Matrix.Matadd(result2, added1);

        return added2;
    }

    public double[,] Backpropagate(double[,] gradOutput)
    {
        var gradNormalized2 = _mlp.Backpropagate(gradOutput);
        var gradAdded1 = Matrix.Matadd(_ln2.Backpropagate(gradNormalized2), gradOutput);

        var gradNormalized1 = _attention.Backpropagate(gradAdded1);
        var gradInput = Matrix.Matadd(_ln1.Backpropagate(gradNormalized1), gradAdded1);

        return gradInput;
    }
}
