/*

namespace Gec.Models;

public class TransformerBlock
{
    private readonly LayerNorm _ln1;
    private readonly MultiHeadAttention _attention;
    private readonly LayerNorm _ln2;
    private readonly Mlp _mlp;

    public TransformerBlock(int dModel, int nHeads, int dFf)
    {
        _ln1 = new LayerNorm(dModel);
        _attention = new MultiHeadAttention(dModel, nHeads);
        _ln2 = new LayerNorm(dModel);
        _mlp = new Mlp(dModel, dFf);
    }

    public double[,] Forward(double[,] input) // input shape: [seqLen, dModel]
    {
        var normalized1 = Matrix.ApplyRows(input, _ln1.Forward);
        var result1 = _attention.Forward(normalized1);
        var added1 = Matrix.Matadd(result1, input);
        
        var normalized2 = Matrix.ApplyRows(added1, _ln2.Forward);
        var result2 = _mlp.Forward(normalized2);
        var added2 = Matrix.Matadd(result2, added1);
        
        return added2;
    }
}

*/