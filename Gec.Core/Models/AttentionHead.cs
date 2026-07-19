using Gec.Core.Common;
using Gec.Core.Extensions;

namespace Gec.Core.Models;

public class AttentionHead
{
    private readonly Linear _query;
    private readonly Linear _key;
    private readonly Linear _value;
    private readonly int _dHead;
    
    // Populated by Forward and read by Backward; only valid after a Forward call.
    private double[,] _q = null!;
    private double[,] _k = null!;
    private double[,] _v = null!;
    private double[,] _scores = null!;
    private double[,] _weights = null!;

    public AttentionHead(int dModel, int dHead)
    {
        _query = new Linear(dModel, dHead);
        _key = new Linear(dModel, dHead);
        _value = new Linear(dModel, dHead);
        _dHead = dHead;
    }

    public double[,] Forward(double[,] input) // input shape: [seqLen, dModel]
    {
        _q = _query.Forward(input);
        _k = _key.Forward(input);
        _v = _value.Forward(input);

        _scores = Matmul.Forward(_q, _k.Transpose());
        var scaledScores = Matrix.ApplyElement(_scores, x => x / Math.Sqrt(_dHead));
        _weights = Matrix.ApplyRows(scaledScores, Softmax.Forward);

        return Matmul.Forward(_weights, _v);
    }
    
    public (double[,] gradInput, double[,] gradWq, double[] gradBq, double[,] gradWk, double[] gradBk, double[,] gradWv, double[] gradBv)
        Backward(double[,] output, double[,] gradOutput)
    {
        var d = 1.0 / Math.Sqrt(_dHead);
        
        var (gradWeights, gradV) = Matmul.Backward(_weights, _v, gradOutput);
        var gradScaledScores = Matrix.ApplyRowsBackward(_weights, gradWeights, Softmax.Backward);
        var gradScores = Matrix.ApplyElementBackward(_scores, gradScaledScores, x => d);
        
        var (gradQ, gradKt) = Matmul.Backward(_q, _k.Transpose(), gradScores);
        var gradK = gradKt.Transpose();

        var (gradInputQ, gradWq, gradBq) = _query.Backward(output, gradQ);
        var (gradInputK, gradWk, gradBk) = _key.Backward(output, gradK);
        var (gradInputV, gradWv, gradBv) = _value.Backward(output, gradV);

        var gradInput = Matrix.Matadd(Matrix.Matadd(gradInputQ, gradInputK), gradInputV);

        return (gradInput, gradWq, gradBq, gradWk, gradBk, gradWv, gradBv);
    }
}