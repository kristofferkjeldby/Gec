using Gec.Core.Common;
using Gec.Core.Extensions;
using Gec.Core.Training;

namespace Gec.Core.Models;

public class AttentionHead
{
    private readonly Linear _query;
    private readonly Linear _key;
    private readonly Linear _value;
    private readonly int _dHead;
    private readonly bool _causal;

    // Populated by Forward and read by Backward; only valid after a Forward call.
    private double[,] _q = null!;
    private double[,] _k = null!;
    private double[,] _v = null!;
    private double[,] _scores = null!;
    private double[,] _weights = null!;

    public AttentionHead(int dModel, int dHead, bool causal = false, Random? random = null, string name = "head")
    {
        _query = new Linear(dModel, dHead, random, $"{name}.query");
        _key = new Linear(dModel, dHead, random, $"{name}.key");
        _value = new Linear(dModel, dHead, random, $"{name}.value");
        _dHead = dHead;
        _causal = causal;
    }

    public IEnumerable<Parameter> Parameters() =>
        _query.Parameters().Concat(_key.Parameters()).Concat(_value.Parameters());

    public double[,] Forward(double[,] input) // input shape: [seqLen, dModel]
    {
        _q = _query.Forward(input);
        _k = _key.Forward(input);
        _v = _value.Forward(input);

        _scores = Matmul.Forward(_q, _k.Transpose());
        var scaledScores = Matrix.ApplyElement(_scores, x => x / Math.Sqrt(_dHead));

        if (_causal)
            MaskFuturePositions(scaledScores);

        _weights = Matrix.ApplyRows(scaledScores, Softmax.Forward);

        return Matmul.Forward(_weights, _v);
    }

    public (double[,] gradInput, double[,] gradWq, double[] gradBq, double[,] gradWk, double[] gradBk, double[,] gradWv, double[] gradBv)
        Backward(double[,] input, double[,] gradOutput)
    {
        var (gradQ, gradK, gradV) = BackwardToProjections(gradOutput);

        var (gradInputQ, gradWq, gradBq) = _query.Backward(input, gradQ);
        var (gradInputK, gradWk, gradBk) = _key.Backward(input, gradK);
        var (gradInputV, gradWv, gradBv) = _value.Backward(input, gradV);

        var gradInput = Matrix.Matadd(Matrix.Matadd(gradInputQ, gradInputK), gradInputV);

        return (gradInput, gradWq, gradBq, gradWk, gradBk, gradWv, gradBv);
    }

    public double[,] Backpropagate(double[,] gradOutput)
    {
        var (gradQ, gradK, gradV) = BackwardToProjections(gradOutput);

        var gradInputQ = _query.Backpropagate(gradQ);
        var gradInputK = _key.Backpropagate(gradK);
        var gradInputV = _value.Backpropagate(gradV);

        return Matrix.Matadd(Matrix.Matadd(gradInputQ, gradInputK), gradInputV);
    }

    private (double[,] gradQ, double[,] gradK, double[,] gradV) BackwardToProjections(double[,] gradOutput)
    {
        var d = 1.0 / Math.Sqrt(_dHead);

        var (gradWeights, gradV) = Matmul.Backward(_weights, _v, gradOutput);
        var gradScaledScores = Matrix.ApplyRowsBackward(_weights, gradWeights, Softmax.Backward);
        var gradScores = Matrix.ApplyElementBackward(_scores, gradScaledScores, _ => d);

        var (gradQ, gradKt) = Matmul.Backward(_q, _k.Transpose(), gradScores);

        return (gradQ, gradKt.Transpose(), gradV);
    }

    private static void MaskFuturePositions(double[,] scores)
    {
        var rows = scores.Rows();
        var cols = scores.Cols();

        for (var i = 0; i < rows; i++)
        for (var j = i + 1; j < cols; j++)
            scores[i, j] = double.NegativeInfinity;
    }
}
