using Gec.Core.Common;

namespace Gec.Core.Training;

public static class CrossEntropy
{
    public static (double Loss, double[,] GradLogits) Forward(double[,] logits, int[] targets)
    {
        var rows = logits.GetLength(0);
        var cols = logits.GetLength(1);

        if (targets.Length != rows)
            throw new ArgumentException($"Expected {rows} targets to match the logit rows, but got {targets.Length}.", nameof(targets));

        var probabilities = Matrix.ApplyRows(logits, Softmax.Forward);
        var gradLogits = new double[rows, cols];

        double loss = 0;

        for (var r = 0; r < rows; r++)
        {
            var target = targets[r];
            if (target < 0 || target >= cols)
                throw new ArgumentOutOfRangeException(nameof(targets), $"Target {target} at row {r} is outside the vocabulary of {cols} tokens.");

            loss -= Math.Log(Math.Max(probabilities[r, target], double.Epsilon));

            for (var c = 0; c < cols; c++)
                gradLogits[r, c] = (probabilities[r, c] - (c == target ? 1 : 0)) / rows;
        }

        return (loss / rows, gradLogits);
    }
}
