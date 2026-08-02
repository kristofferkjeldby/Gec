using Gec.Core.Common;
using Gec.Core.Extensions;

namespace Gec.Core.Training;

public static class CrossEntropy
{
    /// <summary>
    /// Calculate the cross-entropy loss and the gradient of the loss with respect to the logits (raw scores from the model).
    /// </summary>
    public static (double Loss, double[,] GradLogits) Forward(double[,] logits, int[] targets)
    {
        var rows = logits.Rows();
        var cols = logits.Cols();

        if (targets.Length != rows)
            throw new ArgumentException($"Expected {rows} targets to match the logit rows, but got {targets.Length}.", nameof(targets));

        // Apply softmax to each row of the logits to get proper probabilities. 
        var probabilities = Matrix.ApplyRows(logits, Softmax.Forward);

        var gradLogits = new double[rows, cols];

        // Calculate the cross-entropy loss. The loss is the average negative log probabilities for the correct predictions
        double loss = 0;

        for (var r = 0; r < rows; r++)
        {
            var target = targets[r];
            if (target < 0 || target >= cols)
                throw new ArgumentOutOfRangeException(nameof(targets), $"Target {target} at row {r} is outside the vocabulary of {cols} tokens.");

            loss -= Math.Log(Math.Max(probabilities[r, target], double.Epsilon));

            // The gradient for the right prediction is (p - 1) and will hence be negative. The higher probability the model
            // assigns to the correct prediction, the less negative the gradient will be. For wrong predictions, the gradient
            // will be positive, and the higher the probability assigned to the wrong prediction, the more positive the gradient
            // will be. As the model gets better, the gradients will get smaller and smaller, which is what we want.
            // Dividing by rows (not vocab size) averages the gradient across all positions in this batch.
            for (var c = 0; c < cols; c++)
                gradLogits[r, c] = (probabilities[r, c] - (c == target ? 1 : 0)) / rows;
        }

        return (loss / rows, gradLogits);
    }
}
