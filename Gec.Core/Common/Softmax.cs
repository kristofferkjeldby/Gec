namespace Gec.Core.Common;

/// <summary>
/// Softmax function.
/// </summary>
public static class Softmax
{
    /// <summary>
    /// Applies the softmax function to the input array.
    /// </summary>
    public static double[] Forward(double[] input)
    {
        var output = new double[input.Length];
        Forward((ReadOnlySpan<double>)input, output);
        return output;
    }

    /// <summary>
    /// Applies the softmax function to the input span and stores the result in the output span.
    /// The softmax function takes an input vector and turns it into a probability distribution where the probabilities of each value
    /// are proportional to the exponential of the input values. It does this by first subtracting the maximum value from each input
    /// value. This is not part of the softmax function itself, but but prevents overflow and even e^30 is over a billion.
    /// So by subtracting the maximum value, we ensure that the largest value is 0 (and e^0 = 1).
    /// Then the actual softmax, all elements are exponentiated and then divided by the sum of all exponentials.
    /// This ensures that the output values are in the range (0, 1) and sum to 1.
    /// </summary>
    public static void Forward(ReadOnlySpan<double> input, Span<double> output)
    {
        var max = double.NegativeInfinity;
        for (var i = 0; i < input.Length; i++)
            if (input[i] > max)
                max = input[i];

        double sum = 0;
        for (var i = 0; i < input.Length; i++)
        {
            var e = Math.Exp(input[i] - max);
            output[i] = e;
            sum += e;
        }

        for (var i = 0; i < output.Length; i++)
            output[i] /= sum;
    }

    public static double[] Backward(double[] output, double[] gradOutput)
    {
        var gradInput = new double[output.Length];
        Backward((ReadOnlySpan<double>)output, gradOutput, gradInput);
        return gradInput;
    }

    public static void Backward(ReadOnlySpan<double> output, ReadOnlySpan<double> gradOutput, Span<double> gradInput)
    {
        double dot = 0;
        for (var i = 0; i < output.Length; i++)
            dot += output[i] * gradOutput[i];

        for (var i = 0; i < output.Length; i++)
            gradInput[i] = output[i] * (gradOutput[i] - dot);
    }
}
