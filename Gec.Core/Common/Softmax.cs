namespace Gec.Core.Common;

public static class Softmax
{
    public static double[] Forward(double[] input)
    {
        var output = new double[input.Length];
        Forward((ReadOnlySpan<double>)input, output);
        return output;
    }

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
