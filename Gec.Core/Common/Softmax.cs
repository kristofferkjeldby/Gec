namespace Gec.Core.Common;

public static class Softmax
{
    public static double[] Forward(double[] input)
    {
        var max = input.Max();
        var sum = input.Sum(e => Math.Exp(e - max));
        return input.Select(x => Math.Exp(x - max) / sum).ToArray();
    }
    
    public static double[] Backward(double[] output, double[] gradOutput)
    {
        var dot = output.Select((e, i) => e * gradOutput[i]).Sum();
        return output.Select((e, i) => e * (gradOutput[i] - dot)).ToArray();
    }
}