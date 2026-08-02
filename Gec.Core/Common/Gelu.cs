namespace Gec.Core.Common;

/// <summary>
/// GELU (Gaussian Error Linear Unit) activation function.
/// </summary>
public static class Gelu
{
    // Precomputed constant used in the tanh approximation below.
    private static readonly double SqrtTwoOverPi = Math.Sqrt(2/Math.PI);

    /// <summary>
    /// The GELU activation function is an activation function used in neural networks. An activation function determines the strength
    /// of an output signal based on the input signal. The GELU function suppresses small negative values and allows positive values to pass through,
    /// which can help the network learn complex patterns in the data. The original GELU definition uses the Gauss error function (erf), 
    /// which is not available in C#. However, it can be approximated using the tanh function, which is what we do here.
    /// The approximation is based on the paper "Gaussian Error Linear Units (GELUs)" by Hendrycks and Gimpel (2016).
    /// </summary>
    public static double GeluApproxForward(double x)
    {
        var u = SqrtTwoOverPi * (x + 0.044715 * x * x * x);

        return 0.5 * x * (1 + Math.Tanh(u));
    }

    /// <summary>
    /// The derivative of the GELU activation function is used during backpropagation to update the weights of the neural network.
    /// Given a value x, the derivative of the GELU function tells us the slope of the GELU function at that point.
    /// Again, we use a tanh approximation for computational efficiency.
    /// </summary>
    public static double GeluApproxDerivative(double x)
    {
        var u = SqrtTwoOverPi * (x + 0.044715 * x * x * x);
        var tanhU = Math.Tanh(u);
        var duDx = SqrtTwoOverPi * (1 + 0.134145 * x * x);

        return 0.5 * (1 + tanhU) + 0.5 * x * (1 - tanhU * tanhU) * duDx;
    }
}