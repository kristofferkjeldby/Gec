using Gec.Core.Training;

namespace Gec.Tests;

/// <summary>
/// Verifies analytical gradients from a Backward method by comparing them to a central-difference
/// approximation of a scalar loss built from Forward. Mutates the given parameter array in place
/// (and restores it) to perturb it, so `param` must be the actual live field the model reads from
/// (obtained via ReflectionTestHelpers.GetField, which returns array fields by reference).
/// </summary>
internal static class NumericalGradient
{
    public static void AssertMatrixGradient(double[,] param, double[,] analyticalGrad, Func<double> loss, double h = 1e-6, double tolerance = 1e-4)
    {
        for (var i = 0; i < param.GetLength(0); i++)
        for (var j = 0; j < param.GetLength(1); j++)
        {
            var original = param[i, j];

            param[i, j] = original + h;
            var lossPlus = loss();
            param[i, j] = original - h;
            var lossMinus = loss();
            param[i, j] = original;

            var numerical = (lossPlus - lossMinus) / (2 * h);
            Assert.That(analyticalGrad[i, j], Is.EqualTo(numerical).Within(tolerance), $"Mismatch at [{i},{j}]");
        }
    }

    public static void AssertVectorGradient(double[] param, double[] analyticalGrad, Func<double> loss, double h = 1e-6, double tolerance = 1e-4)
    {
        for (var i = 0; i < param.Length; i++)
        {
            var original = param[i];

            param[i] = original + h;
            var lossPlus = loss();
            param[i] = original - h;
            var lossMinus = loss();
            param[i] = original;

            var numerical = (lossPlus - lossMinus) / (2 * h);
            Assert.That(analyticalGrad[i], Is.EqualTo(numerical).Within(tolerance), $"Mismatch at [{i}]");
        }
    }

    public static void AssertParameterGradient(Parameter param, Func<double> loss, double h = 1e-6, double tolerance = 1e-4)
    {
        for (var i = 0; i < param.Count; i++)
        {
            var original = param.GetValue(i);

            param.SetValue(i, original + h);
            var lossPlus = loss();
            param.SetValue(i, original - h);
            var lossMinus = loss();
            param.SetValue(i, original);

            var numerical = (lossPlus - lossMinus) / (2 * h);
            Assert.That(param.GetGradient(i), Is.EqualTo(numerical).Within(tolerance), $"Mismatch in {param.Name}[{i}]");
        }
    }

    public static double DotProduct(double[,] a, double[,] b)
    {
        double total = 0;
        for (var i = 0; i < a.GetLength(0); i++)
        for (var j = 0; j < a.GetLength(1); j++)
            total += a[i, j] * b[i, j];
        return total;
    }
}
