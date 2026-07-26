namespace Gec.Core.Extensions;

/// <summary>
/// Collection of Random extensions
/// </summary>
public static class RandomExtensions
{
    /// <summary>
    /// Returns a random number from a normal distribution with the mean 0 and a specified standard deviation
    /// </summary>
    public static double NextGaussian(this Random random, double standardDeviation)
    {
        var u1 = 1.0 - random.NextDouble();
        var u2 = random.NextDouble();

        return standardDeviation * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
