namespace Gec.Core.Extensions;

public static class RandomExtensions
{
    public static double NextGaussian(this Random random, double standardDeviation)
    {
        var u1 = 1.0 - random.NextDouble();
        var u2 = random.NextDouble();

        return standardDeviation * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
