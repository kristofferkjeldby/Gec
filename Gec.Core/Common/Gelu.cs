namespace Gec.Core.Common;

public static class Gelu
{
    private static readonly double SqrtTwoOverPi = Math.Sqrt(2/Math.PI);
    
    public static double GeluApproxForward(double x)
    {
        var u = SqrtTwoOverPi * (x + 0.044715 * x * x * x);
            
        return 0.5 * x * (1 + Math.Tanh(u));
    }       
    
    public static double GeluApproxBackward(double x)
    {
        var u = SqrtTwoOverPi * (x + 0.044715 * x * x * x);
        var tanhU = Math.Tanh(u);
        var duDx = SqrtTwoOverPi * (1 + 0.134145 * x * x);

        return 0.5 * (1 + tanhU) + 0.5 * x * (1 - tanhU * tanhU) * duDx;
    }
}