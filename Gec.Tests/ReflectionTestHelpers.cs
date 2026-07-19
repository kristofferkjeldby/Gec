using System.Reflection;
using Gec.Core.Models;

namespace Gec.Tests;

/// <summary>
/// Weight initialization in Linear is random with no way to inject values, so these helpers
/// reach into private fields to set up deterministic scenarios and to read out internals to
/// build independently-computed expected values for composition tests.
/// </summary>
internal static class ReflectionTestHelpers
{
    public static T GetField<T>(object target, string fieldName)
    {
        var field = target.GetType().GetField(fieldName, BindingFlags.NonPublic | BindingFlags.Instance)
                    ?? throw new InvalidOperationException($"Field '{fieldName}' not found on {target.GetType().Name}");
        return (T)field.GetValue(target)!;
    }

    public static void SetField(object target, string fieldName, object value)
    {
        var field = target.GetType().GetField(fieldName, BindingFlags.NonPublic | BindingFlags.Instance)
                    ?? throw new InvalidOperationException($"Field '{fieldName}' not found on {target.GetType().Name}");
        field.SetValue(target, value);
    }

    public static void SetLinearWeights(Linear linear, double[,] weights, double[] bias)
    {
        SetField(linear, "_weights", weights);
        SetField(linear, "_bias", bias);
    }

    public static void ZeroOutLinear(Linear linear)
    {
        var weights = GetField<double[,]>(linear, "_weights");
        var bias = GetField<double[]>(linear, "_bias");
        SetLinearWeights(linear, new double[weights.GetLength(0), weights.GetLength(1)], new double[bias.Length]);
    }

    public static void ZeroOutAttentionHead(AttentionHead head)
    {
        ZeroOutLinear(GetField<Linear>(head, "_query"));
        ZeroOutLinear(GetField<Linear>(head, "_key"));
        ZeroOutLinear(GetField<Linear>(head, "_value"));
    }

    public static void ZeroOutMultiHeadAttention(MultiHeadAttention mha)
    {
        var heads = GetField<AttentionHead[]>(mha, "_heads");
        foreach (var head in heads)
            ZeroOutAttentionHead(head);

        ZeroOutLinear(GetField<Linear>(mha, "_outputProjection"));
    }

    public static void ZeroOutMlp(Mlp mlp)
    {
        ZeroOutLinear(GetField<Linear>(mlp, "_up"));
        ZeroOutLinear(GetField<Linear>(mlp, "_down"));
    }
}
