using Gec.Core.Models;
using Gec.Tests;

namespace Gec.Tests.Models;

public class MultiHeadAttentionTests
{
    [Test]
    public void Constructor_ThrowsWhenDModelNotDivisibleByHeads()
    {
        Assert.Throws<Exception>(() => new MultiHeadAttention(dModel: 5, nHeads: 2));
    }

    [Test]
    public void Forward_ConcatenatesHeadsInOrderAndAppliesOutputProjection()
    {
        // 2 heads of dHead=2 each. Zero Q/K weights make attention input-independent (uniform
        // averaging), and a constant V bias per head makes each head's output a known constant.
        // An identity output projection then lets us check the concatenation order directly:
        // head0 -> columns [1,1], head1 -> columns [2,2].
        var mha = new MultiHeadAttention(dModel: 4, nHeads: 2);
        var heads = ReflectionTestHelpers.GetField<AttentionHead[]>(mha, "_heads");

        SetConstantHead(heads[0], constant: 1);
        SetConstantHead(heads[1], constant: 2);

        var outputProjection = ReflectionTestHelpers.GetField<Linear>(mha, "_outputProjection");
        ReflectionTestHelpers.SetLinearWeights(
            outputProjection,
            new double[,]
            {
                { 1, 0, 0, 0 },
                { 0, 1, 0, 0 },
                { 0, 0, 1, 0 },
                { 0, 0, 0, 1 }
            },
            new double[4]);

        var input = new double[,] { { 10, -3, 4, 7 }, { -1, 2, 0, 5 }, { 8, 8, -8, -8 } };

        var result = mha.Forward(input);

        for (var i = 0; i < 3; i++)
        {
            Assert.That(result[i, 0], Is.EqualTo(1).Within(1e-9));
            Assert.That(result[i, 1], Is.EqualTo(1).Within(1e-9));
            Assert.That(result[i, 2], Is.EqualTo(2).Within(1e-9));
            Assert.That(result[i, 3], Is.EqualTo(2).Within(1e-9));
        }
    }

    [Test]
    public void Forward_OutputShapeIsSeqLenByDModel()
    {
        var mha = new MultiHeadAttention(dModel: 6, nHeads: 3);

        var result = mha.Forward(new double[4, 6]);

        Assert.That(result.GetLength(0), Is.EqualTo(4));
        Assert.That(result.GetLength(1), Is.EqualTo(6));
    }

    private static void SetConstantHead(AttentionHead head, double constant)
    {
        var query = ReflectionTestHelpers.GetField<Linear>(head, "_query");
        var key = ReflectionTestHelpers.GetField<Linear>(head, "_key");
        var value = ReflectionTestHelpers.GetField<Linear>(head, "_value");

        ReflectionTestHelpers.ZeroOutLinear(query);
        ReflectionTestHelpers.ZeroOutLinear(key);
        ReflectionTestHelpers.ZeroOutLinear(value);
        ReflectionTestHelpers.SetField(value, "_bias", new[] { constant, constant });
    }
}
