using Gec.Core.Common;
using Gec.Core.Models;
using Gec.Tests;

namespace Gec.Tests.Models;

public class MlpTests
{
    [Test]
    public void Forward_MatchesManualUpGeluDownComposition()
    {
        var mlp = new Mlp(dModel: 2, dFf: 3);

        var upWeights = new double[,] { { 1, 0, -1 }, { 0, 1, 1 } };
        var upBias = new double[] { 0, 0, 0 };
        var downWeights = new double[,] { { 1, 1 }, { 0, 1 }, { 1, 0 } };
        var downBias = new double[] { 5, -5 };

        ReflectionTestHelpers.SetLinearWeights(ReflectionTestHelpers.GetField<Linear>(mlp, "_up"), upWeights, upBias);
        ReflectionTestHelpers.SetLinearWeights(ReflectionTestHelpers.GetField<Linear>(mlp, "_down"), downWeights, downBias);

        var input = new double[,] { { 1, 2 }, { 3, -1 } };

        var result = mlp.Forward(input);

        // Independently compose the same primitives (already unit-tested elsewhere) to
        // derive the expected result, verifying Mlp wires up -> gelu -> down correctly.
        var hidden = Matmul.Forward(input, upWeights);
        var activated = Matrix.ApplyElement(hidden, Gelu.GeluApproxForward);
        var expected = Matmul.Forward(activated, downWeights);
        for (var i = 0; i < expected.GetLength(0); i++)
        for (var j = 0; j < expected.GetLength(1); j++)
            expected[i, j] += downBias[j];

        for (var i = 0; i < 2; i++)
        for (var j = 0; j < 2; j++)
            Assert.That(result[i, j], Is.EqualTo(expected[i, j]).Within(1e-9), $"Mismatch at [{i},{j}]");
    }

    [Test]
    public void Forward_ZeroWeights_ReturnsBiasOnly()
    {
        var mlp = new Mlp(dModel: 2, dFf: 3);
        ReflectionTestHelpers.ZeroOutMlp(mlp);

        var down = ReflectionTestHelpers.GetField<Linear>(mlp, "_down");
        ReflectionTestHelpers.SetLinearWeights(
            down,
            ReflectionTestHelpers.GetField<double[,]>(down, "_weights"),
            new double[] { 2, -2 });

        var result = mlp.Forward(new double[,] { { 1, 2 }, { 3, 4 } });

        for (var i = 0; i < 2; i++)
        {
            Assert.That(result[i, 0], Is.EqualTo(2).Within(1e-9));
            Assert.That(result[i, 1], Is.EqualTo(-2).Within(1e-9));
        }
    }

    [Test]
    public void Forward_OutputShapeMatchesSeqLenByDModel()
    {
        var mlp = new Mlp(dModel: 4, dFf: 8);

        var result = mlp.Forward(new double[3, 4]);

        Assert.That(result.GetLength(0), Is.EqualTo(3));
        Assert.That(result.GetLength(1), Is.EqualTo(4));
    }

    [Test]
    public void Backward_MatchesNumericalGradientOfForward()
    {
        var mlp = new Mlp(dModel: 2, dFf: 3);

        var up = ReflectionTestHelpers.GetField<Linear>(mlp, "_up");
        var down = ReflectionTestHelpers.GetField<Linear>(mlp, "_down");
        ReflectionTestHelpers.SetLinearWeights(
            up,
            new double[,] { { 0.5, -0.3, 0.2 }, { 0.1, 0.4, -0.6 } },
            new double[] { 0.05, -0.02, 0.1 });
        ReflectionTestHelpers.SetLinearWeights(
            down,
            new double[,] { { 0.3, -0.1 }, { -0.4, 0.2 }, { 0.5, 0.15 } },
            new double[] { -0.1, 0.05 });

        var input = new double[,] { { 1, -0.5 }, { 0.3, 2 } };
        var gradOutput = new double[,] { { 1, -1 }, { 0.5, 0.2 } };

        mlp.Forward(input);
        var (gradInput, gradWUp, gradBUp, gradWDown, gradBDown) = mlp.Backward(input, gradOutput);

        double Loss(double[,] x) => NumericalGradient.DotProduct(mlp.Forward(x), gradOutput);

        NumericalGradient.AssertMatrixGradient(input, gradInput, () => Loss(input));

        var upWeights = ReflectionTestHelpers.GetField<double[,]>(up, "_weights");
        var upBias = ReflectionTestHelpers.GetField<double[]>(up, "_bias");
        var downWeights = ReflectionTestHelpers.GetField<double[,]>(down, "_weights");
        var downBias = ReflectionTestHelpers.GetField<double[]>(down, "_bias");

        NumericalGradient.AssertMatrixGradient(upWeights, gradWUp, () => Loss(input));
        NumericalGradient.AssertVectorGradient(upBias, gradBUp, () => Loss(input));
        NumericalGradient.AssertMatrixGradient(downWeights, gradWDown, () => Loss(input));
        NumericalGradient.AssertVectorGradient(downBias, gradBDown, () => Loss(input));
    }
}
