using Gec.Core.Models;
using Gec.Core.Training;

namespace Gec.Tests.Core.Training;

public class AdamTests
{
    [Test]
    public void Step_MovesParametersDownhill()
    {
        var value = new[] { 1.0, -2.0 };
        var parameter = new VectorParameter("p", value);
        var optimizer = new Adam([parameter], learningRate: 0.1);

        parameter.AddGradient([1.0, -1.0]);
        optimizer.Step();

        Assert.That(value[0], Is.LessThan(1.0));
        Assert.That(value[1], Is.GreaterThan(-2.0));
    }

    [Test]
    public void Step_FirstStepIsBiasCorrectedToTheFullLearningRate()
    {
        var value = new[] { 0.0 };
        var parameter = new VectorParameter("p", value);
        var optimizer = new Adam([parameter], learningRate: 0.01);

        parameter.AddGradient([3.0]);
        optimizer.Step();

        Assert.That(value[0], Is.EqualTo(-0.01).Within(1e-6));
    }

    [Test]
    public void ZeroGradients_ClearsAccumulatedGradients()
    {
        var parameter = new VectorParameter("p", new[] { 1.0 });
        var optimizer = new Adam([parameter]);

        parameter.AddGradient([5.0]);
        optimizer.ZeroGradients();

        Assert.That(parameter.GetGradient(0), Is.EqualTo(0));
    }

    [Test]
    public void ClipGradients_ScalesDownOnlyWhenTheGlobalNormExceedsTheLimit()
    {
        var a = new VectorParameter("a", new[] { 0.0 });
        var b = new VectorParameter("b", new[] { 0.0 });
        var optimizer = new Adam([a, b]);

        a.AddGradient([3.0]);
        b.AddGradient([4.0]);
        optimizer.ClipGradients(1.0);

        Assert.That(a.GetGradient(0), Is.EqualTo(0.6).Within(1e-12));
        Assert.That(b.GetGradient(0), Is.EqualTo(0.8).Within(1e-12));
    }

    [Test]
    public void ClipGradients_LeavesSmallGradientsAlone()
    {
        var parameter = new VectorParameter("p", new[] { 0.0 });
        var optimizer = new Adam([parameter]);

        parameter.AddGradient([0.25]);
        optimizer.ClipGradients(1.0);

        Assert.That(parameter.GetGradient(0), Is.EqualTo(0.25).Within(1e-12));
    }

    [Test]
    public void Step_RepeatedlyOnOneSequence_DrivesTrainingLossDown()
    {
        var config = new GptConfig(VocabSize: 7, DModel: 8, NHeads: 2, NLayers: 1, DFf: 16, MaxSeqLen: 6);
        var model = new GptModel(config, new Random(5));
        var optimizer = new Adam(model.Parameters(), learningRate: 0.01);

        int[] tokens = [1, 2, 3, 4];
        int[] targets = [2, 3, 4, 5];

        var (initialLoss, _) = CrossEntropy.Forward(model.Forward(tokens), targets);

        for (var step = 0; step < 40; step++)
        {
            optimizer.ZeroGradients();
            var (_, gradLogits) = CrossEntropy.Forward(model.Forward(tokens), targets);
            model.Backpropagate(gradLogits);
            optimizer.Step();
        }

        var (finalLoss, _) = CrossEntropy.Forward(model.Forward(tokens), targets);

        Assert.That(finalLoss, Is.LessThan(initialLoss * 0.5), $"Loss went from {initialLoss} to {finalLoss}");
    }
}
