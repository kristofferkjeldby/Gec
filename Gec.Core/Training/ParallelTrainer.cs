using Gec.Core.Models;

namespace Gec.Core.Training;

public sealed class ParallelTrainer
{
    private readonly GptModel[] _replicas;
    private readonly Parameter[][] _parameters;
    private readonly ParallelOptions _options;

    public ParallelTrainer(GptConfig config, int replicaCount, Random random, int maxDegreeOfParallelism)
    {
        if (replicaCount < 1)
            throw new ArgumentOutOfRangeException(nameof(replicaCount), $"At least one replica is required, but got {replicaCount}.");

        _replicas = new GptModel[replicaCount];
        for (var i = 0; i < replicaCount; i++)
            _replicas[i] = new GptModel(config, random);

        _parameters = _replicas.Select(replica => replica.Parameters().ToArray()).ToArray();
        _options = new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism };

        BroadcastWeights();
    }

    public GptModel Model => _replicas[0];

    public int ReplicaCount => _replicas.Length;

    public double Backpropagate(IReadOnlyList<(int[] Inputs, int[] Targets)> batch)
    {
        RequireFitsReplicas(batch.Count);

        foreach (var parameters in _parameters)
        foreach (var parameter in parameters)
            parameter.ZeroGradient();

        var losses = new double[batch.Count];

        Parallel.For(0, batch.Count, _options, index =>
        {
            var replica = _replicas[index];
            var (inputs, targets) = batch[index];

            var (loss, gradLogits) = CrossEntropy.Forward(replica.Forward(inputs), targets);
            replica.Backpropagate(gradLogits);

            losses[index] = loss;
        });

        for (var r = 1; r < _replicas.Length; r++)
        for (var p = 0; p < _parameters[0].Length; p++)
            _parameters[0][p].AddGradientFrom(_parameters[r][p]);

        return losses.Sum();
    }

    public double Evaluate(IReadOnlyList<(int[] Inputs, int[] Targets)> batch)
    {
        RequireFitsReplicas(batch.Count);

        var losses = new double[batch.Count];

        Parallel.For(0, batch.Count, _options, index =>
        {
            var (inputs, targets) = batch[index];
            var (loss, _) = CrossEntropy.Forward(_replicas[index].Forward(inputs), targets);

            losses[index] = loss;
        });

        return losses.Sum();
    }

    public void BroadcastWeights()
    {
        for (var r = 1; r < _replicas.Length; r++)
        for (var p = 0; p < _parameters[0].Length; p++)
            _parameters[r][p].CopyValuesFrom(_parameters[0][p]);
    }

    private void RequireFitsReplicas(int count)
    {
        if (count > _replicas.Length)
            throw new ArgumentException($"{count} sequences exceeds the {_replicas.Length} replicas; each sequence needs its own.");
    }
}
