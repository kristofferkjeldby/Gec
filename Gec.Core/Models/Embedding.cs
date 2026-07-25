using Gec.Core.Extensions;
using Gec.Core.Training;

namespace Gec.Core.Models;

public class Embedding
{
    private readonly double[,] _weights;
    private readonly MatrixParameter _weightParameter;

    private int[] _indices = [];

    public Embedding(int count, int dModel, Random? random = null, string name = "embedding")
    {
        _weights = new double[count, dModel];

        random ??= new Random();
        for (var i = 0; i < count; i++)
        for (var j = 0; j < dModel; j++)
            _weights[i, j] = random.NextGaussian(0.02);

        _weightParameter = new MatrixParameter($"{name}.weights", _weights);
    }

    public int Count => _weights.Rows();

    public IEnumerable<Parameter> Parameters()
    {
        yield return _weightParameter;
    }

    public double[,] Forward(int[] indices)
    {
        _indices = indices;

        var dModel = _weights.Cols();
        var output = new double[indices.Length, dModel];

        for (var i = 0; i < indices.Length; i++)
        {
            var index = indices[i];
            if (index < 0 || index >= Count)
                throw new ArgumentOutOfRangeException(nameof(indices), $"Index {index} is outside the embedding table of {Count} rows.");

            for (var j = 0; j < dModel; j++)
                output[i, j] = _weights[index, j];
        }

        return output;
    }

    public void Backpropagate(double[,] gradOutput)
    {
        for (var i = 0; i < _indices.Length; i++)
            _weightParameter.AddRowGradient(_indices[i], gradOutput.Row(i));
    }
}
