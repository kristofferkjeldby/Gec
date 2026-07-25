using Gec.Core.Common;
using Gec.Core.Training;

namespace Gec.Core.Models;

public class GptModel
{
    private readonly Embedding _tokenEmbedding;
    private readonly Embedding _positionEmbedding;
    private readonly TransformerBlock[] _blocks;
    private readonly LayerNorm _finalNorm;
    private readonly Linear _head;

    public GptModel(GptConfig config, Random? random = null)
    {
        config.Validate();
        Config = config;

        random ??= new Random();

        _tokenEmbedding = new Embedding(config.VocabSize, config.DModel, random, "tokenEmbedding");
        _positionEmbedding = new Embedding(config.MaxSeqLen, config.DModel, random, "positionEmbedding");

        _blocks = new TransformerBlock[config.NLayers];
        for (var i = 0; i < config.NLayers; i++)
            _blocks[i] = new TransformerBlock(config.DModel, config.NHeads, config.DFf, causal: true, random, $"block{i}");

        _finalNorm = new LayerNorm(config.DModel, name: "finalNorm");
        _head = new Linear(config.DModel, config.VocabSize, random, "head");
    }

    public GptConfig Config { get; }

    public IEnumerable<Parameter> Parameters() =>
        _tokenEmbedding.Parameters()
            .Concat(_positionEmbedding.Parameters())
            .Concat(_blocks.SelectMany(block => block.Parameters()))
            .Concat(_finalNorm.Parameters())
            .Concat(_head.Parameters());

    public int ParameterCount() => Parameters().Sum(parameter => parameter.Count);

    public double[,] Forward(int[] tokens)
    {
        if (tokens.Length == 0)
            throw new ArgumentException("At least one token is required.", nameof(tokens));

        if (tokens.Length > Config.MaxSeqLen)
            throw new ArgumentException($"Sequence of {tokens.Length} tokens exceeds the model's maximum of {Config.MaxSeqLen}.", nameof(tokens));

        var positions = Enumerable.Range(0, tokens.Length).ToArray();
        var x = Matrix.Matadd(_tokenEmbedding.Forward(tokens), _positionEmbedding.Forward(positions));

        foreach (var block in _blocks)
            x = block.Forward(x);

        x = _finalNorm.Forward(x);

        return _head.Forward(x);
    }

    public void Backpropagate(double[,] gradLogits)
    {
        var gradX = _head.Backpropagate(gradLogits);
        gradX = _finalNorm.Backpropagate(gradX);

        for (var i = _blocks.Length - 1; i >= 0; i--)
            gradX = _blocks[i].Backpropagate(gradX);

        _tokenEmbedding.Backpropagate(gradX);
        _positionEmbedding.Backpropagate(gradX);
    }
}
