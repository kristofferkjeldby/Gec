using Gec.Core.Models;
using Gec.Core.Serialization;

namespace Gec.Tests.Core.Serialization;

public class ModelSerializerTests
{
    private static GptConfig TinyConfig => new(VocabSize: 7, DModel: 4, NHeads: 2, NLayers: 2, DFf: 6, MaxSeqLen: 5);

    [Test]
    public void SaveThenLoad_ReproducesIdenticalLogits()
    {
        var model = new GptModel(TinyConfig, new Random(11));
        int[] tokens = [1, 3, 2];
        var expected = model.Forward(tokens);

        var path = Path.Combine(Path.GetTempPath(), $"gec-model-{Guid.NewGuid():N}.json");
        try
        {
            ModelSerializer.Save(model, path);
            var loaded = ModelSerializer.Load(path);

            Assert.That(loaded.Config, Is.EqualTo(TinyConfig));

            var actual = loaded.Forward(tokens);
            for (var r = 0; r < expected.GetLength(0); r++)
            for (var c = 0; c < expected.GetLength(1); c++)
                Assert.That(actual[r, c], Is.EqualTo(expected[r, c]).Within(1e-12), $"Mismatch at [{r},{c}]");
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Test]
    public void Load_ThrowsWhenTheConfigDoesNotMatchTheStoredParameters()
    {
        var model = new GptModel(TinyConfig, new Random(11));

        var path = Path.Combine(Path.GetTempPath(), $"gec-model-{Guid.NewGuid():N}.json");
        try
        {
            ModelSerializer.Save(model, path);

            var original = File.ReadAllText(path);
            var text = original.Replace("\"dModel\": 4", "\"dModel\": 8");
            Assert.That(text, Is.Not.EqualTo(original), "the config edit did not apply, so the assertion below would pass for the wrong reason");
            File.WriteAllText(path, text);

            Assert.Throws<InvalidDataException>(() => ModelSerializer.Load(path));
        }
        finally
        {
            File.Delete(path);
        }
    }
}
