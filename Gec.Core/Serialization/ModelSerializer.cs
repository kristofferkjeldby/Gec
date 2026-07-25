using System.Globalization;
using System.Text.Json;
using System.Text.Json.Serialization;
using Gec.Core.Models;

namespace Gec.Core.Serialization;

public static class ModelSerializer
{
    private const int ValuesPerLine = 32;

    private static readonly JsonSerializerOptions Options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.Never
    };

    public static void Save(GptModel model, string filePath)
    {
        var config = model.Config;
        var parameters = model.Parameters().ToArray();

        using var writer = new StreamWriter(filePath);

        writer.WriteLine("{");
        writer.WriteLine("  \"config\": {");
        writer.WriteLine($"    \"vocabSize\": {config.VocabSize},");
        writer.WriteLine($"    \"dModel\": {config.DModel},");
        writer.WriteLine($"    \"nHeads\": {config.NHeads},");
        writer.WriteLine($"    \"nLayers\": {config.NLayers},");
        writer.WriteLine($"    \"dFf\": {config.DFf},");
        writer.WriteLine($"    \"maxSeqLen\": {config.MaxSeqLen}");
        writer.WriteLine("  },");
        writer.WriteLine("  \"parameters\": [");

        for (var p = 0; p < parameters.Length; p++)
        {
            var parameter = parameters[p];

            writer.WriteLine("    {");
            writer.WriteLine($"      \"name\": {JsonSerializer.Serialize(parameter.Name)},");
            writer.WriteLine($"      \"rows\": {parameter.Rows},");
            writer.WriteLine($"      \"cols\": {parameter.Cols},");
            writer.WriteLine("      \"values\": [");

            for (var r = 0; r < parameter.Rows; r++)
            {
                for (var c = 0; c < parameter.Cols; c += ValuesPerLine)
                {
                    var start = r * parameter.Cols + c;
                    var count = Math.Min(ValuesPerLine, parameter.Cols - c);

                    var values = Enumerable.Range(start, count)
                        .Select(i => parameter.GetValue(i).ToString(CultureInfo.InvariantCulture));

                    var isLast = start + count == parameter.Count;

                    writer.WriteLine($"        {string.Join(", ", values)}{(isLast ? string.Empty : ",")}");
                }
            }

            writer.WriteLine("      ]");
            writer.WriteLine(p < parameters.Length - 1 ? "    }," : "    }");
        }

        writer.WriteLine("  ]");
        writer.WriteLine("}");
    }

    public static GptModel Load(string filePath)
    {
        using var stream = File.OpenRead(filePath);

        var file = JsonSerializer.Deserialize<ModelFile>(stream, Options)
                   ?? throw new InvalidDataException($"{filePath} does not contain a model.");

        if (file.Config is null)
            throw new InvalidDataException($"{filePath} is missing its config.");

        var model = new GptModel(file.Config);
        var parameters = model.Parameters().ToArray();

        if (file.Parameters.Length != parameters.Length)
            throw new InvalidDataException(
                $"{filePath} holds {file.Parameters.Length} parameters, but this config builds a model with {parameters.Length}.");

        for (var p = 0; p < parameters.Length; p++)
        {
            var saved = file.Parameters[p];
            var parameter = parameters[p];

            if (saved.Name != parameter.Name || saved.Rows != parameter.Rows || saved.Cols != parameter.Cols)
                throw new InvalidDataException(
                    $"{filePath} parameter {p} is '{saved.Name}' [{saved.Rows}x{saved.Cols}], but this config expects '{parameter.Name}' [{parameter.Rows}x{parameter.Cols}].");

            if (saved.Values.Length != parameter.Count)
                throw new InvalidDataException(
                    $"{filePath} parameter '{saved.Name}' holds {saved.Values.Length} values, but {parameter.Count} were expected.");

            for (var i = 0; i < saved.Values.Length; i++)
                parameter.SetValue(i, saved.Values[i]);
        }

        return model;
    }

    private sealed record ModelFile(GptConfig? Config, ParameterFile[] Parameters);

    private sealed record ParameterFile(string Name, int Rows, int Cols, double[] Values);
}
