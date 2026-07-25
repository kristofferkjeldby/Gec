namespace Gec.Tokenizer.Extensions;

internal static class Extensions
{
    public static List<int> ToInts(this string text, IDictionary<string, int> vocabulary)
    {
        return text.PreTokenize().Select(c => vocabulary[c.ToString()]).ToList();
    }

    public static string[] FromInts(this List<int> tokens, IDictionary<int, string> reverseVocabulary)
    {
        return tokens.Select(t => reverseVocabulary[t]).ToArray();
    }

    // Maps raw text into the character alphabet the tokenizer trains and encodes over: whitespace is
    // turned into visible marker chars so it never collides with the space-delimited merges file.
    // Shared by ToInts and the BPE trainer so trained ids reproduce exactly at encode time.
    public static string PreTokenize(this string text)
    {
        return text
            .Replace("\r\n", "\n")
            .Replace('\r', '\n')
            .Replace('\t', Constants.Space)
            .Replace('\n', Constants.NewLineToken)
            .Replace(Constants.Space, Constants.SpaceToken);
    }

    // Inverse of PreTokenize: concatenates decoded tokens and restores the whitespace markers.
    public static string Detokenize(this IEnumerable<string> tokens)
    {
        return string.Concat(tokens)
            .Replace(Constants.SpaceToken, Constants.Space)
            .Replace(Constants.NewLineToken, '\n');
    }

    public static string[] Lines(this string text)
    {
        var lines = new List<string>();

        using (StringReader reader = new StringReader(text))
        {
            string? line;
            while ((line = reader.ReadLine()) != null)
            {
                lines.Add(line);
            }
        }

        return lines.ToArray();
    }

    public static string JsonUnescape(this string text)
    {
        return text.Replace("\\\"", "\"").Replace("\\\\", "\\");
    }
}