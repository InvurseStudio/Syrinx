// Reads the header of a GGUF file: its metadata and the list of its tensors, without the
// weights. Enough to say what a model is (architecture, size, quantization, context length)
// and whether Vireo's engine can run it. The format is documented in ggml's docs/gguf.md;
// the engine's own reader is Syrinx/crates/syrinx-engine/src/gguf.rs.
using System.Text;

namespace Syrinx.Manager.Models;

/// <summary>ggml's tensor types, by their number in the file.</summary>
public static class GgufTensorTypes
{
    private static readonly Dictionary<uint, string> Names = new()
    {
        [0] = "F32", [1] = "F16", [2] = "Q4_0", [3] = "Q4_1", [6] = "Q5_0", [7] = "Q5_1", [8] = "Q8_0", [9] = "Q8_1",
        [10] = "Q2_K", [11] = "Q3_K", [12] = "Q4_K", [13] = "Q5_K", [14] = "Q6_K", [15] = "Q8_K",
        [16] = "IQ2_XXS", [17] = "IQ2_XS", [18] = "IQ3_XXS", [19] = "IQ1_S", [20] = "IQ4_NL", [21] = "IQ3_S", [22] = "IQ2_S",
        [23] = "IQ4_XS", [24] = "I8", [25] = "I16", [26] = "I32", [27] = "I64", [28] = "F64", [29] = "IQ1_M", [30] = "BF16",
        [34] = "TQ1_0", [35] = "TQ2_0", [39] = "MXFP4",
    };

    public static string Name(uint type) => Names.GetValueOrDefault(type) ?? $"type {type}";
}

/// <summary>llama.cpp's general.file_type: the quantization a file was made with.</summary>
public static class GgufFileTypes
{
    private static readonly Dictionary<uint, string> Names = new()
    {
        [0] = "F32", [1] = "F16", [2] = "Q4_0", [3] = "Q4_1", [7] = "Q8_0", [8] = "Q5_0", [9] = "Q5_1",
        [10] = "Q2_K", [11] = "Q3_K_S", [12] = "Q3_K_M", [13] = "Q3_K_L", [14] = "Q4_K_S", [15] = "Q4_K_M",
        [16] = "Q5_K_S", [17] = "Q5_K_M", [18] = "Q6_K", [19] = "IQ2_XXS", [20] = "IQ2_XS", [21] = "Q2_K_S",
        [22] = "IQ3_XS", [23] = "IQ3_XXS", [24] = "IQ1_S", [25] = "IQ4_NL", [26] = "IQ3_S", [27] = "IQ3_M",
        [28] = "IQ2_S", [29] = "IQ2_M", [30] = "IQ4_XS", [31] = "IQ1_M", [32] = "BF16", [36] = "TQ1_0", [37] = "TQ2_0",
        [38] = "MXFP4",
    };

    public static string? Name(uint type) => Names.GetValueOrDefault(type);
}

public sealed record GgufTensor(string Name, ulong[] Dimensions, uint Type)
{
    public ulong Elements => Dimensions.Aggregate(1UL, (product, d) => product * d);
}

/// <summary>What a GGUF file's header says about the model in it.</summary>
public sealed record GgufInfo
{
    public required uint Version { get; init; }
    /// <summary>Metadata values other than arrays (which are skipped: the tokenizer's run to
    /// hundreds of thousands of entries). Strings, numbers and booleans as read.</summary>
    public required IReadOnlyDictionary<string, object> Metadata { get; init; }
    public required IReadOnlyList<GgufTensor> Tensors { get; init; }

    public string? Architecture => Text("general.architecture");
    public string? Name => Text("general.name");
    public string? SizeLabel => Text("general.size_label");

    /// <summary>The context length the model was trained for.</summary>
    public long? ContextLength => Architecture is { } arch ? Number($"{arch}.context_length") : null;

    public long? LayerCount => Architecture is { } arch ? Number($"{arch}.block_count") : null;

    /// <summary>Every weight in the file, counted from the tensors' shapes.</summary>
    public long ParameterCount => (long)Tensors.Aggregate(0UL, (sum, t) => sum + t.Elements);

    /// <summary>The tensor types the file uses, by name (F32, Q4_K…).</summary>
    public IReadOnlySet<string> TensorTypes => Tensors.Select(t => GgufTensorTypes.Name(t.Type)).ToHashSet();

    /// <summary>The quantization's usual name (Q4_K_M), from general.file_type, or else the
    /// type most of the weights use.</summary>
    public string Quantization
    {
        get
        {
            if (Number("general.file_type") is { } fileType && GgufFileTypes.Name((uint)fileType) is { } name) return name;
            var most = Tensors.GroupBy(t => t.Type).MaxBy(g => g.Aggregate(0UL, (sum, t) => sum + t.Elements));
            return most is null ? "unknown" : GgufTensorTypes.Name(most.Key);
        }
    }

    /// <summary>One of several files a model was split into (split.count above 1).</summary>
    public int SplitCount => (int)(Number("split.count") ?? 1);

    /// <summary>A vision projector (mmproj), not a model that chats on its own.</summary>
    public bool IsVisionProjector => Architecture == "clip";

    public string? Text(string key) => Metadata.GetValueOrDefault(key) as string;

    public long? Number(string key) => Metadata.GetValueOrDefault(key) switch
    {
        byte v => v, sbyte v => v, ushort v => v, short v => v, uint v => v, int v => v,
        ulong v => (long)v, long v => v,
        _ => null,
    };
}

public sealed class GgufFormatException(string message) : Exception(message);

public static class GgufReader
{
    private const uint Magic = 0x46554747; // "GGUF", little-endian

    // A sanity limit, so a damaged file can't ask for gigabytes of strings or tensors.
    private const ulong MaxCount = 1_000_000;
    private const ulong MaxString = 64 * 1024 * 1024;

    public static GgufInfo Read(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete, 1 << 16);
        return Read(stream);
    }

    public static GgufInfo Read(Stream stream)
    {
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        try
        {
            if (reader.ReadUInt32() != Magic) throw new GgufFormatException("Not a GGUF file.");
            uint version = reader.ReadUInt32();
            if (version is < 2 or > 3) throw new GgufFormatException($"GGUF version {version} isn't supported.");
            ulong tensorCount = Count(reader.ReadUInt64());
            ulong metadataCount = Count(reader.ReadUInt64());

            var metadata = new Dictionary<string, object>();
            for (ulong i = 0; i < metadataCount; i++)
            {
                string key = ReadString(reader);
                uint type = reader.ReadUInt32();
                if (ReadValue(reader, type) is { } value) metadata[key] = value;
            }

            var tensors = new List<GgufTensor>((int)Math.Min(tensorCount, 4096));
            for (ulong i = 0; i < tensorCount; i++)
            {
                string name = ReadString(reader);
                uint dimensions = reader.ReadUInt32();
                if (dimensions > 8) throw new GgufFormatException($"Tensor {name} has {dimensions} dimensions.");
                var shape = new ulong[dimensions];
                for (int d = 0; d < dimensions; d++) shape[d] = reader.ReadUInt64();
                uint tensorType = reader.ReadUInt32();
                reader.ReadUInt64(); // offset of its data
                tensors.Add(new GgufTensor(name, shape, tensorType));
            }
            return new GgufInfo { Version = version, Metadata = metadata, Tensors = tensors };
        }
        catch (EndOfStreamException)
        {
            throw new GgufFormatException("The file ends inside its header: it may be incomplete.");
        }
    }

    private static ulong Count(ulong count) =>
        count <= MaxCount ? count : throw new GgufFormatException($"The header claims {count:N0} entries.");

    private static string ReadString(BinaryReader reader)
    {
        ulong length = reader.ReadUInt64();
        if (length > MaxString) throw new GgufFormatException("A string in the header is too long.");
        return Encoding.UTF8.GetString(reader.ReadBytes((int)length));
    }

    /// <summary>A metadata value; arrays are skipped and give null.</summary>
    private static object? ReadValue(BinaryReader reader, uint type)
    {
        switch (type)
        {
            case 0: return reader.ReadByte();
            case 1: return reader.ReadSByte();
            case 2: return reader.ReadUInt16();
            case 3: return reader.ReadInt16();
            case 4: return reader.ReadUInt32();
            case 5: return reader.ReadInt32();
            case 6: return reader.ReadSingle();
            case 7: return reader.ReadByte() != 0;
            case 8: return ReadString(reader);
            case 9:
                uint elementType = reader.ReadUInt32();
                ulong count = reader.ReadUInt64();
                if (count > 100 * MaxCount) throw new GgufFormatException("An array in the header is too long.");
                SkipArray(reader, elementType, count);
                return null;
            case 10: return reader.ReadUInt64();
            case 11: return reader.ReadInt64();
            case 12: return reader.ReadDouble();
            default: throw new GgufFormatException($"Unknown metadata type {type}.");
        }
    }

    private static void SkipArray(BinaryReader reader, uint elementType, ulong count)
    {
        int size = elementType switch
        {
            0 or 1 or 7 => 1,
            2 or 3 => 2,
            4 or 5 or 6 => 4,
            10 or 11 or 12 => 8,
            _ => 0,
        };
        var stream = reader.BaseStream;
        if (size > 0)
        {
            Skip(stream, (long)count * size);
            return;
        }
        for (ulong i = 0; i < count; i++)
        {
            if (elementType == 8)
            {
                ulong length = reader.ReadUInt64();
                if (length > MaxString) throw new GgufFormatException("A string in the header is too long.");
                Skip(stream, (long)length);
            }
            else if (elementType == 9)
            {
                uint nestedType = reader.ReadUInt32();
                SkipArray(reader, nestedType, reader.ReadUInt64());
            }
            else
            {
                throw new GgufFormatException($"Unknown array type {elementType}.");
            }
        }
    }

    private static void Skip(Stream stream, long bytes)
    {
        if (stream.CanSeek)
        {
            if (stream.Position + bytes > stream.Length) throw new EndOfStreamException();
            stream.Seek(bytes, SeekOrigin.Current);
            return;
        }
        var buffer = new byte[Math.Min(bytes, 81920)];
        while (bytes > 0)
        {
            int read = stream.Read(buffer, 0, (int)Math.Min(buffer.Length, bytes));
            if (read == 0) throw new EndOfStreamException();
            bytes -= read;
        }
    }
}

/// <summary>Writes small GGUF files: for tests, and nothing else needs it yet.</summary>
public sealed class GgufWriter
{
    private readonly List<(string Key, uint Type, Action<BinaryWriter> Write)> _metadata = [];
    private readonly List<(string Name, ulong[] Shape, uint Type)> _tensors = [];

    public GgufWriter Add(string key, string value) => Put(key, 8, w => WriteString(w, value));
    public GgufWriter Add(string key, uint value) => Put(key, 4, w => w.Write(value));
    public GgufWriter Add(string key, ulong value) => Put(key, 10, w => w.Write(value));
    public GgufWriter Add(string key, float value) => Put(key, 6, w => w.Write(value));
    public GgufWriter Add(string key, bool value) => Put(key, 7, w => w.Write((byte)(value ? 1 : 0)));

    public GgufWriter Add(string key, string[] values) => Put(key, 9, w =>
    {
        w.Write(8u);
        w.Write((ulong)values.Length);
        foreach (var v in values) WriteString(w, v);
    });

    public GgufWriter Add(string key, int[] values) => Put(key, 9, w =>
    {
        w.Write(5u);
        w.Write((ulong)values.Length);
        foreach (var v in values) w.Write(v);
    });

    public GgufWriter Tensor(string name, uint type, params ulong[] shape)
    {
        _tensors.Add((name, shape, type));
        return this;
    }

    private GgufWriter Put(string key, uint type, Action<BinaryWriter> write)
    {
        _metadata.Add((key, type, write));
        return this;
    }

    public byte[] ToBytes()
    {
        using var memory = new MemoryStream();
        using (var w = new BinaryWriter(memory, Encoding.UTF8, leaveOpen: true))
        {
            w.Write(0x46554747u);
            w.Write(3u);
            w.Write((ulong)_tensors.Count);
            w.Write((ulong)_metadata.Count);
            foreach (var (key, type, write) in _metadata)
            {
                WriteString(w, key);
                w.Write(type);
                write(w);
            }
            ulong offset = 0;
            foreach (var (name, shape, type) in _tensors)
            {
                WriteString(w, name);
                w.Write((uint)shape.Length);
                foreach (var d in shape) w.Write(d);
                w.Write(type);
                w.Write(offset);
                offset += 32;
            }
        }
        return memory.ToArray();
    }

    private static void WriteString(BinaryWriter w, string value)
    {
        var bytes = Encoding.UTF8.GetBytes(value);
        w.Write((ulong)bytes.Length);
        w.Write(bytes);
    }
}
