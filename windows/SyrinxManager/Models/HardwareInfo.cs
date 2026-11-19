// What this PC has for running models: its GPU and how much memory it has, the CPU, and the
// system memory. Read from the registry once (the display driver records each adapter's
// memory there), so no GPU library is loaded just to show Settings.
using Microsoft.Win32;

namespace Syrinx.Manager.Models;

internal sealed record GpuInfo(string Name, long MemoryBytes);

internal static class HardwareInfo
{
    private const string DisplayAdapters = @"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}";

    private static readonly Lazy<IReadOnlyList<GpuInfo>> AllGpus = new(ReadGpus);
    private static readonly Lazy<string> Cpu = new(ReadCpu);

    /// <summary>Display adapters, the one with the most memory first.</summary>
    public static IReadOnlyList<GpuInfo> Gpus => AllGpus.Value;

    /// <summary>The GPU models run on: the one with the most dedicated memory.</summary>
    public static GpuInfo? MainGpu => Gpus.FirstOrDefault();

    public static string CpuName => Cpu.Value;

    public static int LogicalProcessors => Environment.ProcessorCount;

    public static long SystemMemoryBytes => GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;

    /// <summary>The engine runs on NVIDIA GPUs (CUDA) for now.</summary>
    public static bool HasCudaGpu => MainGpu?.Name.Contains("NVIDIA", StringComparison.OrdinalIgnoreCase) == true;

    public static MemoryBudget Budget => new(HasCudaGpu ? MainGpu!.MemoryBytes : 0, SystemMemoryBytes);

    private static IReadOnlyList<GpuInfo> ReadGpus()
    {
        var gpus = new List<GpuInfo>();
        try
        {
            using var adapters = Registry.LocalMachine.OpenSubKey(DisplayAdapters);
            if (adapters is null) return gpus;
            foreach (string name in adapters.GetSubKeyNames().Where(n => n.All(char.IsDigit)))
            {
                try
                {
                    using var adapter = adapters.OpenSubKey(name);
                    if (adapter?.GetValue("DriverDesc") is not string description) continue;
                    long memory = adapter.GetValue("HardwareInformation.qwMemorySize") switch
                    {
                        long q => q,
                        byte[] { Length: 8 } bytes => BitConverter.ToInt64(bytes),
                        _ => adapter.GetValue("HardwareInformation.MemorySize") switch
                        {
                            int d => (uint)d,
                            byte[] { Length: 4 } bytes => BitConverter.ToUInt32(bytes),
                            _ => 0,
                        },
                    };
                    // Microsoft's Basic Display Adapter and remote displays have none.
                    if (memory > 0) gpus.Add(new GpuInfo(description, memory));
                }
                catch (System.Security.SecurityException) { /* a key we can't read */ }
            }
        }
        catch (System.Security.SecurityException) { }
        return gpus.OrderByDescending(g => g.MemoryBytes).ToList();
    }

    private static string ReadCpu()
    {
        try
        {
            using var key = Registry.LocalMachine.OpenSubKey(@"HARDWARE\DESCRIPTION\System\CentralProcessor\0");
            if (key?.GetValue("ProcessorNameString") is string name) return name.Trim();
        }
        catch (System.Security.SecurityException) { }
        return "Unknown processor";
    }
}
