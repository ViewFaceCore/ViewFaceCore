// System.Runtime.InteropServices.RuntimeInformation on .NET Framework

#if NETFRAMEWORK

namespace System.Runtime.InteropServices
{
    /// <summary>
    /// Provides information about the .NET Framework runtime installation.
    /// </summary>
    public class RuntimeInformation
    {
        /// <summary>
        /// Gets the name of the .NET installation on which an app is running.
        /// </summary>
        public static string FrameworkDescription => $".NET Framework {Environment.Version}";

        /// <summary>
        /// Gets the platform architecture on which the current app is running.
        /// </summary>
        public static Architecture OSArchitecture => Environment.Is64BitOperatingSystem ? Architecture.X64 : Architecture.X86;

        /// <summary>
        /// Gets a string that describes the operating system on which the app is running.
        /// </summary>
        public static string OSDescription => $"Microsoft Windows {Environment.OSVersion.Version}";

        /// <summary>
        /// Gets the process architecture of the currently running app.
        /// </summary>
        public static Architecture ProcessArchitecture => Environment.Is64BitProcess ? Architecture.X64 : Architecture.X86;

        /// <summary>
        /// Gets the platform on which an app is running.
        /// </summary>
        public static string RuntimeIdentifier => $"win{Environment.OSVersion.Version.Major}-{(Environment.Is64BitOperatingSystem ? "x64" : "x86")}";

        /// <summary>
        /// Indicates whether the current application is running on the specified platform.
        /// </summary>
        /// <param name="platform">A platform.</param>
        /// <returns>true if the current app is running on the specified platform; otherwise, false.</returns>
        public static bool IsOSPlatform(OSPlatform platform) => platform == OSPlatform.Windows;
    }

    /// <summary>
    /// Represents an operating system platform.
    /// </summary>
    public enum OSPlatform
    {
        /// <summary>
        /// Gets an object that represents the FreeBSD operating system.
        /// </summary>
        FreeBSD,

        /// <summary>
        /// Gets an object that represents the Linux operating system.
        /// </summary>
        Linux,

        /// <summary>
        /// Gets an object that represents the OSX operating system.
        /// </summary>
        OSX,

        /// <summary>
        /// Gets an object that represents the Windows operating system.
        /// </summary>
        Windows,
    }

    /// <summary>
    /// Indicates the processor architecture.
    /// </summary>
    public enum Architecture
    {
        /// <summary>
        /// An Intel-based 32-bit processor architecture.
        /// </summary>
        X86,
        /// <summary>
        /// An Intel-based 64-bit processor architecture.
        /// </summary>
        X64,
        /// <summary>
        /// A 32-bit ARM processor architecture.
        /// </summary>
        Arm,
        /// <summary>
        /// A 64-bit ARM processor architecture.
        /// </summary>
        Arm64,
    }
}

#endif
