namespace YANum.Core

open System
open System.Runtime.CompilerServices

/// <summary>Memory layout order for creation, reshaping and iteration.</summary>
type Order =
    /// <summary>Row-major: the last axis varies fastest.</summary>
    | C
    /// <summary>Column-major: the first axis varies fastest.</summary>
    | F
    /// <summary>Keep whatever order the input already has, preferring C when ambiguous.</summary>
    | K

/// <summary>Which allocator backs an array's storage.</summary>
type StorageBackend =
    /// <summary>A pinned GC array. Capped at <see cref="F:System.Array.MaxLength"/>; disposal is a no-op.</summary>
    | Managed
    /// <summary>64-byte-aligned native memory. Reference counted; capacity limited only by the address space.</summary>
    | Native

/// <summary>How strictly numerical edge cases are reported.</summary>
type ErrorMode =
    /// <summary>Do the cheapest correct thing; do not spend time detecting conditions NumPy warns about.</summary>
    | Fast
    /// <summary>Match NumPy's warning and error behaviour.</summary>
    | NumPy
    /// <summary>Turn anything lossy or ambiguous into an exception.</summary>
    | Strict

/// <summary>When element index bounds are checked.</summary>
type BoundsCheckMode =
    /// <summary>Always check, in every build.</summary>
    | Always
    /// <summary>Check only when YANum.Core itself was built with <c>DEBUG</c>.</summary>
    | DebugOnly
    /// <summary>Never check. Out-of-range indices corrupt memory.</summary>
    | Never

/// <summary>What happens when a string value is longer than its fixed-width dtype.</summary>
type StringOverflow =
    /// <summary>Raise <see cref="T:YANum.Core.YANumOverflowError"/>.</summary>
    | Throw
    /// <summary>Silently keep the leading characters that fit.</summary>
    | Truncate

/// <summary>
/// A complete set of configuration values. Mutable so that
/// <see cref="M:YANum.Core.Config.push"/> can hand one to a mutator.
/// </summary>
[<Sealed>]
type ConfigSnapshot () =

    /// <summary>Layout order used by creation functions that are not told otherwise.</summary>
    member val DefaultOrder : Order = C with get, set
    /// <summary>Dtype used by floating-point creation functions that are not told otherwise.</summary>
    member val DefaultFloat : DType = DType.Float64 with get, set
    /// <summary>Dtype used by integer creation functions that are not told otherwise. <c>Int64</c> on every platform.</summary>
    member val DefaultInt : DType = DType.Int64 with get, set
    /// <summary>Allocator used when none is named.</summary>
    member val DefaultStorage : StorageBackend = Managed with get, set
    /// <summary>How strictly numerical edge cases are reported.</summary>
    member val ErrorMode : ErrorMode = Fast with get, set
    /// <summary>When element index bounds are checked.</summary>
    member val BoundsChecking : BoundsCheckMode = DebugOnly with get, set
    /// <summary>Whether views and native storage lifetimes are audited at runtime.</summary>
    member val CheckAliasing : bool = false with get, set
    /// <summary>When set, a signed/unsigned promotion with no exact signed result raises instead of yielding <c>Float64</c>.</summary>
    member val StrictIntegerPromotion : bool = false with get, set
    /// <summary>What happens when a string value exceeds its dtype width.</summary>
    member val StringOverflow : StringOverflow = Throw with get, set
    /// <summary>Whether SIMD kernels may be selected. Unused in Phase 1.</summary>
    member val EnableSimd : bool = true with get, set
    /// <summary>Element count above which an operation may go parallel. Unused in Phase 1.</summary>
    member val ParallelThreshold : int = 65536 with get, set
    /// <summary>Upper bound on worker threads. Unused in Phase 1.</summary>
    member val MaxDegreeOfParallelism : int = Environment.ProcessorCount with get, set
    /// <summary>Whether reductions must produce bit-identical results across runs. Unused in Phase 1.</summary>
    member val DeterministicReductions : bool = false with get, set
    /// <summary>Element count above which printing summarises. Full summarisation lands in Phase 5.</summary>
    member val PrintThreshold : int = 1000 with get, set
    /// <summary>Elements shown at each end of a summarised axis.</summary>
    member val PrintEdgeItems : int = 3 with get, set
    /// <summary>Significant digits used when printing floating-point values.</summary>
    member val PrintPrecision : int = 8 with get, set

    /// <summary>Produces an independent copy of every setting.</summary>
    /// <returns>A new snapshot with the same values.</returns>
    member this.Clone () : ConfigSnapshot =
        let c = ConfigSnapshot()
        c.DefaultOrder <- this.DefaultOrder
        c.DefaultFloat <- this.DefaultFloat
        c.DefaultInt <- this.DefaultInt
        c.DefaultStorage <- this.DefaultStorage
        c.ErrorMode <- this.ErrorMode
        c.BoundsChecking <- this.BoundsChecking
        c.CheckAliasing <- this.CheckAliasing
        c.StrictIntegerPromotion <- this.StrictIntegerPromotion
        c.StringOverflow <- this.StringOverflow
        c.EnableSimd <- this.EnableSimd
        c.ParallelThreshold <- this.ParallelThreshold
        c.MaxDegreeOfParallelism <- this.MaxDegreeOfParallelism
        c.DeterministicReductions <- this.DeterministicReductions
        c.PrintThreshold <- this.PrintThreshold
        c.PrintEdgeItems <- this.PrintEdgeItems
        c.PrintPrecision <- this.PrintPrecision
        c

/// <summary>
/// Process-global configuration. Read at operation dispatch, never inside a loop.
/// </summary>
/// <remarks>
/// <b>This is process-wide, not thread-local, in v1.</b> A <see cref="M:YANum.Core.Config.push"/>
/// scope changes behaviour for every thread until it is disposed. Use it at start-up or in
/// single-threaded tests, not to vary behaviour between concurrent workers.
/// </remarks>
[<Sealed>]
type Config private () =

    static let mutable state = ConfigSnapshot()

    /// <summary>The live snapshot. Mutating it directly changes global behaviour immediately.</summary>
    static member Current = state

    /// <summary>Layout order used by creation functions that are not told otherwise.</summary>
    static member DefaultOrder = state.DefaultOrder
    /// <summary>Dtype used by floating-point creation functions that are not told otherwise.</summary>
    static member DefaultFloat = state.DefaultFloat
    /// <summary>Dtype used by integer creation functions that are not told otherwise.</summary>
    static member DefaultInt = state.DefaultInt
    /// <summary>Allocator used when none is named.</summary>
    static member DefaultStorage = state.DefaultStorage
    /// <summary>How strictly numerical edge cases are reported.</summary>
    static member ErrorMode = state.ErrorMode
    /// <summary>When element index bounds are checked.</summary>
    static member BoundsChecking = state.BoundsChecking
    /// <summary>Whether views and native storage lifetimes are audited at runtime.</summary>
    static member CheckAliasing = state.CheckAliasing
    /// <summary>When set, a signed/unsigned promotion with no exact signed result raises.</summary>
    static member StrictIntegerPromotion = state.StrictIntegerPromotion
    /// <summary>What happens when a string value exceeds its dtype width.</summary>
    static member StringOverflow = state.StringOverflow
    /// <summary>Whether SIMD kernels may be selected.</summary>
    static member EnableSimd = state.EnableSimd
    /// <summary>Element count above which an operation may go parallel.</summary>
    static member ParallelThreshold = state.ParallelThreshold
    /// <summary>Upper bound on worker threads.</summary>
    static member MaxDegreeOfParallelism = state.MaxDegreeOfParallelism
    /// <summary>Whether reductions must produce bit-identical results across runs.</summary>
    static member DeterministicReductions = state.DeterministicReductions
    /// <summary>Element count above which printing summarises.</summary>
    static member PrintThreshold = state.PrintThreshold
    /// <summary>Elements shown at each end of a summarised axis.</summary>
    static member PrintEdgeItems = state.PrintEdgeItems
    /// <summary>Significant digits used when printing floating-point values.</summary>
    static member PrintPrecision = state.PrintPrecision

    /// <summary>
    /// Whether bounds checking is active right now, resolving
    /// <see cref="T:YANum.Core.BoundsCheckMode"/>'s <c>DebugOnly</c> against the build
    /// configuration of YANum.Core itself.
    /// </summary>
    static member BoundsCheckActive : bool =
        match state.BoundsChecking with
        | Always -> true
        | Never -> false
        | DebugOnly ->
#if DEBUG
            true
#else
            false
#endif

    /// <summary>Restores every setting to its documented default.</summary>
    static member reset () : unit = state <- ConfigSnapshot()

    /// <summary>
    /// Applies a scoped override. Use as
    /// <c>use _ = Config.push (fun c -&gt; c.ErrorMode &lt;- Strict)</c>.
    /// </summary>
    /// <param name="mutate">Receives a fresh copy of the current settings to modify.</param>
    /// <returns>A handle that restores the previous settings when disposed.</returns>
    /// <remarks>The override is process-wide, not thread-local. Nested pushes restore in order.</remarks>
    static member push (mutate: ConfigSnapshot -> unit) : IDisposable =
        let saved = state
        let next = state.Clone()
        mutate next
        state <- next
        { new IDisposable with
            member _.Dispose() = state <- saved }
