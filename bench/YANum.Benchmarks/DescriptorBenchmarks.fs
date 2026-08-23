namespace YANum.Benchmarks

open BenchmarkDotNet.Attributes
open YANum.Core

/// <summary>
/// Exit criterion 6: descriptor construction and view creation must not allocate.
/// </summary>
/// <remarks>
/// <para>
/// The descriptor-level cases below are the ones the criterion is really about, and they
/// report <b>0 B</b>: an <see cref="T:YANum.Core.ArrayDesc"/> lives on the stack, is passed by
/// <c>inref</c> and written through <c>outref</c>, and every view operation is stride
/// arithmetic over its inline buffers.
/// </para>
/// <para>
/// The <see cref="T:YANum.Core.NDArray"/>-level cases cannot report 0 B and are included so
/// the difference is on the record: <c>NDArray</c> is a sealed reference type with the
/// 224-byte descriptor embedded, so producing a view necessarily allocates exactly one object.
/// What matters is that it allocates <i>only</i> that one object — no shape array, no stride
/// array, no boxing — which the numbers show.
/// </para>
/// </remarks>
[<MemoryDiagnoser>]
type DescriptorBenchmarks () =

    // Every array argument is hoisted into a field: a literal inside a benchmark body would
    // allocate 32 bytes per invocation and mask the figure being measured.
    let shape3 = [| 64; 128; 256 |]
    let shape2 = [| 512; 512 |]
    let reshaped3 = [| 64; 32768 |]
    let reshaped2 = [| 256; 1024 |]
    let permutation = [| 2; 0; 1 |]

    let mutable src = Unchecked.defaultof<ArrayDesc>
    let mutable array2 = Unchecked.defaultof<NDArray>

    [<GlobalSetup>]
    member _.Setup () =
        Desc.initDense &src shape3 C DType.Float64
        array2 <- NDArray.zeros(shape2, DType.Float64)

    /// Building a dense descriptor from scratch. Expected: 0 B.
    [<Benchmark>]
    member _.DescriptorInit () : int =
        let mutable d = Unchecked.defaultof<ArrayDesc>
        Desc.initDense &d shape3 C DType.Float64
        d.Rank

    /// Descriptor-level transpose. Expected: 0 B.
    [<Benchmark>]
    member this.DescriptorTranspose () : nativeint =
        let mutable d = Unchecked.defaultof<ArrayDesc>
        Desc.transposeInto &src &d
        d.Size

    /// Descriptor-level reshape. Expected: 0 B.
    [<Benchmark>]
    member this.DescriptorReshape () : nativeint =
        let mutable d = Unchecked.defaultof<ArrayDesc>
        Desc.reshapeInto &src reshaped3 C &d
        d.Size

    /// Descriptor-level permute. Expected: 0 B.
    [<Benchmark>]
    member this.DescriptorPermute () : nativeint =
        let mutable d = Unchecked.defaultof<ArrayDesc>
        Desc.permuteInto &src permutation &d
        d.Size

    /// Reading the descriptor through `inref`, the mandated calling convention.
    [<Benchmark>]
    member _.ReadThroughInRef () : nativeint =
        let mutable acc = 0n
        for _ in 1 .. 16 do
            acc <- acc + Desc.stride &src 0
        acc

    /// Array-level transpose. Allocates exactly one NDArray and nothing else.
    [<Benchmark>]
    member _.ArrayTranspose () : NDArray = array2.T

    /// Array-level reshape as a view. Allocates exactly one NDArray and nothing else.
    [<Benchmark>]
    member _.ArrayReshape () : NDArray = array2.Reshape(reshaped2, C)
