namespace YANum.Tests

open System
open Xunit
open YANum.Core

/// <summary>
/// Exit criterion 6, asserted rather than merely measured. The BenchmarkDotNet project under
/// <c>bench/</c> reports the same figures with a <c>[MemoryDiagnoser]</c>; these tests fail the
/// build if descriptor construction or view creation ever starts allocating.
/// </summary>
module AllocationTests =

    /// Runs `f` `iterations` times and returns the bytes allocated on this thread, after a
    /// warm-up pass so that JIT and first-call costs are not counted.
    let private allocatedBy (iterations: int) (f: unit -> unit) : int64 =
        f ()
        f ()
        GC.Collect()
        GC.WaitForPendingFinalizers()
        let before = GC.GetAllocatedBytesForCurrentThread()
        for _ in 1 .. iterations do f ()
        GC.GetAllocatedBytesForCurrentThread() - before

    let private shape = [| 64; 128; 256 |]
    let private newShape = [| 64; 32768 |]
    let private perm = [| 2; 0; 1 |]

    [<Fact>]
    let ``descriptor construction allocates nothing`` () =
        let mutable sink = 0
        let bytes =
            allocatedBy 1000 (fun () ->
                let mutable d = Unchecked.defaultof<ArrayDesc>
                Desc.initDense &d shape C DType.Float64
                sink <- sink + d.Rank)
        Assert.Equal(0L, bytes)
        Assert.True(sink > 0)

    [<Fact>]
    let ``descriptor transpose allocates nothing`` () =
        let mutable src = Unchecked.defaultof<ArrayDesc>
        Desc.initDense &src shape C DType.Float64
        let mutable sink = 0n
        let bytes =
            allocatedBy 1000 (fun () ->
                let mutable d = Unchecked.defaultof<ArrayDesc>
                Desc.transposeInto &src &d
                sink <- sink + d.Size)
        Assert.Equal(0L, bytes)
        Assert.True(sink > 0n)

    [<Fact>]
    let ``descriptor reshape allocates nothing`` () =
        let mutable src = Unchecked.defaultof<ArrayDesc>
        Desc.initDense &src shape C DType.Float64
        let mutable sink = 0n
        let bytes =
            allocatedBy 1000 (fun () ->
                let mutable d = Unchecked.defaultof<ArrayDesc>
                Desc.reshapeInto &src newShape C &d
                sink <- sink + d.Size)
        Assert.Equal(0L, bytes)
        Assert.True(sink > 0n)

    [<Fact>]
    let ``descriptor permute allocates nothing`` () =
        let mutable src = Unchecked.defaultof<ArrayDesc>
        Desc.initDense &src shape C DType.Float64
        let mutable sink = 0n
        let bytes =
            allocatedBy 1000 (fun () ->
                let mutable d = Unchecked.defaultof<ArrayDesc>
                Desc.permuteInto &src perm &d
                sink <- sink + d.Size)
        Assert.Equal(0L, bytes)
        Assert.True(sink > 0n)

    [<Fact>]
    let ``reading a descriptor through inref allocates nothing`` () =
        let mutable src = Unchecked.defaultof<ArrayDesc>
        Desc.initDense &src shape C DType.Float64
        let mutable sink = 0n
        let bytes =
            allocatedBy 1000 (fun () ->
                sink <- sink + Desc.stride &src 0 + nativeint (Desc.extent &src 1))
        Assert.Equal(0L, bytes)
        Assert.True(sink > 0n)

    /// <summary>
    /// The array-level view operations cannot be zero-allocation, and this pins how far from
    /// zero they are.
    /// </summary>
    /// <remarks>
    /// <see cref="T:YANum.Core.NDArray"/> is a sealed reference type with the 224-byte
    /// descriptor embedded, so producing a view allocates exactly one object. The point of
    /// this test is that it allocates <b>only</b> that one object: no shape array, no stride
    /// array, no boxing, no closure.
    /// </remarks>
    [<Fact>]
    let ``an array view allocates exactly one NDArray and nothing else`` () =
        let a = NDArray.zeros([| 512; 512 |], DType.Float64)
        let target = [| 256; 1024 |]
        let mutable sink = 0
        // the harness itself costs a little per iteration; measure and subtract it
        let baseline = allocatedBy 1000 (fun () -> sink <- sink + a.Rank) / 1000L
        let perView = allocatedBy 1000 (fun () -> sink <- sink + a.T.Rank) / 1000L - baseline
        let perReshape = allocatedBy 1000 (fun () -> sink <- sink + (a.Reshape(target, C)).Rank) / 1000L - baseline
        // one object: the 224-byte descriptor plus an object header and three fields, rounded
        let oneObject = int64 (System.Runtime.CompilerServices.Unsafe.SizeOf<ArrayDesc>() + 64)
        Assert.InRange(perView, 1L, oneObject)
        Assert.InRange(perReshape, 1L, oneObject)
        // no shape array, no stride array, no closure, no boxing: a reshape view costs exactly
        // what a transpose view costs
        Assert.Equal(perView, perReshape)
        Assert.True(sink > 0)
