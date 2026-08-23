namespace YANum.Tests

open System
open Xunit
open YANum.Core

module StorageTests =

    let private forceCompactingGen2 () =
        // allocation pressure, then a forced blocking compacting collection
        let mutable sink = Unchecked.defaultof<obj>
        for _ in 1 .. 2000 do
            sink <- box (Array.zeroCreate<byte> 8192)
        ignore sink
        GC.Collect(2, GCCollectionMode.Forced, blocking = true, compacting = true)
        GC.WaitForPendingFinalizers()
        GC.Collect(2, GCCollectionMode.Forced, blocking = true, compacting = true)

    [<Fact>]
    let ``a managed pointer survives five forced compacting gen2 collections`` () =
        let a = NDArray.arange(1024.0, dtype = DType.Float64, backend = Managed)
        let addr0 = a.Storage.Pointer
        Assert.NotEqual(0n, addr0)
        for _ in 1 .. 5 do
            forceCompactingGen2 ()
            Assert.Equal(addr0, a.Storage.Pointer)
        // and the contents are still intact
        let back = a.ToArray<float>()
        for i in 0 .. 1023 do
            Assert.Equal(float i, back.[i])

    [<Fact>]
    let ``native allocations are 64-byte aligned at every size`` () =
        for size in [ 1; 7; 64; 100; 1024; 4096; 1_000_003 ] do
            use s = Storage.AllocateZeroed(unativeint size, Native)
            Assert.Equal(Native, s.Backend)
            Assert.Equal(0L, int64 s.Pointer % 64L)
            Assert.Equal(unativeint size, s.Length)

    [<Fact>]
    let ``a native block is zeroed on request and freed on dispose`` () =
        let s = Storage.AllocateZeroed(256un, Native)
        let span = s.AsByteSpan()
        for i in 0 .. 255 do Assert.Equal(0uy, span.[i])
        Assert.False s.IsFreed
        s.Dispose()
        Assert.True s.IsFreed

    [<Fact>]
    let ``the reference count keeps a native block alive while a view exists`` () =
        let owner = NDArray.arange(16.0, dtype = DType.Float64, backend = Native).Reshape([| 4; 4 |]).Copy()
        Assert.Equal(Native, owner.Backend)
        let view = owner.T
        Assert.Equal(2, owner.Storage.ReferenceCount)
        // disposing the owner must NOT free memory the view still points at
        owner.Dispose()
        Assert.False owner.Storage.IsFreed
        Assert.Equal(1, view.Storage.ReferenceCount)
        Assert.Equal(0.0, view.GetItem<float> [| 0; 0 |])
        Assert.Equal(4.0, view.GetItem<float> [| 0; 1 |])
        view.Dispose()
        Assert.True view.Storage.IsFreed

    [<Fact>]
    let ``CheckAliasing turns a premature native dispose into an error`` () =
        let owner = NDArray.zeros([| 4; 4 |], DType.Float64, backend = Native)
        let view = owner.T
        use _ = Config.push (fun c -> c.CheckAliasing <- true)
        let ex = Assert.Throws<YANumException>(fun () -> owner.Dispose())
        Assert.Contains("live references", ex.Message)
        Assert.False owner.Storage.IsFreed
        Assert.Equal(0.0, view.GetItem<float> [| 0; 0 |])

    [<Fact>]
    let ``disposing a managed-backed array is a no-op for the data`` () =
        let a = NDArray.arange(8.0, dtype = DType.Float64, backend = Managed)
        a.Dispose()
        // the GC owns the lifetime, so the elements are still readable
        Assert.Equal(3.0, a.GetItem<float> [| 3 |])

    [<Fact>]
    let ``a double dispose drops only one reference`` () =
        let owner = NDArray.zeros([| 4 |], DType.Float64, backend = Native)
        let view = owner.Reshape([| 2; 2 |])
        Assert.Equal(2, owner.Storage.ReferenceCount)
        owner.Dispose()
        owner.Dispose()
        Assert.False owner.Storage.IsFreed
        Assert.Equal(1, owner.Storage.ReferenceCount)
        view.Dispose()
        Assert.True owner.Storage.IsFreed

    [<Fact>]
    let ``the managed backend refuses an allocation above its cap`` () =
        let ex = Assert.Throws<YANumValueError>(fun () ->
            Storage.AllocateZeroed(unativeint Int32.MaxValue + 1024un, Managed) |> ignore)
        Assert.Contains("Native", ex.Message)

    [<Fact>]
    let ``typed access reads and writes through the block`` () =
        use s = Storage.AllocateZeroed(64un, Native)
        let span = s.AsSpan<int64>(0n, 8n)
        for i in 0 .. 7 do span.[i] <- int64 (i * i)
        for i in 0 .. 7 do
            Assert.Equal(int64 (i * i), Microsoft.FSharp.NativeInterop.NativePtr.get (s.PointerAs<int64> 0n) i)

    [<Fact>]
    let ``AsSpan refuses a segment that cannot carry an int length`` () =
        use s = Storage.AllocateZeroed(64un, Managed)
        Assert.Throws<YANumValueError>(fun () ->
            s.AsSpan<byte>(0n, nativeint Int32.MaxValue + 1n).Length |> ignore) |> ignore

    [<Fact>]
    let ``a zero-length allocation is legal on both backends`` () =
        for backend in [ Managed; Native ] do
            use s = Storage.AllocateZeroed(0un, backend)
            Assert.Equal(0un, s.Length)
        let a = NDArray.zeros([| 0; 3 |], DType.Float64)
        Assert.Equal(0L, a.Size)
        Assert.True a.IsCContiguous
        Assert.True a.IsFContiguous
