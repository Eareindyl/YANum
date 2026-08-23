namespace YANum.Tests

open System
open Xunit
open YANum.Core

module ViewTests =

    /// A 3x4 array that owns its data: Reshape of a contiguous source returns a view,
    /// so the .Copy() is what makes this the root of a base chain.
    let private m34 () = NDArray.arange(12.0, dtype = DType.Float64).Reshape([| 3; 4 |]).Copy()

    [<Fact>]
    let ``a view shares storage with its base in both directions`` () =
        let a = m34 ()
        let v = a.T
        v.SetItem([| 0; 0 |], 99.0)
        Assert.Equal(99.0, a.GetItem<float> [| 0; 0 |])
        a.SetItem([| 2; 3 |], -5.0)
        Assert.Equal(-5.0, v.GetItem<float> [| 3; 2 |])
        Assert.Same(a.Storage, v.Storage)

    [<Fact>]
    let ``the base chain records the immediate source and the root owner`` () =
        let a = m34 ()
        Assert.Null a.Base
        Assert.True a.OwnsData
        let v = a.T
        Assert.Same(a, v.Base)
        Assert.False v.OwnsData
        let w = v.Reshape([| 12 |], F)
        Assert.Same(v, w.Base)
        Assert.Same(a, w.RootBase)

    [<Fact>]
    let ``transpose composed with transpose is the identity`` () =
        let a = m34 ()
        let b = a.T.T
        Assert.Equal<int[]>(a.Shape, b.Shape)
        Assert.Equal<nativeint[]>(a.Strides, b.Strides)
        Assert.True(a.ArrayEqual b)

    [<Fact>]
    let ``a C-contiguous array transposed is F-contiguous`` () =
        let a = NDArray.zeros([| 3; 4; 5 |], DType.Float64, C)
        Assert.True a.IsCContiguous
        Assert.False a.IsFContiguous
        let t = a.T
        Assert.True t.IsFContiguous
        Assert.False t.IsCContiguous
        // and the other way round
        let b = NDArray.zeros([| 3; 4; 5 |], DType.Float64, F)
        Assert.True b.IsFContiguous
        Assert.True b.T.IsCContiguous

    [<Fact>]
    let ``transpose reverses shape and strides`` () =
        let a = NDArray.zeros([| 2; 3; 4 |], DType.Float64, C)
        Assert.Equal<int[]>([| 4; 3; 2 |], a.T.Shape)
        Assert.Equal<nativeint[]>([| 1n; 4n; 12n |], a.T.Strides)

    [<Fact>]
    let ``permute reorders axes and rejects a non-permutation`` () =
        let a = NDArray.zeros([| 2; 3; 4 |], DType.Float64)
        Assert.Equal<int[]>([| 3; 4; 2 |], (a.Permute [| 1; 2; 0 |]).Shape)
        Assert.Equal<int[]>([| 4; 2; 3 |], (a.Permute [| -1; 0; 1 |]).Shape)
        Assert.Throws<YANumShapeError>(fun () -> a.Permute [| 0; 0; 1 |] |> ignore) |> ignore
        Assert.Throws<YANumShapeError>(fun () -> a.Permute [| 0; 1 |] |> ignore) |> ignore

    [<Fact>]
    let ``swapAxes and moveAxis agree with permute`` () =
        let a = NDArray.zeros([| 2; 3; 4 |], DType.Float64)
        Assert.Equal<int[]>([| 4; 3; 2 |], (a.SwapAxes(0, 2)).Shape)
        Assert.Equal<int[]>([| 3; 4; 2 |], (a.MoveAxis(0, 2)).Shape)
        Assert.Equal<int[]>([| 4; 2; 3 |], (a.MoveAxis(2, 0)).Shape)
        Assert.Equal<int[]>([| 2; 3; 4 |], (a.MoveAxis(1, 1)).Shape)

    [<Fact>]
    let ``expandDims inserts a length-1 axis and squeeze removes it`` () =
        let a = NDArray.arange(6.0, dtype = DType.Float64).Reshape([| 2; 3 |])
        let e = a.ExpandDims 1
        Assert.Equal<int[]>([| 2; 1; 3 |], e.Shape)
        Assert.True e.IsCContiguous
        Assert.Equal<int[]>([| 2; 3 |], (e.Squeeze()).Shape)
        Assert.Equal<int[]>([| 2; 3 |], (e.Squeeze [| 1 |]).Shape)
        Assert.Equal<int[]>([| 1; 2; 3 |], (a.ExpandDims 0).Shape)
        Assert.Equal<int[]>([| 2; 3; 1 |], (a.ExpandDims -1).Shape)
        Assert.True(a.ArrayEqual (e.Squeeze()))

    [<Fact>]
    let ``squeeze refuses an axis whose extent is not one`` () =
        let a = NDArray.zeros([| 2; 1; 3 |], DType.Float64)
        Assert.Throws<YANumShapeError>(fun () -> a.Squeeze [| 0 |] |> ignore) |> ignore

    [<Fact>]
    let ``reshape returns a view when the source is contiguous in the requested order`` () =
        let a = NDArray.arange(12.0, dtype = DType.Float64, order = C).Reshape([| 3; 4 |], C)
        let r = a.Reshape([| 2; 6 |], C)
        Assert.False r.OwnsData
        Assert.Same(a, r.Base)
        Assert.Same(a.Storage, r.Storage)
        // writing through the view is visible in the source
        r.SetItem([| 0; 0 |], 42.0)
        Assert.Equal(42.0, a.GetItem<float> [| 0; 0 |])

    [<Fact>]
    let ``reshape copies when no stride-only view exists`` () =
        let a = NDArray.arange(12.0, dtype = DType.Float64).Reshape([| 3; 4 |], C)
        let t = a.T                                  // 4x3, F-contiguous, not C-contiguous
        let r = t.Reshape([| 12 |], C)
        Assert.True r.OwnsData
        Assert.Null r.Base
        Assert.NotSame(a.Storage, r.Storage)
        // the copy took the C-order reading of the transposed view
        Assert.Equal<float[]>([| 0.0; 4.0; 8.0; 1.0; 5.0; 9.0; 2.0; 6.0; 10.0; 3.0; 7.0; 11.0 |], r.ToArray<float>())
        // and it is genuinely detached
        r.SetItem([| 0 |], -1.0)
        Assert.Equal(0.0, a.GetItem<float> [| 0; 0 |])

    [<Fact>]
    let ``reshape infers a single -1 extent`` () =
        let a = NDArray.arange(12.0, dtype = DType.Float64)
        Assert.Equal<int[]>([| 3; 4 |], (a.Reshape [| 3; -1 |]).Shape)
        Assert.Equal<int[]>([| 6; 2 |], (a.Reshape [| -1; 2 |]).Shape)
        Assert.Throws<YANumShapeError>(fun () -> a.Reshape [| -1; -1 |] |> ignore) |> ignore
        Assert.Throws<YANumShapeError>(fun () -> a.Reshape [| 5; -1 |] |> ignore) |> ignore

    [<Fact>]
    let ``reshape refuses a shape of a different size`` () =
        let a = NDArray.arange(12.0, dtype = DType.Float64)
        let ex = Assert.Throws<YANumShapeError>(fun () -> a.Reshape [| 5; 3 |] |> ignore)
        Assert.Contains("12", ex.Message)

    [<Fact>]
    let ``ravel flattens and AsContiguous is a no-op when already contiguous`` () =
        let a = NDArray.arange(12.0, dtype = DType.Float64).Reshape([| 3; 4 |], C)
        Assert.Equal<int[]>([| 12 |], (a.Ravel()).Shape)
        Assert.Same(a, a.AsContiguous())
        Assert.NotSame(a, a.AsFortran())
        Assert.True(a.ArrayEqual (a.AsFortran()))
        Assert.True (a.AsFortran()).IsFContiguous

    [<Fact>]
    let ``copy is detached and honours the requested order`` () =
        let a = NDArray.arange(6.0, dtype = DType.Float64).Reshape([| 2; 3 |], C)
        let c = a.Copy F
        Assert.True c.OwnsData
        Assert.True c.IsFContiguous
        Assert.True(a.ArrayEqual c)
        c.SetItem([| 0; 0 |], 77.0)
        Assert.Equal(0.0, a.GetItem<float> [| 0; 0 |])

    [<Fact>]
    let ``a transposed view reads the right elements`` () =
        let a = NDArray.arange(6.0, dtype = DType.Float64).Reshape([| 2; 3 |], C)
        let t = a.T
        Assert.Equal<int[]>([| 3; 2 |], t.Shape)
        for i in 0 .. 1 do
            for j in 0 .. 2 do
                Assert.Equal(a.GetItem<float> [| i; j |], t.GetItem<float> [| j; i |])

    [<Fact>]
    let ``AsSpan refuses a non-contiguous array`` () =
        let a = NDArray.arange(12.0, dtype = DType.Float64).Reshape([| 3; 4 |], C)
        let t = a.T.Reshape([| 4; 3 |], F).T          // still a plain view
        Assert.Throws<YANumShapeError>(fun () ->
            NDArray.arange(24.0, dtype = DType.Float64).Reshape([| 2; 3; 4 |]).Permute([| 1; 0; 2 |]).AsSpan<float>().Length
            |> ignore) |> ignore
        ignore t

    [<Fact>]
    let ``views of an empty array stay valid`` () =
        let a = NDArray.zeros([| 0; 3 |], DType.Float64)
        Assert.Equal<int[]>([| 3; 0 |], a.T.Shape)
        Assert.Equal(0L, a.T.Size)
        Assert.True(a.ArrayEqual (a.Copy()))
        Assert.Equal<int[]>([| 0 |], (a.Ravel()).Shape)
