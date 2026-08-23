namespace YANum.Tests

open System
open Xunit
open YANum.Core

module CreationTests =

    let private approx (expected: float[]) (actual: float[]) =
        Assert.Equal(expected.Length, actual.Length)
        for i in 0 .. expected.Length - 1 do
            Assert.True(abs (expected.[i] - actual.[i]) < 1e-12,
                        sprintf "element %d: expected %.17g, got %.17g" i expected.[i] actual.[i])

    // -----------------------------------------------------------------------------------
    //  arange
    // -----------------------------------------------------------------------------------

    [<Fact>]
    let ``arange element counts match NumPy`` () =
        // start, stop, step, expected count
        let cases =
            [ 0.0, 5.0, 1.0, 5
              2.0, 10.0, 3.0, 3
              0.0, 1.0, 0.3, 4
              0.0, 1.0, 0.1, 10
              5.0, 0.0, -1.0, 5
              0.0, 5.0, -1.0, 0
              0.0, 0.0, 1.0, 0
              1.0, 2.5, 0.5, 3
              -3.0, 3.0, 2.0, 3 ]
        for (start, stop, step, expected) in cases do
            let a = NDArray.arange(start, stop, step, DType.Float64)
            Assert.Equal(int64 expected, a.Size)

    [<Fact>]
    let ``arange values are start plus i times step, not accumulated`` () =
        let a = NDArray.arange(0.0, 1.0, 0.1, DType.Float64)
        let v = a.ToArray<float>()
        Assert.Equal(10, v.Length)
        for i in 0 .. 9 do
            Assert.Equal(0.0 + float i * 0.1, v.[i])
        // the naive accumulation would already have drifted by element 3
        Assert.Equal(0.30000000000000004, v.[3])

    [<Fact>]
    let ``arange defaults to the integer dtype when every argument is integral`` () =
        Assert.Equal(Config.DefaultInt, NDArray.arange(5.0).DType)
        Assert.Equal(Config.DefaultInt, NDArray.arange(2.0, 10.0, 2.0).DType)
        Assert.Equal(Config.DefaultFloat, NDArray.arange(0.0, 1.0, 0.25).DType)
        Assert.Equal(DType.Int32, NDArray.arange(5.0, dtype = DType.Int32).DType)

    [<Fact>]
    let ``arange from a single bound starts at zero`` () =
        Assert.Equal<int64[]>([| 0L; 1L; 2L; 3L; 4L |], (NDArray.arange 5.0).ToArray<int64>())

    [<Fact>]
    let ``arange rejects a zero or non-finite step`` () =
        Assert.Throws<YANumValueError>(fun () -> NDArray.arange(0.0, 5.0, 0.0) |> ignore) |> ignore
        Assert.Throws<YANumValueError>(fun () -> NDArray.arange(0.0, 5.0, nan) |> ignore) |> ignore

    // -----------------------------------------------------------------------------------
    //  linspace, logspace, geomspace
    // -----------------------------------------------------------------------------------

    [<Fact>]
    let ``linspace includes the endpoint by default and hits it exactly`` () =
        let a = NDArray.linspace(0.0, 1.0, 5)
        approx [| 0.0; 0.25; 0.5; 0.75; 1.0 |] (a.ToArray<float>())
        Assert.Equal(1.0, a.GetItem<float> [| -1 |])
        // an endpoint that no interpolation would land on exactly
        let b = NDArray.linspace(0.0, 1.0, 7)
        Assert.Equal(1.0, b.GetItem<float> [| -1 |])
        let c = NDArray.linspace(-2.5, 7.25, 13)
        Assert.Equal(7.25, c.GetItem<float> [| -1 |])

    [<Fact>]
    let ``linspace without the endpoint matches NumPy`` () =
        approx [| 0.0; 0.2; 0.4; 0.6; 0.8 |] ((NDArray.linspace(0.0, 1.0, 5, false)).ToArray<float>())

    [<Fact>]
    let ``linspace edge counts match NumPy`` () =
        Assert.Equal(0L, (NDArray.linspace(0.0, 1.0, 0)).Size)
        approx [| 2.0 |] ((NDArray.linspace(2.0, 3.0, 1)).ToArray<float>())
        Assert.Throws<YANumValueError>(fun () -> NDArray.linspace(0.0, 1.0, -1) |> ignore) |> ignore

    [<Fact>]
    let ``logspace spaces exponents`` () =
        approx [| 1.0; 10.0; 100.0 |] ((NDArray.logspace(0.0, 2.0, 3)).ToArray<float>())
        approx [| 1.0; 2.0; 4.0; 8.0 |] ((NDArray.logspace(0.0, 3.0, 4, 2.0)).ToArray<float>())

    [<Fact>]
    let ``geomspace spaces ratios and hits both endpoints exactly`` () =
        let a = NDArray.geomspace(1.0, 1000.0, 4)
        approx [| 1.0; 10.0; 100.0; 1000.0 |] (a.ToArray<float>())
        Assert.Equal(1.0, a.GetItem<float> [| 0 |])
        Assert.Equal(1000.0, a.GetItem<float> [| -1 |])
        approx [| -1.0; -10.0 |] ((NDArray.geomspace(-1.0, -10.0, 2)).ToArray<float>())

    [<Fact>]
    let ``geomspace refuses a zero endpoint or a sign change`` () =
        Assert.Throws<YANumValueError>(fun () -> NDArray.geomspace(0.0, 10.0, 3) |> ignore) |> ignore
        Assert.Throws<YANumValueError>(fun () -> NDArray.geomspace(-1.0, 10.0, 3) |> ignore) |> ignore

    // -----------------------------------------------------------------------------------
    //  zeros, ones, empty, full and their -Like forms
    // -----------------------------------------------------------------------------------

    [<Fact>]
    let ``zeros with a zero extent is legal and yields a valid empty array`` () =
        let a = NDArray.zeros([| 3; 0; 4 |], DType.Float64)
        Assert.Equal(0L, a.Size)
        Assert.Equal(0L, a.NBytes)
        Assert.Equal<int[]>([| 3; 0; 4 |], a.Shape)
        Assert.True a.IsCContiguous
        Assert.True a.IsFContiguous
        Assert.Empty(a.ToArray<float>())
        Assert.False(String.IsNullOrWhiteSpace(a.ToString()))

    [<Fact>]
    let ``zeros ones and full produce the right values in both orders`` () =
        for order in [ C; F ] do
            let z = NDArray.zeros([| 2; 3 |], DType.Int32, order)
            Assert.All(z.ToArray<int32>(), fun v -> Assert.Equal(0, v))
            let o = NDArray.ones([| 2; 3 |], DType.Int32, order)
            Assert.All(o.ToArray<int32>(), fun v -> Assert.Equal(1, v))
            let f = NDArray.full([| 2; 3 |], 7.5, order)
            Assert.All(f.ToArray<float>(), fun v -> Assert.Equal(7.5, v))
            if order = C then Assert.True z.IsCContiguous else Assert.True z.IsFContiguous

    [<Fact>]
    let ``ones works for every dtype that has a numeric one`` () =
        for dt in Generators.allDTypes do
            let a = NDArray.ones([| 3 |], dt)
            Assert.Equal(dt, a.DType)
            Assert.Equal(3L, a.Size)

    [<Fact>]
    let ``the -Like forms copy shape and dtype`` () =
        let proto = NDArray.zeros([| 2; 3; 4 |], DType.Int16)
        for a in [ NDArray.zerosLike proto; NDArray.onesLike proto; NDArray.emptyLike proto ] do
            Assert.Equal<int[]>(proto.Shape, a.Shape)
            Assert.Equal(proto.DType, a.DType)
        let f = NDArray.fullLike(proto, 3.5)
        Assert.Equal<int[]>(proto.Shape, f.Shape)
        Assert.Equal(DType.Float64, f.DType)
        Assert.Equal(DType.Float32, (NDArray.zerosLike(proto, DType.Float32)).DType)

    // -----------------------------------------------------------------------------------
    //  eye and identity
    // -----------------------------------------------------------------------------------

    [<Fact>]
    let ``eye with k and m matches NumPy`` () =
        Assert.Equal<float[]>([| 1.0; 0.0; 0.0; 0.0; 1.0; 0.0; 0.0; 0.0; 1.0 |],
                              (NDArray.eye 3).ToArray<float>())
        let wide = NDArray.eye(2, 3)
        Assert.Equal<int[]>([| 2; 3 |], wide.Shape)
        Assert.Equal<float[]>([| 1.0; 0.0; 0.0; 0.0; 1.0; 0.0 |], wide.ToArray<float>())
        let up = NDArray.eye(3, 3, 1)
        Assert.Equal<float[]>([| 0.0; 1.0; 0.0; 0.0; 0.0; 1.0; 0.0; 0.0; 0.0 |], up.ToArray<float>())
        let down = NDArray.eye(3, 3, -1)
        Assert.Equal<float[]>([| 0.0; 0.0; 0.0; 1.0; 0.0; 0.0; 0.0; 1.0; 0.0 |], down.ToArray<float>())
        // an offset that falls off the matrix leaves it empty
        Assert.All((NDArray.eye(3, 3, 5)).ToArray<float>(), fun v -> Assert.Equal(0.0, v))

    [<Fact>]
    let ``eye is correct in F order too`` () =
        let a = NDArray.eye(3, 3, 0, DType.Float64, F)
        Assert.True a.IsFContiguous
        for i in 0 .. 2 do
            for j in 0 .. 2 do
                Assert.Equal((if i = j then 1.0 else 0.0), a.GetItem<float> [| i; j |])

    [<Fact>]
    let ``identity is the square eye`` () =
        Assert.True((NDArray.identity 4).ArrayEqual(NDArray.eye 4))

    // -----------------------------------------------------------------------------------
    //  Ingest
    // -----------------------------------------------------------------------------------

    [<Fact>]
    let ``ofSeq ofSpan and ofArray agree`` () =
        let expected = [| 1.0; 2.0; 3.0 |]
        Assert.Equal<float[]>(expected, (NDArray.ofArray expected).ToArray<float>())
        Assert.Equal<float[]>(expected, (NDArray.ofSeq (seq { 1.0; 2.0; 3.0 })).ToArray<float>())
        Assert.Equal<float[]>(expected, (NDArray.ofSpan(ReadOnlySpan<float>(expected))).ToArray<float>())

    [<Fact>]
    let ``ofArray3D round-trips through element access`` () =
        let src = Array3D.init 2 3 4 (fun i j k -> float (i * 100 + j * 10 + k))
        let a = NDArray.ofArray3D src
        Assert.Equal<int[]>([| 2; 3; 4 |], a.Shape)
        for i in 0 .. 1 do
            for j in 0 .. 2 do
                for k in 0 .. 3 do
                    Assert.Equal(src.[i, j, k], a.GetItem<float> [| i; j; k |])

    [<Fact>]
    let ``wrap aliases the caller's array without copying`` () =
        let data = [| 1.0; 2.0; 3.0 |]
        let a = NDArray.wrap data
        Assert.Equal<int[]>([| 3 |], a.Shape)
        a.SetItem([| 1 |], 99.0)
        Assert.Equal(99.0, data.[1])
        data.[2] <- -7.0
        Assert.Equal(-7.0, a.GetItem<float> [| 2 |])

    [<Fact>]
    let ``scalar builds a rank-0 array`` () =
        let a = NDArray.scalar 42L
        Assert.Equal(0, a.Rank)
        Assert.Equal(1L, a.Size)
        Assert.Equal(DType.Int64, a.DType)
        Assert.Equal(42L, a.GetItem<int64> [||])
        Assert.True a.IsCContiguous
        Assert.True a.IsFContiguous
        Assert.Empty a.Shape

    [<Fact>]
    let ``fromFunction fills from the index`` () =
        let a = NDArray.fromFunction([| 2; 3 |], Func<int[], float>(fun idx -> float (idx.[0] * 10 + idx.[1])))
        Assert.Equal<float[]>([| 0.0; 1.0; 2.0; 10.0; 11.0; 12.0 |], a.ToArray<float>())
        let b = NDArray.fromFunction([| 2; 3 |], Func<int[], float>(fun idx -> float (idx.[0] * 10 + idx.[1])), F)
        Assert.True b.IsFContiguous
        Assert.True(a.ArrayEqual b)

    [<Fact>]
    let ``copy detaches from the source`` () =
        let a = NDArray.arange(6.0, dtype = DType.Float64)
        let b = NDArray.copy a
        b.SetItem([| 0 |], 99.0)
        Assert.Equal(0.0, a.GetItem<float> [| 0 |])
        Assert.NotSame(a.Storage, b.Storage)

    [<Fact>]
    let ``creation honours the requested backend`` () =
        Assert.Equal(Managed, (NDArray.zeros([| 4 |], DType.Float64, C, Managed)).Backend)
        let n = NDArray.zeros([| 4 |], DType.Float64, C, Native)
        Assert.Equal(Native, n.Backend)
        Assert.Equal(0L, int64 n.Storage.Pointer % 64L)

    [<Fact>]
    let ``a negative extent is rejected`` () =
        Assert.Throws<YANumShapeError>(fun () -> NDArray.zeros [| 2; -1 |] |> ignore) |> ignore

    // -----------------------------------------------------------------------------------
    //  Exit criterion 2: the core operations over all 25 dtypes in both orders
    // -----------------------------------------------------------------------------------

    [<Fact>]
    let ``zeros arange eye transpose reshape copy and printing work for all 25 dtypes in both orders`` () =
        Assert.Equal(25, Generators.allDTypes.Length)
        for dt in Generators.allDTypes do
            for order in [ C; F ] do
                let z = NDArray.zeros([| 2; 3 |], dt, order)
                Assert.Equal(dt, z.DType)
                Assert.Equal(6L, z.Size)
                if order = C then Assert.True z.IsCContiguous else Assert.True z.IsFContiguous

                let r = NDArray.arange(0.0, 6.0, 1.0, dt, order)
                Assert.Equal(6L, r.Size)
                Assert.Equal(dt, r.DType)

                let i = NDArray.eye(3, 3, 0, dt, order)
                Assert.Equal<int[]>([| 3; 3 |], i.Shape)
                Assert.Equal(dt, i.DType)

                let t = i.T
                Assert.Equal<int[]>([| 3; 3 |], t.Shape)
                Assert.True(i.ArrayEqual t.T)

                let s = r.Reshape([| 2; 3 |], order)
                Assert.Equal<int[]>([| 2; 3 |], s.Shape)
                Assert.True(s.Reshape([| 6 |], order).ArrayEqual r)

                let c = s.Copy(if order = C then F else C)
                Assert.True(s.ArrayEqual c, sprintf "%s: a reordered copy differs from its source" dt.Name)

                for text in [ z.ToString(); r.ToString(); i.ToString(); t.ToString(); s.ToString() ] do
                    Assert.False(String.IsNullOrWhiteSpace text)
                    Assert.Contains(dt.Name, text)

    [<Fact>]
    let ``element get and set work through a view for every fixed dtype`` () =
        // the generic accessors are exercised per dtype in DTypeTests; this checks the
        // strided path, where the offset arithmetic rather than the dtype is what can break
        for dt in Generators.allDTypes do
            let a = (NDArray.arange(0.0, 6.0, 1.0, dt)).Reshape([| 2; 3 |]).T
            Assert.Equal<int[]>([| 3; 2 |], a.Shape)
            let flat = a.Copy(C).Ravel()
            Assert.Equal(6L, flat.Size)
            Assert.True(a.ArrayEqual (a.Copy C))
