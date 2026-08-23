namespace YANum.Tests

open System
open Xunit
open FsCheck
open FsCheck.FSharp
open YANum.Core

module ShapeTests =

    let private check (p: 'a) = Check.QuickThrowOnFailure p

    // -----------------------------------------------------------------------------------
    //  Property tests over generated shapes
    // -----------------------------------------------------------------------------------

    [<Fact>]
    let ``size is the product of the extents`` () =
        check (Prop.forAll Generators.arbShape (fun (ArrayShape s) ->
            let expected = Array.fold (fun acc e -> acc * nativeint e) 1n s
            Shape.size s = expected))

    [<Fact>]
    let ``a shape with a zero extent has size zero`` () =
        check (Prop.forAll Generators.arbShape (fun (ArrayShape s) ->
            (Array.exists ((=) 0) s) ==> lazy (Shape.size s = 0n)))

    [<Fact>]
    let ``cStrides makes the shape C-contiguous`` () =
        check (Prop.forAll Generators.arbShape (fun (ArrayShape s) ->
            Shape.isCContiguous s (Shape.cStrides s)))

    [<Fact>]
    let ``fStrides makes the shape F-contiguous`` () =
        check (Prop.forAll Generators.arbShape (fun (ArrayShape s) ->
            Shape.isFContiguous s (Shape.fStrides s)))

    [<Fact>]
    let ``cStrides is the reverse of fStrides on the reversed shape`` () =
        check (Prop.forAll Generators.arbShape (fun (ArrayShape s) ->
            let a = Shape.cStrides s
            let b = Array.rev (Shape.fStrides (Array.rev s))
            a = b))

    [<Fact>]
    let ``the innermost C stride and the outermost F stride are one`` () =
        check (Prop.forAll Generators.arbShape (fun (ArrayShape s) ->
            (s.Length > 0) ==> lazy
                (Shape.cStrides s |> Array.last = 1n && Shape.fStrides s |> Array.head = 1n)))

    [<Fact>]
    let ``an array of size at most one is both C- and F-contiguous whatever the strides`` () =
        check (Prop.forAll Generators.arbShape (fun (ArrayShape s) ->
            (Shape.size s <= 1n) ==> lazy
                (let junk = Array.map (fun e -> nativeint (e * 7919 + 13)) s
                 Shape.isCContiguous s junk && Shape.isFContiguous s junk)))

    [<Fact>]
    let ``an axis of extent one imposes no constraint on contiguity`` () =
        check (Prop.forAll Generators.arbNonEmptyShape (fun (NonEmptyShape s) ->
            // force an extent-1 axis into the shape, then corrupt its stride
            (s.Length > 0) ==> lazy
                (let shape = Array.copy s
                 shape.[0] <- 1
                 let cs = Shape.cStrides shape
                 cs.[0] <- 123456n
                 let fs = Shape.fStrides shape
                 fs.[0] <- 123456n
                 Shape.isCContiguous shape cs && Shape.isFContiguous shape fs)))

    [<Fact>]
    let ``rank 0 is legal, has size 1, and is contiguous both ways`` () =
        let s : int[] = [||]
        Shape.validate s
        Assert.Equal(1n, Shape.size s)
        Assert.True(Shape.isCContiguous s [||])
        Assert.True(Shape.isFContiguous s [||])

    [<Fact>]
    let ``canReshape requires the sizes to match`` () =
        check (Prop.forAll Generators.arbShape (fun (ArrayShape s) ->
            let strides = Shape.cStrides s
            let bad = Array.append s [| 7 |]
            (Shape.size s <> Shape.size bad)
            ==> lazy (not (Shape.canReshape s strides bad C))))

    [<Fact>]
    let ``a C-contiguous array can always be reshaped as a view in C order`` () =
        check (Prop.forAll Generators.arbShape (fun (ArrayShape s) ->
            let strides = Shape.cStrides s
            let flat = [| int (Shape.size s) |]
            Shape.canReshape s strides flat C))

    [<Fact>]
    let ``an F-contiguous non-trivial array is not reshapeable as a C-order view`` () =
        let s = [| 2; 3 |]
        let f = Shape.fStrides s
        Assert.False(Shape.canReshape s f [| 6 |] C)
        Assert.True(Shape.canReshape s f [| 6 |] F)
        Assert.True(Shape.canReshape s f [| 6 |] K)

    // -----------------------------------------------------------------------------------
    //  Axis normalisation and validation
    // -----------------------------------------------------------------------------------

    [<Fact>]
    let ``normalizeAxis counts negatives from the end`` () =
        Assert.Equal(0, Shape.normalizeAxis 3 0)
        Assert.Equal(2, Shape.normalizeAxis 3 2)
        Assert.Equal(2, Shape.normalizeAxis 3 -1)
        Assert.Equal(0, Shape.normalizeAxis 3 -3)

    [<Fact>]
    let ``normalizeAxis rejects an out-of-range axis`` () =
        Assert.Throws<YANumShapeError>(fun () -> Shape.normalizeAxis 3 3 |> ignore) |> ignore
        Assert.Throws<YANumShapeError>(fun () -> Shape.normalizeAxis 3 -4 |> ignore) |> ignore
        Assert.Throws<YANumShapeError>(fun () -> Shape.normalizeAxis 0 0 |> ignore) |> ignore

    [<Fact>]
    let ``validate accepts zero extents and rejects negative ones`` () =
        Shape.validate [| 3; 0; 4 |]
        let ex = Assert.Throws<YANumShapeError>(fun () -> Shape.validate [| 3; -1 |])
        Assert.Contains("negative", ex.Message)

    [<Fact>]
    let ``rank 16 is legal and rank 17 is a shape error`` () =
        Shape.validate (Array.create 16 1)
        let ex = Assert.Throws<YANumShapeError>(fun () -> Shape.validate (Array.create 17 1))
        Assert.Contains("17", ex.Message)
        Assert.Contains("16", ex.Message)

    [<Fact>]
    let ``an array of rank 16 can be built, viewed and printed`` () =
        let shape = Array.create 16 1
        let a = NDArray.zeros(shape, DType.Float64)
        Assert.Equal(16, a.Rank)
        Assert.Equal(1L, a.Size)
        Assert.Equal(16, a.T.Rank)
        Assert.False(String.IsNullOrWhiteSpace(a.ToString()))
        Assert.Throws<YANumShapeError>(fun () -> a.ExpandDims 0 |> ignore) |> ignore

    [<Fact>]
    let ``an array of rank 17 is refused`` () =
        Assert.Throws<YANumShapeError>(fun () -> NDArray.zeros(Array.create 17 1) |> ignore) |> ignore

    [<Fact>]
    let ``strides are in elements, and StridesBytes scales them by the item size`` () =
        let a = NDArray.zeros([| 2; 3; 4 |], DType.Float64)
        Assert.Equal<nativeint[]>([| 12n; 4n; 1n |], a.Strides)
        Assert.Equal<nativeint[]>([| 96n; 32n; 8n |], a.StridesBytes)
        let b = NDArray.zeros([| 2; 3; 4 |], DType.Int16)
        Assert.Equal<nativeint[]>([| 12n; 4n; 1n |], b.Strides)
        Assert.Equal<nativeint[]>([| 24n; 8n; 2n |], b.StridesBytes)
