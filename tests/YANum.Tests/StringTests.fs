namespace YANum.Tests

open System
open Xunit
open YANum.Core

module StringTests =

    [<Theory>]
    [<InlineData(1)>]
    [<InlineData(2)>]
    [<InlineData(5)>]
    [<InlineData(16)>]
    let ``utf16 round-trips at several widths`` (width: int) =
        let source = [| ""; "a"; String('x', min width 16) |]
        let a = NDArray.ofStrings(source, width)
        Assert.Equal(DType.Utf16 width, a.DType)
        Assert.Equal(2 * width, a.ItemSize)
        let back = a.ToStringArray()
        for i in 0 .. source.Length - 1 do
            let expected = if source.[i].Length > width then source.[i].Substring(0, width) else source.[i]
            Assert.Equal(expected, back.[i])
            Assert.Equal(expected, a.GetString [| i |])

    [<Theory>]
    [<InlineData(1)>]
    [<InlineData(4)>]
    [<InlineData(16)>]
    let ``ascii round-trips at several widths`` (width: int) =
        let source = [| ""; "a"; String('x', min width 16) |]
        let a = NDArray.ofStrings(source, width, true)
        Assert.Equal(DType.Ascii width, a.DType)
        Assert.Equal(width, a.ItemSize)
        let back = a.ToStringArray()
        for i in 0 .. source.Length - 1 do
            let expected = if source.[i].Length > width then source.[i].Substring(0, width) else source.[i]
            Assert.Equal(expected, back.[i])

    [<Fact>]
    let ``the default width is the longest string present`` () =
        let a = NDArray.ofStrings [| "hi"; "world"; "" |]
        Assert.Equal(DType.Utf16 5, a.DType)
        Assert.Equal<string[]>([| "hi"; "world"; "" |], a.ToStringArray())

    [<Fact>]
    let ``an all-empty input still gets a positive width`` () =
        let a = NDArray.ofStrings [| ""; "" |]
        Assert.Equal(DType.Utf16 1, a.DType)
        Assert.Equal<string[]>([| ""; "" |], a.ToStringArray())

    [<Fact>]
    let ``values are NUL-padded and the padding is not part of the value`` () =
        let a = NDArray.ofStrings([| "ab" |], 5)
        let span = a.SpanOf [| 0 |]
        Assert.Equal(5, span.Length)
        Assert.Equal('a', span.[0])
        Assert.Equal('b', span.[1])
        for i in 2 .. 4 do Assert.Equal('\000', span.[i])
        Assert.Equal("ab", a.GetString [| 0 |])

    [<Fact>]
    let ``a shorter value overwrites the tail of a longer one`` () =
        let a = NDArray.ofStrings([| "abcde" |], 5)
        a.SetString([| 0 |], "xy")
        Assert.Equal("xy", a.GetString [| 0 |])
        let span = a.SpanOf [| 0 |]
        for i in 2 .. 4 do Assert.Equal('\000', span.[i])

    [<Fact>]
    let ``overflow throws by default`` () =
        Config.reset ()
        let a = NDArray.ofStrings([| "ab" |], 2)
        let ex = Assert.Throws<YANumOverflowError>(fun () -> a.SetString([| 0 |], "abcdef"))
        Assert.Contains("utf16<2>", ex.Message)
        Assert.Equal("ab", a.GetString [| 0 |])

    [<Fact>]
    let ``overflow truncates under Config.StringOverflow = Truncate`` () =
        let a = NDArray.ofStrings([| "ab" |], 2)
        use _ = Config.push (fun c -> c.StringOverflow <- Truncate)
        a.SetString([| 0 |], "abcdef")
        Assert.Equal("ab", a.GetString [| 0 |])

    [<Fact>]
    let ``overflow policy applies to ascii too`` () =
        let a = NDArray.ofStrings([| "ab" |], 2, true)
        Assert.Throws<YANumOverflowError>(fun () -> a.SetString([| 0 |], "abc")) |> ignore
        use _ = Config.push (fun c -> c.StringOverflow <- Truncate)
        a.SetString([| 0 |], "zzz")
        Assert.Equal("zz", a.GetString [| 0 |])

    [<Fact>]
    let ``a null value stores as the empty string`` () =
        let a = NDArray.ofStrings([| "abc" |], 3)
        a.SetString([| 0 |], null)
        Assert.Equal("", a.GetString [| 0 |])

    [<Fact>]
    let ``string arrays reshape, transpose and copy like any other dtype`` () =
        let a = (NDArray.ofStrings([| "a"; "b"; "c"; "d"; "e"; "f" |], 2)).Reshape([| 2; 3 |])
        Assert.Equal<int[]>([| 2; 3 |], a.Shape)
        Assert.Equal("e", a.GetString [| 1; 1 |])
        Assert.Equal("e", a.T.GetString [| 1; 1 |])
        Assert.Equal("d", a.T.GetString [| 0; 1 |])
        let c = a.Copy F
        Assert.True(a.ArrayEqual c)
        Assert.Equal<string[]>(a.ToStringArray(), c.ToStringArray())

    [<Fact>]
    let ``string access is refused on a non-string dtype`` () =
        let a = NDArray.zeros([| 3 |], DType.Float64)
        Assert.Throws<YANumTypeError>(fun () -> a.GetString [| 0 |] |> ignore) |> ignore
        Assert.Throws<YANumTypeError>(fun () -> a.SetString([| 0 |], "x")) |> ignore
        Assert.Throws<YANumTypeError>(fun () -> a.ToStringArray() |> ignore) |> ignore

    [<Fact>]
    let ``SpanOf needs utf16 rather than ascii`` () =
        let a = NDArray.ofStrings([| "ab" |], 2, true)
        Assert.Throws<YANumTypeError>(fun () -> a.SpanOf([| 0 |]).Length |> ignore) |> ignore

    [<Fact>]
    let ``equal string arrays of the same dtype compare equal`` () =
        let a = NDArray.ofStrings([| "hi"; "yo" |], 4)
        let b = NDArray.ofStrings([| "hi"; "yo" |], 4)
        Assert.True(a.ArrayEqual b)
        let c = NDArray.ofStrings([| "hi"; "yo" |], 5)
        Assert.False(a.ArrayEqual c)     // different dtype: utf16<4> is not utf16<5>
        let d = NDArray.ofStrings([| "hi"; "no" |], 4)
        Assert.False(a.ArrayEqual d)
