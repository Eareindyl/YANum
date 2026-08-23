namespace YANum.Tests

open System
open Xunit
open YANum.Core

module PrintingTests =

    [<Fact>]
    let ``the header names shape and dtype`` () =
        let s = (NDArray.zeros([| 2; 3 |], DType.Float64)).ToString()
        Assert.StartsWith("NDArray(shape=[2;3], dtype=float64)", s)
        Assert.StartsWith("NDArray(shape=[], dtype=int64)", (NDArray.scalar 1L).ToString())

    [<Fact>]
    let ``rank 0 to 3 get nested bracket layouts`` () =
        Assert.EndsWith("42", (NDArray.scalar 42L).ToString())
        Assert.EndsWith("[0 1 2]", (NDArray.arange 3.0).ToString())
        Assert.EndsWith("[[0 1 2]\n [3 4 5]]", (NDArray.arange(6.0).Reshape [| 2; 3 |]).ToString())
        Assert.EndsWith("[[[0 1]\n  [2 3]]\n\n [[4 5]\n  [6 7]]]", (NDArray.arange(8.0).Reshape [| 2; 2; 2 |]).ToString())

    [<Fact>]
    let ``rank 4 and above print a flat listing under the shape header`` () =
        let s = (NDArray.arange(16.0).Reshape [| 2; 2; 2; 2 |]).ToString()
        Assert.StartsWith("NDArray(shape=[2;2;2;2], dtype=int64)", s)
        Assert.EndsWith("[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]", s)

    [<Fact>]
    let ``an F-order array prints the same values as its C-order copy`` () =
        let a = NDArray.arange(6.0, dtype = DType.Float64).Reshape([| 2; 3 |], C)
        Assert.Equal(a.ToString(), (a.Copy F).ToString())

    [<Fact>]
    let ``an empty array prints as empty brackets`` () =
        Assert.EndsWith("[]", (NDArray.zeros([| 0; 3 |], DType.Float64)).ToString())

    [<Fact>]
    let ``floats render with a trailing dot and specials by name`` () =
        let s = (NDArray.ofArray [| 1.0; -0.5; nan; infinity; -infinity |]).ToString()
        Assert.Contains("1.", s)
        Assert.Contains("-0.5", s)
        Assert.Contains("nan", s)
        Assert.Contains("inf", s)
        Assert.Contains("-inf", s)

    [<Fact>]
    let ``bools print as True and False`` () =
        Assert.EndsWith("[True False]", (NDArray.ofArray [| true; false |]).ToString())

    [<Fact>]
    let ``strings and chars print quoted`` () =
        Assert.EndsWith("['hi' 'yo']", (NDArray.ofStrings [| "hi"; "yo" |]).ToString())
        Assert.EndsWith("['a' 'b']", (NDArray.ofArray [| 'a'; 'b' |]).ToString())

    [<Fact>]
    let ``every dtype prints without throwing in both orders`` () =
        for dt in Generators.allDTypes do
            for order in [ C; F ] do
                let a = NDArray.ones([| 2; 3 |], dt, order)
                let s = a.ToString()
                Assert.False(String.IsNullOrWhiteSpace s, sprintf "%s printed nothing" dt.Name)
                Assert.Contains(dt.Name, s)
                Assert.False(String.IsNullOrWhiteSpace(sprintf "%A" a))
                Assert.False(String.IsNullOrWhiteSpace(sprintf "%O" a))

    [<Fact>]
    let ``a large array is truncated rather than printed in full`` () =
        use _ = Config.push (fun c -> c.PrintThreshold <- 10)
        let s = (NDArray.arange 100.0).ToString()
        Assert.Contains("...", s)
        Assert.Contains("100 elements total", s)

    [<Fact>]
    let ``StructuredFormatDisplay routes %A through ToString`` () =
        let a = NDArray.arange 3.0
        Assert.Equal(a.ToString(), sprintf "%A" a)
