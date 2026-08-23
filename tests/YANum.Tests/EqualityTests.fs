namespace YANum.Tests

open System
open System.Numerics
open Xunit
open YANum.Core

module EqualityTests =

    [<Fact>]
    let ``equal arrays of the same dtype and shape compare equal`` () =
        let a = NDArray.ofArray [| 1.0; 2.0; 3.0 |]
        let b = NDArray.ofArray [| 1.0; 2.0; 3.0 |]
        Assert.True((a = b))
        Assert.True(a.ArrayEqual b)
        Assert.True(a.Equals(box b))

    [<Fact>]
    let ``a different dtype makes arrays unequal even with identical values`` () =
        let a = NDArray.ofArray [| 1.0; 2.0 |]
        let b = NDArray.ofArray [| 1.0f; 2.0f |]
        Assert.False((a = b))

    [<Fact>]
    let ``a different shape makes arrays unequal even with identical elements`` () =
        let a = NDArray.arange(6.0, dtype = DType.Float64).Reshape [| 2; 3 |]
        let b = NDArray.arange(6.0, dtype = DType.Float64).Reshape [| 3; 2 |]
        Assert.False((a = b))
        Assert.False((a = a.Ravel()))

    [<Fact>]
    let ``a NaN array is not equal to itself`` () =
        let a = NDArray.ofArray [| 1.0; nan; 3.0 |]
        Assert.False(a.ArrayEqual a)
        Assert.False((a = NDArray.ofArray [| 1.0; nan; 3.0 |]))
        // and the same at single precision and for complex components
        let b = NDArray.ofArray [| Single.NaN |]
        Assert.False(b.ArrayEqual b)
        let c = NDArray.ofArray [| Complex32(Single.NaN, 0.0f) |]
        Assert.False(c.ArrayEqual c)
        let d = NDArray.ofArray [| Complex(Double.NaN, 0.0) |]
        Assert.False(d.ArrayEqual d)
        // Float16 too, which is not memcmp-legal for exactly this reason
        let e = NDArray.ofArray [| Half.NaN |]
        Assert.False(e.ArrayEqual e)

    [<Fact>]
    let ``negative zero equals positive zero despite differing bytes`` () =
        Assert.True((NDArray.ofArray [| -0.0 |]).ArrayEqual(NDArray.ofArray [| 0.0 |]))
        Assert.True((NDArray.ofArray [| -0.0f |]).ArrayEqual(NDArray.ofArray [| 0.0f |]))
        Assert.True((NDArray.ofArray [| Half.NegativeZero |]).ArrayEqual(NDArray.ofArray [| Half.Zero |]))
        // the bytes really do differ, which is why memcmp is not allowed for floats
        Assert.NotEqual<byte[]>(Bytes.ofValue -0.0, Bytes.ofValue 0.0)

    [<Fact>]
    let ``decimal equality ignores trailing scale`` () =
        Assert.True((NDArray.ofArray [| 1.0m |]).ArrayEqual(NDArray.ofArray [| 1.00m |]))
        Assert.NotEqual<byte[]>(Bytes.ofValue 1.0m, Bytes.ofValue 1.00m)

    [<Fact>]
    let ``a memcmp-legal dtype still compares correctly through non-contiguous views`` () =
        let a = NDArray.arange(6.0, dtype = DType.Int64).Reshape([| 2; 3 |], C)
        let b = a.Copy F
        Assert.True a.IsCContiguous
        Assert.True b.IsFContiguous
        Assert.True(a.ArrayEqual b)      // different layouts, same values
        Assert.True(a.T.ArrayEqual b.T)

    [<Fact>]
    let ``a float array compares correctly through a transposed view`` () =
        let a = NDArray.arange(6.0, dtype = DType.Float64).Reshape([| 2; 3 |], C)
        Assert.True(a.T.ArrayEqual (a.Copy(C).T))
        Assert.True(a.T.T.ArrayEqual a)

    [<Fact>]
    let ``empty arrays of the same shape and dtype are equal`` () =
        Assert.True((NDArray.zeros([| 0; 3 |], DType.Float64)).ArrayEqual(NDArray.zeros([| 0; 3 |], DType.Float64)))
        Assert.False((NDArray.zeros([| 0; 3 |], DType.Float64)).ArrayEqual(NDArray.zeros([| 3; 0 |], DType.Float64)))

    [<Fact>]
    let ``comparing against null or another type is false`` () =
        let a = NDArray.ofArray [| 1.0 |]
        Assert.False(a.ArrayEqual null)
        Assert.False(a.Equals(box 3))

    [<Fact>]
    let ``the hash covers shape and dtype only`` () =
        let a = NDArray.zeros([| 2; 3 |], DType.Float64)
        let b = NDArray.ones([| 2; 3 |], DType.Float64)
        Assert.Equal(a.GetHashCode(), b.GetHashCode())              // documented collision
        Assert.NotEqual(a.GetHashCode(), (NDArray.zeros([| 3; 2 |], DType.Float64)).GetHashCode())
        Assert.NotEqual(a.GetHashCode(), (NDArray.zeros([| 2; 3 |], DType.Float32)).GetHashCode())
        Assert.NotEqual((NDArray.zeros([| 2 |], DType.Utf16 3)).GetHashCode(),
                        (NDArray.zeros([| 2 |], DType.Utf16 4)).GetHashCode())

    [<Fact>]
    let ``every dtype compares equal to a copy of itself`` () =
        for dt in Generators.allDTypes do
            let a = NDArray.ones([| 2; 3 |], dt)
            Assert.True(a.ArrayEqual (a.Copy()), sprintf "%s failed to equal its own copy" dt.Name)
            Assert.True(a.ArrayEqual (a.Copy F), sprintf "%s failed to equal its F-order copy" dt.Name)
