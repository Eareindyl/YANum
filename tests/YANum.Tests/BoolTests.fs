namespace YANum.Tests

open System
open System.Runtime.InteropServices
open Xunit
open YANum.Core

module BoolTests =

    /// Produces a `bool[]` whose bytes are 0x02 — reachable through YANum's own
    /// MemoryMarshal reinterpretation, and the exact value that compares as not-`true`
    /// under `=` while `if b` takes the true branch.
    let private weirdBools (n: int) : bool[] =
        let bytes = Array.create n 2uy
        MemoryMarshal.Cast<byte, bool>(ReadOnlySpan<byte>(bytes)).ToArray()

    [<Fact>]
    let ``the hazard this normalisation exists for is real`` () =
        let b = (weirdBools 1).[0]
        // the raw byte disagrees with itself: this is what must never reach a YANum array
        Assert.False((b = true))
        Assert.True(if b then true else false)
        Assert.Equal(2uy, (Bytes.ofValue b).[0])

    [<Fact>]
    let ``wrap normalises a 0x02 byte to 1`` () =
        let data = weirdBools 3
        let a = NDArray.wrap data
        Assert.Equal(DType.Bool, a.DType)
        for i in 0 .. 2 do
            let v = a.GetItem<bool> [| i |]
            Assert.Equal(1uy, (Bytes.ofValue v).[0])
            Assert.True((v = true))
            Assert.True(if v then true else false)
            // wrap is zero-copy, so the caller's array is normalised in place
            Assert.Equal(1uy, (Bytes.ofValue data.[i]).[0])

    [<Fact>]
    let ``ofSpan normalises a 0x02 byte to 1`` () =
        let data = weirdBools 4
        let a = NDArray.ofSpan(ReadOnlySpan<bool>(data))
        for i in 0 .. 3 do
            let v = a.GetItem<bool> [| i |]
            Assert.Equal(1uy, (Bytes.ofValue v).[0])
            Assert.True((v = true))

    [<Fact>]
    let ``ofArray ofArray2D ofArray3D scalar and fromFunction all normalise`` () =
        let one = (weirdBools 1).[0]
        let checkAll (a: NDArray) =
            for b in a.ToArray<bool>() do
                Assert.Equal(1uy, (Bytes.ofValue b).[0])
                Assert.True((b = true))
        checkAll (NDArray.ofArray (weirdBools 3))
        checkAll (NDArray.ofArray2D (Array2D.create 2 2 one))
        checkAll (NDArray.ofArray3D (Array3D.create 2 2 2 one))
        checkAll (NDArray.scalar one)
        checkAll (NDArray.fromFunction([| 4 |], Func<int[], bool>(fun _ -> one)))
        checkAll (NDArray.full([| 4 |], one))
        checkAll (NDArray.ofSeq (Seq.replicate 3 one))

    [<Fact>]
    let ``equality, branching and byte comparison all agree on a normalised array`` () =
        let a = NDArray.ofArray (weirdBools 3)
        let b = NDArray.ofArray [| true; true; true |]
        Assert.True(a.ArrayEqual b)
        Assert.Equal<byte[]>(Bytes.ofArray (b.ToArray<bool>()), Bytes.ofArray (a.ToArray<bool>()))

    [<Fact>]
    let ``bool zeros and ones hold exactly 0 and 1`` () =
        Assert.All((NDArray.zeros([| 4 |], DType.Bool)).ToArray<bool>(),
                   fun b -> Assert.Equal(0uy, (Bytes.ofValue b).[0]))
        Assert.All((NDArray.ones([| 4 |], DType.Bool)).ToArray<bool>(),
                   fun b -> Assert.Equal(1uy, (Bytes.ofValue b).[0]))
