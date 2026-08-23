namespace YANum.Tests

open System.Runtime.CompilerServices
open Xunit
open YANum.Core

/// The size assertions from the specification. A failure here means the descriptor design has
/// silently changed shape.
module LayoutTests =

    [<Fact>]
    let ``ShapeBuf is 64 bytes`` () =
        Assert.Equal(64, Unsafe.SizeOf<ShapeBuf>())

    [<Fact>]
    let ``StrideBuf is 128 bytes`` () =
        Assert.Equal(128, Unsafe.SizeOf<StrideBuf>())

    [<Fact>]
    let ``ArrayDesc is 224 bytes`` () =
        Assert.Equal(224, Unsafe.SizeOf<ArrayDesc>())

    [<Fact>]
    let ``Complex32 is 8 bytes`` () =
        Assert.Equal(8, Unsafe.SizeOf<Complex32>())

    [<Fact>]
    let ``DType is 8 bytes`` () =
        Assert.Equal(8, Unsafe.SizeOf<DType>())

    /// The module initialiser in ArrayDesc.fs runs this at process start. Calling it here
    /// proves the check itself is wired up and passes, not merely that the sizes happen to be
    /// right at the moment the test reads them.
    [<Fact>]
    let ``the module-init layout check passes`` () =
        Layout.verify ()
        Complex32Layout.verify ()

    [<Fact>]
    let ``the layout constants match the specification`` () =
        Assert.Equal(64, Layout.ShapeBufSize)
        Assert.Equal(128, Layout.StrideBufSize)
        Assert.Equal(224, Layout.ArrayDescSize)
        Assert.Equal(8, Layout.Complex32Size)

    [<Fact>]
    let ``the inline buffers index without corrupting neighbours`` () =
        let mutable s = Unchecked.defaultof<ShapeBuf>
        for i in 0 .. 15 do s.[i] <- i * 7
        for i in 0 .. 15 do Assert.Equal(i * 7, s.[i])
        let mutable t = Unchecked.defaultof<StrideBuf>
        for i in 0 .. 15 do t.[i] <- nativeint (i * 1000)
        for i in 0 .. 15 do Assert.Equal(nativeint (i * 1000), t.[i])

    [<Fact>]
    let ``Complex32 is the little-endian concatenation of its components`` () =
        let c = Complex32(1.5f, -2.25f)
        let bytes = Bytes.ofValue c
        Assert.Equal(8, bytes.Length)
        Assert.Equal<byte[]>(System.BitConverter.GetBytes 1.5f, bytes.[0..3])
        Assert.Equal<byte[]>(System.BitConverter.GetBytes -2.25f, bytes.[4..7])
