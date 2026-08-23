namespace YANum.Core

#nowarn "9"
#nowarn "51"
// FS3390 validates XML doc parameter lists per accessor, but an F# indexer shares one doc
// comment between its getter and its setter, so no parameter list satisfies both: documenting
// the setter's value is "unknown" to the getter and omitting it is "incomplete" for the setter.
#nowarn "3390"

open System
open System.Runtime.CompilerServices
open System.Runtime.InteropServices

/// <summary>Sixteen <c>int</c> extents held inline, with no heap object.</summary>
/// <remarks>
/// <b>Both <c>[&lt;Struct&gt;]</c> and <c>[&lt;InlineArray&gt;]</c> are mandatory.</b> Omitting
/// <c>[&lt;Struct&gt;]</c> compiles, runs, appears to work, and writes 64 bytes into an 8-byte
/// object with no warning and no fault. Phase 0 hit this for real, which is why
/// <see cref="M:YANum.Core.Layout.verify"/> asserts the size at module initialisation.
/// </remarks>
[<Struct; System.Runtime.CompilerServices.InlineArray(16)>]
type ShapeBuf =
    /// <summary>
    /// The first extent, and the anchor the other fifteen are addressed from.
    /// <c>[&lt;InlineArray&gt;]</c> requires exactly one field, so this is the whole storage.
    /// Prefer the <c>Item</c> indexer.
    /// </summary>
    [<DefaultValue>]
    val mutable e0 : int

    /// <summary>Reads or writes one extent. F# has no native inline-array indexer syntax.</summary>
    /// <param name="i">Element index in <c>0 .. 15</c>. Not bounds checked.</param>
    /// <param name="v">The extent to store.</param>
    /// <returns>The extent at <paramref name="i"/>.</returns>
    member this.Item
        with get (i: int) : int = Unsafe.Add(&Unsafe.AsRef(&this).e0, i)
        and set (i: int) (v: int) = Unsafe.Add(&Unsafe.AsRef(&this).e0, i) <- v

/// <summary>Sixteen <c>nativeint</c> strides held inline, with no heap object.</summary>
/// <remarks>
/// <para><b>Both <c>[&lt;Struct&gt;]</c> and <c>[&lt;InlineArray&gt;]</c> are mandatory.</b></para>
/// <para>The element type is <c>nativeint</c>, not <c>int32</c>, on correctness grounds: the
/// outermost stride equals the product of the inner extents, so an array above 2^31 elements
/// has a stride beyond <c>Int32.MaxValue</c>. Narrowing would silently re-impose the 2 GiB cap
/// that the native storage backend exists to lift.</para>
/// </remarks>
[<Struct; System.Runtime.CompilerServices.InlineArray(16)>]
type StrideBuf =
    /// <summary>
    /// The first stride, and the anchor the other fifteen are addressed from.
    /// <c>[&lt;InlineArray&gt;]</c> requires exactly one field, so this is the whole storage.
    /// Prefer the <c>Item</c> indexer.
    /// </summary>
    [<DefaultValue>]
    val mutable e0 : nativeint

    /// <summary>Reads or writes one stride, in elements.</summary>
    /// <param name="i">Element index in <c>0 .. 15</c>. Not bounds checked.</param>
    /// <param name="v">The stride to store, in elements.</param>
    /// <returns>The stride at <paramref name="i"/>.</returns>
    member this.Item
        with get (i: int) : nativeint = Unsafe.Add(&Unsafe.AsRef(&this).e0, i)
        and set (i: int) (v: nativeint) = Unsafe.Add(&Unsafe.AsRef(&this).e0, i) <- v

/// <summary>Cached facts about an array's layout, computed once at construction.</summary>
[<System.Flags>]
type ArrayFlags =
    /// <summary>No flags set.</summary>
    | None = 0
    /// <summary>The elements are laid out row-major with no gaps.</summary>
    | CContiguous = 1
    /// <summary>The elements are laid out column-major with no gaps.</summary>
    | FContiguous = 2
    /// <summary>Elements may be written.</summary>
    | Writable = 4
    /// <summary>This array owns its storage rather than viewing someone else's.</summary>
    | OwnsData = 8
    /// <summary>The first element's address is a multiple of the dtype's item size.</summary>
    | Aligned = 16

/// <summary>
/// Everything about an array except its bytes: extents, strides, rank, offset, size, dtype and
/// cached layout flags.
/// </summary>
/// <remarks>
/// <para>
/// <b>224 bytes. Never passed by value.</b> Every internal API takes it as <c>inref</c> and
/// returns through <c>outref</c> or a field write. Phase 0 measured about 68 ns by value
/// against about 22 ns by <c>inref</c> on the hottest internal call shape in the library.
/// </para>
/// <para>
/// The field <i>declaration order</i> below differs from the specification's listing: F#
/// structs use sequential layout, and declaring <c>Rank</c> before <c>Offset</c> costs four
/// bytes of padding plus four bytes of tail padding, which lands the struct at 232 bytes.
/// Moving the two <c>int</c> fields to the end packs it to exactly the mandated 224.
/// </para>
/// </remarks>
[<Struct>]
type ArrayDesc =
    /// <summary>Extents per axis. Only the first <see cref="F:YANum.Core.ArrayDesc.Rank"/> entries are meaningful.</summary>
    val mutable Shape : ShapeBuf
    /// <summary>Strides per axis, <b>in elements</b>. May be negative or zero.</summary>
    val mutable Strides : StrideBuf
    /// <summary>Distance from the start of the storage block to element zero, <b>in elements</b>.</summary>
    val mutable Offset : nativeint
    /// <summary>Total element count: the product of the extents.</summary>
    val mutable Size : nativeint
    /// <summary>The element dtype.</summary>
    val mutable DType : DType
    /// <summary>Number of axes, from 0 to 16.</summary>
    val mutable Rank : int
    /// <summary>Cached layout facts.</summary>
    val mutable Flags : ArrayFlags

/// <summary>
/// Fail-fast layout assertions. A unit test asserting the same values is required but is not
/// sufficient: a future refactor that drops <c>[&lt;Struct&gt;]</c> from an inline array must
/// fail at process start, not only under test.
/// </summary>
module Layout =

    /// <summary>Required size of <see cref="T:YANum.Core.ShapeBuf"/> in bytes.</summary>
    [<Literal>]
    let ShapeBufSize = 64
    /// <summary>Required size of <see cref="T:YANum.Core.StrideBuf"/> in bytes.</summary>
    [<Literal>]
    let StrideBufSize = 128
    /// <summary>Required size of <see cref="T:YANum.Core.ArrayDesc"/> in bytes.</summary>
    [<Literal>]
    let ArrayDescSize = 224
    /// <summary>Required size of <see cref="T:YANum.Core.Complex32"/> in bytes.</summary>
    [<Literal>]
    let Complex32Size = 8

    let private check (name: string) (actual: int) (expected: int) =
        if actual <> expected then
            failwithf
                "YANum layout assertion failed: Unsafe.SizeOf<%s>() = %d, expected %d. \
                 The inline-array buffers MUST carry BOTH [<Struct>] and [<InlineArray(16)>]; \
                 dropping [<Struct>] compiles, runs, appears to work, and corrupts memory."
                name actual expected

    /// <summary>Verifies every size the descriptor design depends on. Throws on any mismatch.</summary>
    let verify () =
        check "ShapeBuf" (Unsafe.SizeOf<ShapeBuf>()) ShapeBufSize
        check "StrideBuf" (Unsafe.SizeOf<StrideBuf>()) StrideBufSize
        check "ArrayDesc" (Unsafe.SizeOf<ArrayDesc>()) ArrayDescSize
        check "Complex32" (Unsafe.SizeOf<Complex32>()) Complex32Size

    do verify ()

/// <summary>
/// Descriptor operations. Every one takes the descriptor by <c>inref</c> and writes its result
/// through <c>outref</c>; none of them allocates.
/// </summary>
module Desc =

    /// <summary>A writable span over the extent buffer of a descriptor.</summary>
    /// <param name="d">The descriptor.</param>
    /// <returns>A 16-element span aliasing the descriptor's inline extents.</returns>
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    let shapeSpan (d: inref<ArrayDesc>) : Span<int> =
        MemoryMarshal.CreateSpan(&Unsafe.As<ShapeBuf, int>(&Unsafe.AsRef(&d).Shape), Shape.MaxRank)

    /// <summary>A writable span over the stride buffer of a descriptor.</summary>
    /// <param name="d">The descriptor.</param>
    /// <returns>A 16-element span aliasing the descriptor's inline strides, in elements.</returns>
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    let strideSpan (d: inref<ArrayDesc>) : Span<nativeint> =
        MemoryMarshal.CreateSpan(&Unsafe.As<StrideBuf, nativeint>(&Unsafe.AsRef(&d).Strides), Shape.MaxRank)

    /// <summary>Reads one extent.</summary>
    /// <param name="d">The descriptor.</param>
    /// <param name="i">Axis index.</param>
    /// <returns>The extent of axis <paramref name="i"/>.</returns>
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    let extent (d: inref<ArrayDesc>) (i: int) : int = (shapeSpan &d).[i]

    /// <summary>Reads one stride, in elements.</summary>
    /// <param name="d">The descriptor.</param>
    /// <param name="i">Axis index.</param>
    /// <returns>The stride of axis <paramref name="i"/>.</returns>
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    let stride (d: inref<ArrayDesc>) (i: int) : nativeint = (strideSpan &d).[i]

    /// <summary>Materialises the extents as a fresh array.</summary>
    /// <param name="d">The descriptor.</param>
    /// <returns>A new array of length <c>Rank</c>.</returns>
    let shapeArray (d: inref<ArrayDesc>) : int[] =
        let n = d.Rank
        let r = Array.zeroCreate<int> n
        let s = shapeSpan &d
        for i in 0 .. n - 1 do r.[i] <- s.[i]
        r

    /// <summary>Materialises the strides, in elements, as a fresh array.</summary>
    /// <param name="d">The descriptor.</param>
    /// <returns>A new array of length <c>Rank</c>.</returns>
    let stridesArray (d: inref<ArrayDesc>) : nativeint[] =
        let n = d.Rank
        let r = Array.zeroCreate<nativeint> n
        let s = strideSpan &d
        for i in 0 .. n - 1 do r.[i] <- s.[i]
        r

    /// <summary>Whether the descriptor's layout is row-major contiguous.</summary>
    /// <param name="d">The descriptor.</param>
    /// <returns><c>true</c> when C-contiguous.</returns>
    let isCContiguous (d: inref<ArrayDesc>) : bool =
        if d.Size <= 1n then true
        else
            let sh = shapeSpan &d
            let st = strideSpan &d
            let mutable expected = 1n
            let mutable ok = true
            let mutable i = d.Rank - 1
            while ok && i >= 0 do
                let e = sh.[i]
                if e <> 1 then
                    if st.[i] <> expected then ok <- false
                    expected <- expected * nativeint e
                i <- i - 1
            ok

    /// <summary>Whether the descriptor's layout is column-major contiguous.</summary>
    /// <param name="d">The descriptor.</param>
    /// <returns><c>true</c> when F-contiguous.</returns>
    let isFContiguous (d: inref<ArrayDesc>) : bool =
        if d.Size <= 1n then true
        else
            let sh = shapeSpan &d
            let st = strideSpan &d
            let mutable expected = 1n
            let mutable ok = true
            let mutable i = 0
            while ok && i < d.Rank do
                let e = sh.[i]
                if e <> 1 then
                    if st.[i] <> expected then ok <- false
                    expected <- expected * nativeint e
                i <- i + 1
            ok

    /// <summary>Recomputes and stores the cached layout flags.</summary>
    /// <param name="d">The descriptor to update in place.</param>
    /// <param name="basePtr">Base address of the storage block, used for the alignment flag.</param>
    /// <param name="writable">Whether elements may be written.</param>
    /// <param name="ownsData">Whether the owning array owns the storage.</param>
    let refreshFlags (d: byref<ArrayDesc>) (basePtr: nativeint) (writable: bool) (ownsData: bool) : unit =
        let itemSize = d.DType.ItemSize
        let mutable f = ArrayFlags.None
        if isCContiguous &d then f <- f ||| ArrayFlags.CContiguous
        if isFContiguous &d then f <- f ||| ArrayFlags.FContiguous
        if writable then f <- f ||| ArrayFlags.Writable
        if ownsData then f <- f ||| ArrayFlags.OwnsData
        if itemSize > 0 then
            let addr = basePtr + d.Offset * nativeint itemSize
            if addr % nativeint itemSize = 0n then f <- f ||| ArrayFlags.Aligned
        d.Flags <- f

    /// <summary>Initialises a descriptor for a freshly allocated, densely packed array.</summary>
    /// <param name="d">The descriptor to fill.</param>
    /// <param name="shape">Extents; validated by the caller.</param>
    /// <param name="order">Layout order; <c>K</c> is treated as <c>C</c>.</param>
    /// <param name="dtype">Element dtype.</param>
    let initDense (d: byref<ArrayDesc>) (shape: int[]) (order: Order) (dtype: DType) : unit =
        d.Rank <- shape.Length
        d.Offset <- 0n
        d.DType <- dtype
        let sh = shapeSpan &d
        let st = strideSpan &d
        let mutable size = 1n
        for i in 0 .. shape.Length - 1 do
            sh.[i] <- shape.[i]
            size <- size * nativeint shape.[i]
        d.Size <- size
        match order with
        | F -> Shape.writeFStrides shape st
        | _ -> Shape.writeCStrides shape st

    /// <summary>Copies a descriptor without passing it by value.</summary>
    /// <param name="src">Source descriptor.</param>
    /// <param name="dst">Destination descriptor.</param>
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    let copyTo (src: inref<ArrayDesc>) (dst: byref<ArrayDesc>) : unit =
        dst <- src

    /// <summary>Element offset of a multi-dimensional index, in elements from the buffer start.</summary>
    /// <param name="d">The descriptor.</param>
    /// <param name="idx">One index per axis; negative values count from the end of that axis.</param>
    /// <param name="checkBounds">Whether to validate each index against its extent.</param>
    /// <param name="op">Operation name, used in error messages.</param>
    /// <returns>The element offset.</returns>
    /// <exception cref="T:YANum.Core.YANumShapeError">The index has the wrong number of entries.</exception>
    /// <exception cref="T:YANum.Core.YANumValueError">An index is out of range and checking is on.</exception>
    let offsetOf (d: inref<ArrayDesc>) (idx: int[]) (checkBounds: bool) (op: string) : nativeint =
        if idx.Length <> d.Rank then
            Errors.shapeError
                (sprintf "%s: %d indices supplied for an array of rank %d" op idx.Length d.Rank)
        else
            let sh = shapeSpan &d
            let st = strideSpan &d
            let mutable off = d.Offset
            let mutable k = 0
            while k < d.Rank do
                let e = sh.[k]
                let mutable i = idx.[k]
                if i < 0 then i <- i + e
                if checkBounds && (i < 0 || i >= e) then
                    Errors.indexOutOfRange op k idx.[k] e
                off <- off + nativeint i * st.[k]
                k <- k + 1
            off

    /// <summary>Reverses the axis order, producing a transposed view descriptor.</summary>
    /// <param name="src">Source descriptor.</param>
    /// <param name="dst">Destination descriptor.</param>
    let transposeInto (src: inref<ArrayDesc>) (dst: byref<ArrayDesc>) : unit =
        let n = src.Rank
        let ssh = shapeSpan &src
        let sst = strideSpan &src
        dst.Rank <- n
        dst.Offset <- src.Offset
        dst.Size <- src.Size
        dst.DType <- src.DType
        let dsh = shapeSpan &dst
        let dst' = strideSpan &dst
        for i in 0 .. n - 1 do
            dsh.[i] <- ssh.[n - 1 - i]
            dst'.[i] <- sst.[n - 1 - i]

    /// <summary>Reorders the axes by an explicit permutation.</summary>
    /// <param name="src">Source descriptor.</param>
    /// <param name="perm">A permutation of <c>0 .. rank-1</c>, already normalised.</param>
    /// <param name="dst">Destination descriptor.</param>
    let permuteInto (src: inref<ArrayDesc>) (perm: int[]) (dst: byref<ArrayDesc>) : unit =
        let n = src.Rank
        let ssh = shapeSpan &src
        let sst = strideSpan &src
        dst.Rank <- n
        dst.Offset <- src.Offset
        dst.Size <- src.Size
        dst.DType <- src.DType
        let dsh = shapeSpan &dst
        let dstr = strideSpan &dst
        for i in 0 .. n - 1 do
            dsh.[i] <- ssh.[perm.[i]]
            dstr.[i] <- sst.[perm.[i]]

    /// <summary>Inserts a length-1 axis at the given position.</summary>
    /// <param name="src">Source descriptor.</param>
    /// <param name="axis">Insertion position in <c>0 .. rank</c>.</param>
    /// <param name="dst">Destination descriptor.</param>
    let expandDimsInto (src: inref<ArrayDesc>) (axis: int) (dst: byref<ArrayDesc>) : unit =
        let n = src.Rank
        let ssh = shapeSpan &src
        let sst = strideSpan &src
        dst.Rank <- n + 1
        dst.Offset <- src.Offset
        dst.Size <- src.Size
        dst.DType <- src.DType
        let dsh = shapeSpan &dst
        let dstr = strideSpan &dst
        // Walk backwards so a shared descriptor is never overwritten before it is read.
        for i in n .. -1 .. axis + 1 do
            dsh.[i] <- ssh.[i - 1]
            dstr.[i] <- sst.[i - 1]
        dsh.[axis] <- 1
        // The stride of a length-1 axis is arbitrary; use the stride it splits so that the
        // result stays contiguous under both readings.
        dstr.[axis] <- if axis < n then sst.[axis] * nativeint ssh.[axis] else 1n
        for i in 0 .. axis - 1 do
            dsh.[i] <- ssh.[i]
            dstr.[i] <- sst.[i]

    /// <summary>Removes length-1 axes.</summary>
    /// <param name="src">Source descriptor.</param>
    /// <param name="axes">Axes to drop, already normalised and each of extent 1; null drops every length-1 axis.</param>
    /// <param name="dst">Destination descriptor.</param>
    let squeezeInto (src: inref<ArrayDesc>) (axes: int[]) (dst: byref<ArrayDesc>) : unit =
        let n = src.Rank
        let ssh = shapeSpan &src
        let sst = strideSpan &src
        let dsh = shapeSpan &dst
        let dstr = strideSpan &dst
        let mutable w = 0
        for i in 0 .. n - 1 do
            let drop =
                if isNull (box axes) then ssh.[i] = 1
                else Array.contains i axes
            if not drop then
                dsh.[w] <- ssh.[i]
                dstr.[w] <- sst.[i]
                w <- w + 1
        dst.Rank <- w
        dst.Offset <- src.Offset
        dst.Size <- src.Size
        dst.DType <- src.DType

    /// <summary>Builds a reshaped view descriptor. The caller must have confirmed a view is possible.</summary>
    /// <param name="src">Source descriptor.</param>
    /// <param name="newShape">Requested extents; the total size must match.</param>
    /// <param name="order">Reading order of the reshape.</param>
    /// <param name="dst">Destination descriptor.</param>
    let reshapeInto (src: inref<ArrayDesc>) (newShape: int[]) (order: Order) (dst: byref<ArrayDesc>) : unit =
        let effective =
            match order with
            | C -> C
            | F -> F
            | K -> if isCContiguous &src then C else F
        dst.Rank <- newShape.Length
        dst.Offset <- src.Offset
        dst.Size <- src.Size
        dst.DType <- src.DType
        let dsh = shapeSpan &dst
        let dstr = strideSpan &dst
        for i in 0 .. newShape.Length - 1 do
            dsh.[i] <- newShape.[i]
        match effective with
        | F -> Shape.writeFStrides newShape dstr
        | _ -> Shape.writeCStrides newShape dstr
