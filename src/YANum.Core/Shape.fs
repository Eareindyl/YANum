namespace YANum.Core

open System
open System.Runtime.CompilerServices

/// <summary>
/// Pure functions on shapes and strides. Dtype-agnostic, and allocation-free in every helper
/// that does not have to return a new array.
/// </summary>
/// <remarks>
/// <para><b>Strides are in elements, not bytes.</b> Byte strides are derived by multiplying by
/// <see cref="P:YANum.Core.DType.ItemSize"/>.</para>
/// <para>An array of size 0 or 1 — including rank 0, and any shape with a zero extent — is
/// <b>both</b> C- and F-contiguous. An axis of extent 1 imposes no constraint on contiguity
/// and its stride is arbitrary.</para>
/// </remarks>
module Shape =

    /// <summary>The maximum rank YANum supports. Fixed by the inline buffers in the descriptor.</summary>
    [<Literal>]
    let MaxRank = 16

    /// <summary>The number of elements a shape describes; 0 if any extent is 0, 1 for rank 0.</summary>
    /// <param name="shape">The extents.</param>
    /// <returns>The product of the extents.</returns>
    let size (shape: int[]) : nativeint =
        let mutable acc = 1n
        let mutable i = 0
        while i < shape.Length do
            acc <- acc * nativeint shape.[i]
            i <- i + 1
        acc

    /// <summary>Writes row-major strides, in elements, into a caller-provided buffer.</summary>
    /// <param name="shape">The extents.</param>
    /// <param name="dst">Destination of at least <c>shape.Length</c> entries.</param>
    let writeCStrides (shape: int[]) (dst: Span<nativeint>) : unit =
        let n = shape.Length
        let mutable acc = 1n
        let mutable i = n - 1
        while i >= 0 do
            dst.[i] <- acc
            acc <- acc * nativeint shape.[i]
            i <- i - 1

    /// <summary>Writes column-major strides, in elements, into a caller-provided buffer.</summary>
    /// <param name="shape">The extents.</param>
    /// <param name="dst">Destination of at least <c>shape.Length</c> entries.</param>
    let writeFStrides (shape: int[]) (dst: Span<nativeint>) : unit =
        let n = shape.Length
        let mutable acc = 1n
        let mutable i = 0
        while i < n do
            dst.[i] <- acc
            acc <- acc * nativeint shape.[i]
            i <- i + 1

    /// <summary>Row-major strides, in elements: the last axis varies fastest.</summary>
    /// <param name="shape">The extents.</param>
    /// <returns>A fresh stride array.</returns>
    let cStrides (shape: int[]) : nativeint[] =
        let r = Array.zeroCreate<nativeint> shape.Length
        writeCStrides shape (Span<nativeint>(r))
        r

    /// <summary>Column-major strides, in elements: the first axis varies fastest.</summary>
    /// <param name="shape">The extents.</param>
    /// <returns>A fresh stride array.</returns>
    let fStrides (shape: int[]) : nativeint[] =
        let r = Array.zeroCreate<nativeint> shape.Length
        writeFStrides shape (Span<nativeint>(r))
        r

    /// <summary>Strides for the given order. <c>K</c> is treated as <c>C</c>.</summary>
    /// <param name="order">The layout order.</param>
    /// <param name="shape">The extents.</param>
    /// <returns>A fresh stride array.</returns>
    let stridesFor (order: Order) (shape: int[]) : nativeint[] =
        match order with
        | F -> fStrides shape
        | _ -> cStrides shape

    /// <summary>Whether the strides describe a row-major contiguous layout.</summary>
    /// <param name="shape">The extents.</param>
    /// <param name="strides">Strides in elements.</param>
    /// <returns><c>true</c> when the layout is C-contiguous.</returns>
    let isCContiguous (shape: int[]) (strides: nativeint[]) : bool =
        if size shape <= 1n then true
        else
            let mutable expected = 1n
            let mutable ok = true
            let mutable i = shape.Length - 1
            while ok && i >= 0 do
                let e = shape.[i]
                if e <> 1 then
                    if strides.[i] <> expected then ok <- false
                    expected <- expected * nativeint e
                i <- i - 1
            ok

    /// <summary>Whether the strides describe a column-major contiguous layout.</summary>
    /// <param name="shape">The extents.</param>
    /// <param name="strides">Strides in elements.</param>
    /// <returns><c>true</c> when the layout is F-contiguous.</returns>
    let isFContiguous (shape: int[]) (strides: nativeint[]) : bool =
        if size shape <= 1n then true
        else
            let mutable expected = 1n
            let mutable ok = true
            let mutable i = 0
            while ok && i < shape.Length do
                let e = shape.[i]
                if e <> 1 then
                    if strides.[i] <> expected then ok <- false
                    expected <- expected * nativeint e
                i <- i + 1
            ok

    /// <summary>Whether two shapes are equal.</summary>
    /// <param name="a">First shape.</param>
    /// <param name="b">Second shape.</param>
    /// <returns><c>true</c> when the ranks and every extent match.</returns>
    let equal (a: int[]) (b: int[]) : bool =
        if a.Length <> b.Length then false
        else
            let mutable ok = true
            let mutable i = 0
            while ok && i < a.Length do
                if a.[i] <> b.[i] then ok <- false
                i <- i + 1
            ok

    /// <summary>
    /// Whether a reshape can be served as a view rather than a copy.
    /// </summary>
    /// <param name="shape">Current extents.</param>
    /// <param name="strides">Current strides, in elements.</param>
    /// <param name="newShape">Requested extents; must have the same total size.</param>
    /// <param name="order">The reading order the reshape should follow.</param>
    /// <returns><c>true</c> when a stride-only view exists.</returns>
    /// <remarks>
    /// Conservative but never wrong: a <c>true</c> result guarantees a view exists. Some
    /// non-contiguous layouts admit a view for particular target shapes; those are reported
    /// as <c>false</c> and served by a copy.
    /// </remarks>
    let canReshape (shape: int[]) (strides: nativeint[]) (newShape: int[]) (order: Order) : bool =
        if size shape <> size newShape then false
        elif size newShape <= 1n then true
        else
            match order with
            | C -> isCContiguous shape strides
            | F -> isFContiguous shape strides
            | K -> isCContiguous shape strides || isFContiguous shape strides

    /// <summary>Turns a possibly-negative axis into a non-negative one.</summary>
    /// <param name="rank">The rank the axis indexes into.</param>
    /// <param name="axis">The axis; negative values count from the end.</param>
    /// <returns>The axis in <c>0 .. rank-1</c>.</returns>
    /// <exception cref="T:YANum.Core.YANumShapeError">The axis is out of range for the rank.</exception>
    let normalizeAxis (rank: int) (axis: int) : int =
        let a = if axis < 0 then axis + rank else axis
        if a < 0 || a >= rank then
            Errors.shapeError
                (sprintf "axis %d is out of range for an array of rank %d (valid axes are %d..%d)"
                    axis rank (-rank) (rank - 1))
        else a

    /// <summary>Validates a shape.</summary>
    /// <param name="shape">The extents.</param>
    /// <exception cref="T:YANum.Core.YANumShapeError">The shape is null, exceeds rank 16, or has a negative extent.</exception>
    let validate (shape: int[]) : unit =
        if isNull (box shape) then Errors.shapeError "shape is null"
        elif shape.Length > MaxRank then
            Errors.shapeError
                (sprintf "rank %d exceeds YANum's maximum rank of %d for shape %s"
                    shape.Length MaxRank (Errors.formatShape shape))
        else
            let mutable i = 0
            while i < shape.Length do
                if shape.[i] < 0 then
                    Errors.shapeError
                        (sprintf "extent %d on axis %d of shape %s is negative; zero extents are legal, negative ones are not"
                            shape.[i] i (Errors.formatShape shape))
                i <- i + 1
