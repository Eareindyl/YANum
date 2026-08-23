namespace YANum.Core

#nowarn "9"
#nowarn "51"

open System
open System.Globalization
open System.Numerics
open System.Text
open Microsoft.FSharp.NativeInterop

/// <summary>
/// Basic, NumPy-shaped rendering of an array's contents: enough to debug with.
/// </summary>
/// <remarks>
/// <para>
/// This module deliberately sits below <see cref="T:YANum.Core.NDArray"/> in the compile order
/// and works from a <see cref="T:YANum.Core.Storage"/> plus an
/// <see cref="T:YANum.Core.ArrayDesc"/>, because F# cannot add an <c>override ToString</c> to
/// a type from a later file.
/// </para>
/// <para>
/// Full NumPy summarisation, per-column width alignment and precision handling arrive in
/// Phase 5. What is here is unsummarised except for one safety net: output is truncated after
/// <see cref="P:YANum.Core.Config.PrintThreshold"/> elements with an explicit notice, so that
/// printing a large array cannot produce a gigabyte of text.
/// </para>
/// </remarks>
module Printing =

    let private inv = CultureInfo.InvariantCulture

    let inline private read<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType>
        (addr: nativeint) : 'T =
        NativePtr.read (NativePtr.ofNativeInt<'T> addr)

    let private formatFloat (v: float) =
        if Double.IsNaN v then "nan"
        elif Double.IsPositiveInfinity v then "inf"
        elif Double.IsNegativeInfinity v then "-inf"
        else
            let s = v.ToString("G" + string Config.PrintPrecision, inv)
            // NumPy renders an integral float as "1." so the dtype is visible at a glance.
            if s.IndexOf('.') < 0 && s.IndexOf('E') < 0 && s.IndexOf('e') < 0 then s + "." else s

    /// <summary>Reads the fixed-width string element at an address and trims its null padding.</summary>
    /// <param name="dtype">A <c>Utf16</c> or <c>Ascii</c> dtype.</param>
    /// <param name="addr">Address of the element.</param>
    /// <returns>The stored value, without trailing NULs.</returns>
    let readStringElement (dtype: DType) (addr: nativeint) : string =
        let w = dtype.Width
        match dtype.Code with
        | DTypeCode.Utf16 ->
            let sp = ReadOnlySpan<char>(NativePtr.toVoidPtr (NativePtr.ofNativeInt<char> addr), w)
            let mutable n = 0
            while n < w && sp.[n] <> '\000' do n <- n + 1
            String(sp.Slice(0, n))
        | _ ->
            let sp = ReadOnlySpan<byte>(NativePtr.toVoidPtr (NativePtr.ofNativeInt<byte> addr), w)
            let mutable n = 0
            while n < w && sp.[n] <> 0uy do n <- n + 1
            Encoding.ASCII.GetString(sp.Slice(0, n))

    /// <summary>Renders one element of any dtype.</summary>
    /// <param name="dtype">The element dtype.</param>
    /// <param name="addr">Address of the element.</param>
    /// <returns>The rendered value.</returns>
    let formatElement (dtype: DType) (addr: nativeint) : string =
        match dtype.Code with
        | DTypeCode.Bool -> if read<byte> addr <> 0uy then "True" else "False"
        | DTypeCode.Int8 -> (read<sbyte> addr).ToString(inv)
        | DTypeCode.UInt8 -> (read<byte> addr).ToString(inv)
        | DTypeCode.Int16 -> (read<int16> addr).ToString(inv)
        | DTypeCode.UInt16 -> (read<uint16> addr).ToString(inv)
        | DTypeCode.Int32 -> (read<int32> addr).ToString(inv)
        | DTypeCode.UInt32 -> (read<uint32> addr).ToString(inv)
        | DTypeCode.Int64 -> (read<int64> addr).ToString(inv)
        | DTypeCode.UInt64 -> (read<uint64> addr).ToString(inv)
        | DTypeCode.Int128 -> (read<Int128> addr).ToString(inv)
        | DTypeCode.UInt128 -> (read<UInt128> addr).ToString(inv)
        | DTypeCode.Float16 -> formatFloat (float (read<Half> addr))
        | DTypeCode.Float32 -> formatFloat (float (read<float32> addr))
        | DTypeCode.Float64 -> formatFloat (read<float> addr)
        | DTypeCode.Decimal -> (read<decimal> addr).ToString(inv)
        | DTypeCode.Complex64 ->
            let c = read<Complex32> addr
            "(" + formatFloat (float c.Real) + ", " + formatFloat (float c.Imaginary) + ")"
        | DTypeCode.Complex128 ->
            let c = read<Complex> addr
            "(" + formatFloat c.Real + ", " + formatFloat c.Imaginary + ")"
        | DTypeCode.Char -> "'" + string (read<char> addr) + "'"
        | DTypeCode.DateTime -> (read<DateTime> addr).ToString("O", inv)
        | DTypeCode.DateTimeOffset -> (read<DateTimeOffset> addr).ToString("O", inv)
        | DTypeCode.DateOnly -> (read<DateOnly> addr).ToString("O", inv)
        | DTypeCode.TimeOnly -> (read<TimeOnly> addr).ToString("O", inv)
        | DTypeCode.TimeSpan -> (read<TimeSpan> addr).ToString("c", inv)
        | DTypeCode.Utf16 | DTypeCode.Ascii -> "'" + readStringElement dtype addr + "'"
        | c -> Errors.typeError (sprintf "printing: dtype code %d has no formatter" (int c))

    /// <summary>The repr-style header naming shape and dtype.</summary>
    /// <param name="shape">The extents.</param>
    /// <param name="dtype">The element dtype.</param>
    /// <returns>A string of the form <c>NDArray(shape=[2;3], dtype=float64)</c>.</returns>
    let header (shape: int[]) (dtype: DType) : string =
        "NDArray(shape=[" + String.Join(";", shape) + "], dtype=" + dtype.Name + ")"

    let private renderNested (sb: StringBuilder) (basePtr: nativeint) (dtype: DType)
                             (shape: int[]) (strides: nativeint[]) (offset: nativeint) =
        let itemSize = nativeint dtype.ItemSize
        let rank = shape.Length
        let rec go (axis: int) (off: nativeint) (indent: int) =
            if axis = rank then
                sb.Append(formatElement dtype (basePtr + off * itemSize)) |> ignore
            else
                sb.Append('[') |> ignore
                let e = shape.[axis]
                let s = strides.[axis]
                for i in 0 .. e - 1 do
                    if i > 0 then
                        if axis = rank - 1 then sb.Append(' ') |> ignore
                        else
                            sb.Append('\n') |> ignore
                            if axis < rank - 2 then sb.Append('\n') |> ignore
                            sb.Append(' ', indent + 1) |> ignore
                    go (axis + 1) (off + nativeint i * s) (indent + 1)
                sb.Append(']') |> ignore
        go 0 offset 0

    let private renderFlat (sb: StringBuilder) (basePtr: nativeint) (dtype: DType)
                           (shape: int[]) (strides: nativeint[]) (offset: nativeint) (size: nativeint) =
        let itemSize = nativeint dtype.ItemSize
        let rank = shape.Length
        let limit = nativeint (max 1 Config.PrintThreshold)
        let shown = if size > limit then limit else size
        let idx = Array.zeroCreate<int> rank
        sb.Append('[') |> ignore
        let mutable off = offset
        let mutable k = 0n
        while k < shown do
            if k > 0n then sb.Append(' ') |> ignore
            sb.Append(formatElement dtype (basePtr + off * itemSize)) |> ignore
            // odometer: advance the last axis, carrying towards axis 0
            let mutable ax = rank - 1
            let mutable carrying = true
            while carrying && ax >= 0 do
                idx.[ax] <- idx.[ax] + 1
                off <- off + strides.[ax]
                if idx.[ax] < shape.[ax] then carrying <- false
                else
                    off <- off - strides.[ax] * nativeint shape.[ax]
                    idx.[ax] <- 0
                    ax <- ax - 1
            k <- k + 1n
        if shown < size then
            sb.Append(" ... (") |> ignore
            sb.Append(int64 size) |> ignore
            sb.Append(" elements total)") |> ignore
        sb.Append(']') |> ignore

    /// <summary>
    /// Renders an array as a repr header followed by its values.
    /// </summary>
    /// <param name="storage">The block holding the elements.</param>
    /// <param name="desc">The layout descriptor.</param>
    /// <returns>The rendered array.</returns>
    /// <remarks>
    /// Ranks 0 to 3 get nested bracket layouts. Rank 4 and above get a flat listing under the
    /// shape header; per-axis layout for high ranks lands in Phase 5.
    /// </remarks>
    let render (storage: Storage) (desc: inref<ArrayDesc>) : string =
        let shape = Desc.shapeArray &desc
        let strides = Desc.stridesArray &desc
        let dtype = desc.DType
        let offset = desc.Offset
        let size = desc.Size
        let basePtr = storage.Pointer
        let sb = StringBuilder()
        sb.Append(header shape dtype) |> ignore
        sb.Append('\n') |> ignore
        if size = 0n then
            sb.Append("[]") |> ignore
        elif shape.Length <= 3 && size <= nativeint (max 1 Config.PrintThreshold) then
            renderNested sb basePtr dtype shape strides offset
        else
            renderFlat sb basePtr dtype shape strides offset size
        sb.ToString()
