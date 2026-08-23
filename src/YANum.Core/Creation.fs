namespace YANum.Core

#nowarn "9"
#nowarn "51"

open System
open System.Numerics
open System.Runtime.CompilerServices
open System.Runtime.InteropServices
open Microsoft.FSharp.NativeInterop

/// <summary>Scalar writers used by the creation functions that produce values of any dtype.</summary>
module internal Fill =

    let inline private wr<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType>
        (addr: nativeint) (v: 'T) : unit = NativePtr.write (NativePtr.ofNativeInt<'T> addr) v

    /// Writes one element of any dtype from a double, converting as the dtype requires.
    /// Temporal dtypes take the value as their natural integral unit; string dtypes take its
    /// invariant text rendering.
    let writeFloat (dtype: DType) (addr: nativeint) (v: float) : unit =
        match dtype.Code with
        | DTypeCode.Bool -> wr<byte> addr (if v <> 0.0 then 1uy else 0uy)
        | DTypeCode.Int8 -> wr<sbyte> addr (sbyte v)
        | DTypeCode.UInt8 -> wr<byte> addr (byte v)
        | DTypeCode.Int16 -> wr<int16> addr (int16 v)
        | DTypeCode.UInt16 -> wr<uint16> addr (uint16 v)
        | DTypeCode.Int32 -> wr<int32> addr (int32 v)
        | DTypeCode.UInt32 -> wr<uint32> addr (uint32 v)
        | DTypeCode.Int64 -> wr<int64> addr (int64 v)
        | DTypeCode.UInt64 -> wr<uint64> addr (uint64 v)
        | DTypeCode.Int128 -> wr<Int128> addr (Int128.op_Explicit v)
        | DTypeCode.UInt128 -> wr<UInt128> addr (UInt128.op_Explicit v)
        | DTypeCode.Float16 -> wr<Half> addr (Half.op_Explicit v)
        | DTypeCode.Float32 -> wr<float32> addr (float32 v)
        | DTypeCode.Float64 -> wr<float> addr v
        | DTypeCode.Decimal -> wr<decimal> addr (decimal v)
        | DTypeCode.Complex64 -> wr<Complex32> addr (Complex32(float32 v, 0.0f))
        | DTypeCode.Complex128 -> wr<Complex> addr (Complex(v, 0.0))
        | DTypeCode.Char -> wr<char> addr (char (int v))
        // Temporal dtypes take the number as their natural integral unit: ticks for the
        // tick-based types, the day number for DateOnly. This is what makes `arange`, `eye`
        // and `ones` meaningful on a timeline.
        | DTypeCode.DateTime -> wr<DateTime> addr (DateTime(int64 v))
        | DTypeCode.DateTimeOffset -> wr<DateTimeOffset> addr (DateTimeOffset(DateTime(int64 v, DateTimeKind.Utc)))
        | DTypeCode.DateOnly -> wr<DateOnly> addr (DateOnly.FromDayNumber(int v))
        | DTypeCode.TimeOnly -> wr<TimeOnly> addr (TimeOnly(int64 v))
        | DTypeCode.TimeSpan -> wr<TimeSpan> addr (TimeSpan(int64 v))
        // String dtypes take the invariant rendering of the number, NUL-padded and truncated
        // to the dtype width.
        | DTypeCode.Utf16 ->
            let s = v.ToString(Globalization.CultureInfo.InvariantCulture)
            let w = dtype.Width
            let dst = Span<char>(NativePtr.toVoidPtr (NativePtr.ofNativeInt<char> addr), w)
            dst.Clear()
            s.AsSpan(0, min s.Length w).CopyTo(dst)
        | DTypeCode.Ascii ->
            let s = v.ToString(Globalization.CultureInfo.InvariantCulture)
            let w = dtype.Width
            let dst = Span<byte>(NativePtr.toVoidPtr (NativePtr.ofNativeInt<byte> addr), w)
            dst.Clear()
            Text.Encoding.ASCII.GetBytes(s.AsSpan(0, min s.Length w), dst) |> ignore
        | c ->
            Errors.typeError
                (sprintf "creation: dtype code %d has no scalar writer" (int c))

/// <summary>
/// Array construction. Every function takes an optional <c>order</c> defaulting to
/// <see cref="P:YANum.Core.Config.DefaultOrder"/> and an optional <c>backend</c> defaulting to
/// <see cref="P:YANum.Core.Config.DefaultStorage"/>.
/// </summary>
/// <remarks>
/// These are static members on <see cref="T:YANum.Core.NDArray"/> rather than functions in a
/// module, because F# permits optional parameters only on type members. The call shape
/// (<c>NDArray.zeros [| 2; 3 |]</c>) is unchanged.
/// </remarks>
[<AutoOpen>]
module NDArrayCreation =

    let inline private orderOf (order: Order option) = defaultArg order Config.DefaultOrder
    let inline private backendOf (backend: StorageBackend option) = defaultArg backend Config.DefaultStorage

    let internal newDense (shape: int[]) (dtype: DType) (order: Order) (backend: StorageBackend) (zeroed: bool) : NDArray =
        Shape.validate shape
        if dtype.ItemSize <= 0 then
            Errors.valueError (sprintf "creation: dtype %s has a zero item size" dtype.Name)
        let size = Shape.size shape
        let bytes = unativeint size * unativeint dtype.ItemSize
        let storage =
            if zeroed then Storage.AllocateZeroed(bytes, backend)
            else Storage.AllocateUninitialized(bytes, backend)
        let mutable d = Unchecked.defaultof<ArrayDesc>
        Desc.initDense &d shape order dtype
        NDArray.Build(storage, &d, null)

    /// Copies a contiguous run of 'T into a fresh dense array, normalising Bool bytes.
    let inline internal ofContiguous<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType>
        (src: ReadOnlySpan<'T>) (shape: int[]) (order: Order) (backend: StorageBackend) : NDArray =
        let dtype = DTypeOf<'T>.Value
        let a = newDense shape dtype order backend false
        if src.Length > 0 then
            let bytes = int64 src.Length * int64 (Unsafe.SizeOf<'T>())
            use p = fixed src
            Buffer.MemoryCopy(NativePtr.toVoidPtr p, a.Storage.Pointer.ToPointer(), bytes, bytes)
        if dtype.Code = DTypeCode.Bool then
            Elem.normalizeBools a.Storage.Pointer (nativeint src.Length)
        a

    type NDArray with

        // -------------------------------------------------------------------------------
        //  From existing data
        // -------------------------------------------------------------------------------

        /// <summary>A rank-1 array copied from a .NET array.</summary>
        /// <param name="data">The source elements.</param>
        /// <param name="order">Layout order; immaterial at rank 1.</param>
        /// <param name="backend">Allocator to use.</param>
        /// <returns>A new array owning its data.</returns>
        static member ofArray<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType>
            (data: 'T[], ?order: Order, ?backend: StorageBackend) : NDArray =
            if isNull (box data) then Errors.valueError "ofArray: the source array is null"
            ofContiguous (ReadOnlySpan<'T>(data)) [| data.Length |] (orderOf order) (backendOf backend)

        /// <summary>A rank-1 array copied from a span.</summary>
        /// <param name="data">The source elements.</param>
        /// <param name="order">Layout order; immaterial at rank 1.</param>
        /// <param name="backend">Allocator to use.</param>
        /// <returns>A new array owning its data.</returns>
        /// <remarks>Bool bytes are normalised to 0 or 1 on the way in.</remarks>
        static member ofSpan<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType>
            (data: ReadOnlySpan<'T>, ?order: Order, ?backend: StorageBackend) : NDArray =
            ofContiguous data [| data.Length |] (orderOf order) (backendOf backend)

        /// <summary>A rank-2 array copied from a .NET rectangular array.</summary>
        /// <param name="data">The source elements.</param>
        /// <param name="order">Layout order of the result.</param>
        /// <param name="backend">Allocator to use.</param>
        /// <returns>A new array owning its data.</returns>
        static member ofArray2D<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType>
            (data: 'T[,], ?order: Order, ?backend: StorageBackend) : NDArray =
            if isNull (box data) then Errors.valueError "ofArray2D: the source array is null"
            let shape = [| Array2D.length1 data; Array2D.length2 data |]
            let dtype = DTypeOf<'T>.Value
            let a = newDense shape dtype (orderOf order) (backendOf backend) false
            let size = Shape.size shape
            if size > 0n then
                let handle = GCHandle.Alloc(data, GCHandleType.Pinned)
                try
                    Elem.copyStrided (handle.AddrOfPinnedObject()) 0n (Shape.cStrides shape)
                                     a.Storage.Pointer 0n a.Strides shape size dtype.ItemSize
                finally handle.Free()
                if dtype.Code = DTypeCode.Bool then Elem.normalizeBools a.Storage.Pointer size
            a

        /// <summary>A rank-3 array copied from a .NET rectangular array.</summary>
        /// <param name="data">The source elements.</param>
        /// <param name="order">Layout order of the result.</param>
        /// <param name="backend">Allocator to use.</param>
        /// <returns>A new array owning its data.</returns>
        static member ofArray3D<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType>
            (data: 'T[,,], ?order: Order, ?backend: StorageBackend) : NDArray =
            if isNull (box data) then Errors.valueError "ofArray3D: the source array is null"
            let shape = [| Array3D.length1 data; Array3D.length2 data; Array3D.length3 data |]
            let dtype = DTypeOf<'T>.Value
            let a = newDense shape dtype (orderOf order) (backendOf backend) false
            let size = Shape.size shape
            if size > 0n then
                let handle = GCHandle.Alloc(data, GCHandleType.Pinned)
                try
                    Elem.copyStrided (handle.AddrOfPinnedObject()) 0n (Shape.cStrides shape)
                                     a.Storage.Pointer 0n a.Strides shape size dtype.ItemSize
                finally handle.Free()
                if dtype.Code = DTypeCode.Bool then Elem.normalizeBools a.Storage.Pointer size
            a

        /// <summary>A rank-1 array copied from a sequence.</summary>
        /// <param name="data">The source elements; enumerated exactly once.</param>
        /// <param name="order">Layout order; immaterial at rank 1.</param>
        /// <param name="backend">Allocator to use.</param>
        /// <returns>A new array owning its data.</returns>
        static member ofSeq<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType>
            (data: seq<'T>, ?order: Order, ?backend: StorageBackend) : NDArray =
            if isNull (box data) then Errors.valueError "ofSeq: the source sequence is null"
            let arr = Seq.toArray data
            ofContiguous (ReadOnlySpan<'T>(arr)) [| arr.Length |] (orderOf order) (backendOf backend)

        /// <summary>
        /// A rank-1 array aliasing a caller's array without copying.
        /// </summary>
        /// <param name="data">The array to alias. It is pinned for the lifetime of the result.</param>
        /// <returns>A new array whose storage is the caller's array.</returns>
        /// <remarks>
        /// <b>Writes through the array are visible to the caller and vice versa</b>, and the
        /// array stays pinned until the result's storage is released. For a <c>bool[]</c> the
        /// bytes are normalised to 0 or 1 <b>in place</b>, which is observable in the caller's
        /// array — the only way to honour the Bool invariant without copying.
        /// </remarks>
        static member wrap<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType>
            (data: 'T[]) : NDArray =
            if isNull (box data) then Errors.valueError "wrap: the source array is null"
            let dtype = DTypeOf<'T>.Value
            let storage = Storage.PinArray<'T> data
            let mutable d = Unchecked.defaultof<ArrayDesc>
            Desc.initDense &d [| data.Length |] C dtype
            if dtype.Code = DTypeCode.Bool then
                Elem.normalizeBools storage.Pointer (nativeint data.Length)
            NDArray.Build(storage, &d, null)

        /// <summary>A rank-1 fixed-width string array.</summary>
        /// <param name="data">The source strings; <c>null</c> entries store as empty.</param>
        /// <param name="width">Element width; defaults to the longest string, or 1 if all are empty.</param>
        /// <param name="ascii">When set, uses <c>ascii&lt;n&gt;</c> instead of <c>utf16&lt;n&gt;</c>.</param>
        /// <param name="order">Layout order; immaterial at rank 1.</param>
        /// <param name="backend">Allocator to use.</param>
        /// <returns>A new array owning its data.</returns>
        /// <exception cref="T:YANum.Core.YANumOverflowError">
        /// A string is longer than <paramref name="width"/> and
        /// <see cref="P:YANum.Core.Config.StringOverflow"/> is <c>Throw</c>.
        /// </exception>
        static member ofStrings (data: string[], ?width: int, ?ascii: bool, ?order: Order, ?backend: StorageBackend) : NDArray =
            if isNull (box data) then Errors.valueError "ofStrings: the source array is null"
            let longest =
                let mutable m = 0
                for s in data do
                    if not (isNull s) && s.Length > m then m <- s.Length
                max m 1
            let w = defaultArg width longest
            let dtype = if defaultArg ascii false then DType.Ascii w else DType.Utf16 w
            let a = newDense [| data.Length |] dtype (orderOf order) (backendOf backend) true
            let idx = [| 0 |]
            for i in 0 .. data.Length - 1 do
                idx.[0] <- i
                a.SetString(idx, data.[i])
            a

        /// <summary>A rank-0 array holding a single value.</summary>
        /// <param name="value">The value.</param>
        /// <param name="backend">Allocator to use.</param>
        /// <returns>A new rank-0 array owning its data.</returns>
        static member scalar<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType>
            (value: 'T, ?backend: StorageBackend) : NDArray =
            let dtype = DTypeOf<'T>.Value
            let a = newDense [||] dtype C (backendOf backend) false
            NativePtr.write (a.Storage.PointerAs<'T> 0n) value
            if dtype.Code = DTypeCode.Bool then Elem.normalizeBools a.Storage.Pointer 1n
            a

        // -------------------------------------------------------------------------------
        //  Filled and patterned
        // -------------------------------------------------------------------------------

        /// <summary>An array of zeros.</summary>
        /// <param name="shape">The extents. A zero extent is legal and yields a valid empty array.</param>
        /// <param name="dtype">Element dtype; defaults to <see cref="P:YANum.Core.Config.DefaultFloat"/>.</param>
        /// <param name="order">Layout order.</param>
        /// <param name="backend">Allocator to use.</param>
        /// <returns>A new array owning its data.</returns>
        static member zeros (shape: int[], ?dtype: DType, ?order: Order, ?backend: StorageBackend) : NDArray =
            newDense shape (defaultArg dtype Config.DefaultFloat) (orderOf order) (backendOf backend) true

        /// <summary>An array whose contents are undefined.</summary>
        /// <param name="shape">The extents.</param>
        /// <param name="dtype">Element dtype; defaults to <see cref="P:YANum.Core.Config.DefaultFloat"/>.</param>
        /// <param name="order">Layout order.</param>
        /// <param name="backend">Allocator to use.</param>
        /// <returns>A new array owning its data. Read only after writing every element.</returns>
        static member empty (shape: int[], ?dtype: DType, ?order: Order, ?backend: StorageBackend) : NDArray =
            newDense shape (defaultArg dtype Config.DefaultFloat) (orderOf order) (backendOf backend) false

        /// <summary>An array of ones.</summary>
        /// <param name="shape">The extents.</param>
        /// <param name="dtype">Element dtype; defaults to <see cref="P:YANum.Core.Config.DefaultFloat"/>.</param>
        /// <param name="order">Layout order.</param>
        /// <param name="backend">Allocator to use.</param>
        /// <returns>A new array owning its data.</returns>
        /// <exception cref="T:YANum.Core.YANumTypeError">The dtype is temporal or a string, where "one" has no meaning.</exception>
        static member ones (shape: int[], ?dtype: DType, ?order: Order, ?backend: StorageBackend) : NDArray =
            let dt = defaultArg dtype Config.DefaultFloat
            let a = newDense shape dt (orderOf order) (backendOf backend) false
            let isz = nativeint dt.ItemSize
            let n = nativeint a.Size
            let basePtr = a.Storage.Pointer
            let mutable i = 0n
            while i < n do
                Fill.writeFloat dt (basePtr + i * isz) 1.0
                i <- i + 1n
            a

        /// <summary>An array filled with one value.</summary>
        /// <param name="shape">The extents.</param>
        /// <param name="value">The value; its type determines the dtype.</param>
        /// <param name="order">Layout order.</param>
        /// <param name="backend">Allocator to use.</param>
        /// <returns>A new array owning its data.</returns>
        static member full<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType>
            (shape: int[], value: 'T, ?order: Order, ?backend: StorageBackend) : NDArray =
            let dtype = DTypeOf<'T>.Value
            let a = newDense shape dtype (orderOf order) (backendOf backend) false
            let n = nativeint a.Size
            let mutable p = a.Storage.PointerAs<'T> 0n
            let mutable i = 0n
            while i < n do
                NativePtr.write p value
                p <- NativePtr.add p 1
                i <- i + 1n
            if dtype.Code = DTypeCode.Bool then Elem.normalizeBools a.Storage.Pointer n
            a

        /// <summary>Zeros with the shape of an existing array.</summary>
        /// <param name="prototype">The array whose shape and dtype to copy.</param>
        /// <param name="dtype">Element dtype; defaults to the prototype's.</param>
        /// <param name="order">Layout order.</param>
        /// <param name="backend">Allocator to use.</param>
        /// <returns>A new array owning its data.</returns>
        static member zerosLike (prototype: NDArray, ?dtype: DType, ?order: Order, ?backend: StorageBackend) : NDArray =
            newDense prototype.Shape (defaultArg dtype prototype.DType) (orderOf order) (backendOf backend) true

        /// <summary>Ones with the shape of an existing array.</summary>
        /// <param name="prototype">The array whose shape and dtype to copy.</param>
        /// <param name="dtype">Element dtype; defaults to the prototype's.</param>
        /// <param name="order">Layout order.</param>
        /// <param name="backend">Allocator to use.</param>
        /// <returns>A new array owning its data.</returns>
        static member onesLike (prototype: NDArray, ?dtype: DType, ?order: Order, ?backend: StorageBackend) : NDArray =
            NDArray.ones(prototype.Shape, defaultArg dtype prototype.DType, orderOf order, backendOf backend)

        /// <summary>An uninitialised array with the shape of an existing array.</summary>
        /// <param name="prototype">The array whose shape and dtype to copy.</param>
        /// <param name="dtype">Element dtype; defaults to the prototype's.</param>
        /// <param name="order">Layout order.</param>
        /// <param name="backend">Allocator to use.</param>
        /// <returns>A new array owning its data.</returns>
        static member emptyLike (prototype: NDArray, ?dtype: DType, ?order: Order, ?backend: StorageBackend) : NDArray =
            newDense prototype.Shape (defaultArg dtype prototype.DType) (orderOf order) (backendOf backend) false

        /// <summary>A filled array with the shape of an existing array.</summary>
        /// <param name="prototype">The array whose shape to copy.</param>
        /// <param name="value">The value; its type determines the dtype.</param>
        /// <param name="order">Layout order.</param>
        /// <param name="backend">Allocator to use.</param>
        /// <returns>A new array owning its data.</returns>
        static member fullLike<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType>
            (prototype: NDArray, value: 'T, ?order: Order, ?backend: StorageBackend) : NDArray =
            NDArray.full(prototype.Shape, value, orderOf order, backendOf backend)

        /// <summary>A rank-2 array with ones on a diagonal and zeros elsewhere.</summary>
        /// <param name="n">Number of rows.</param>
        /// <param name="m">Number of columns; defaults to <paramref name="n"/>.</param>
        /// <param name="k">Diagonal offset: 0 is the main diagonal, positive is above it, negative below.</param>
        /// <param name="dtype">Element dtype; defaults to <see cref="P:YANum.Core.Config.DefaultFloat"/>.</param>
        /// <param name="order">Layout order.</param>
        /// <param name="backend">Allocator to use.</param>
        /// <returns>A new array owning its data.</returns>
        static member eye (n: int, ?m: int, ?k: int, ?dtype: DType, ?order: Order, ?backend: StorageBackend) : NDArray =
            let cols = defaultArg m n
            let off = defaultArg k 0
            let dt = defaultArg dtype Config.DefaultFloat
            let a = NDArray.zeros([| n; cols |], dt, orderOf order, backendOf backend)
            let isz = nativeint dt.ItemSize
            let strides = a.Strides
            let basePtr = a.Storage.Pointer
            for i in 0 .. n - 1 do
                let j = i + off
                if j >= 0 && j < cols then
                    let e = nativeint i * strides.[0] + nativeint j * strides.[1]
                    Fill.writeFloat dt (basePtr + e * isz) 1.0
            a

        /// <summary>The square identity matrix.</summary>
        /// <param name="n">Number of rows and columns.</param>
        /// <param name="dtype">Element dtype; defaults to <see cref="P:YANum.Core.Config.DefaultFloat"/>.</param>
        /// <param name="order">Layout order.</param>
        /// <param name="backend">Allocator to use.</param>
        /// <returns>A new array owning its data.</returns>
        static member identity (n: int, ?dtype: DType, ?order: Order, ?backend: StorageBackend) : NDArray =
            NDArray.eye(n, n, 0, defaultArg dtype Config.DefaultFloat, orderOf order, backendOf backend)

        // -------------------------------------------------------------------------------
        //  Ranges
        // -------------------------------------------------------------------------------

        /// <summary>Evenly spaced values over a half-open interval.</summary>
        /// <param name="start">First value.</param>
        /// <param name="stop">Exclusive upper bound.</param>
        /// <param name="step">Spacing; defaults to 1 and must not be 0.</param>
        /// <param name="dtype">Element dtype; defaults to <see cref="P:YANum.Core.Config.DefaultInt"/> when every argument is integral, otherwise <see cref="P:YANum.Core.Config.DefaultFloat"/>.</param>
        /// <param name="order">Layout order; immaterial at rank 1.</param>
        /// <param name="backend">Allocator to use.</param>
        /// <returns>A rank-1 array of <c>ceil((stop - start) / step)</c> values.</returns>
        /// <remarks>
        /// Element <c>i</c> is computed as <c>start + i * step</c>, never by repeated addition,
        /// so rounding error does not accumulate.
        /// </remarks>
        /// <exception cref="T:YANum.Core.YANumValueError">The step is zero or not finite.</exception>
        static member arange (start: float, stop: float, ?step: float, ?dtype: DType, ?order: Order, ?backend: StorageBackend) : NDArray =
            let st = defaultArg step 1.0
            if st = 0.0 then Errors.valueError "arange: step must not be zero"
            if Double.IsNaN st || Double.IsInfinity st then
                Errors.valueError (sprintf "arange: step %g is not a finite number" st)
            let integral =
                Double.IsInteger start && Double.IsInteger stop && Double.IsInteger st
            let dt = defaultArg dtype (if integral then Config.DefaultInt else Config.DefaultFloat)
            let raw = Math.Ceiling((stop - start) / st)
            let count = if Double.IsNaN raw || raw <= 0.0 then 0 else int raw
            let a = newDense [| count |] dt (orderOf order) (backendOf backend) false
            let isz = nativeint dt.ItemSize
            let basePtr = a.Storage.Pointer
            let mutable i = 0
            while i < count do
                Fill.writeFloat dt (basePtr + nativeint i * isz) (start + float i * st)
                i <- i + 1
            a

        /// <summary>Evenly spaced values from zero to an exclusive upper bound, in steps of one.</summary>
        /// <param name="stop">Exclusive upper bound.</param>
        /// <param name="dtype">Element dtype; defaults as for the three-argument form.</param>
        /// <param name="order">Layout order; immaterial at rank 1.</param>
        /// <param name="backend">Allocator to use.</param>
        /// <returns>A rank-1 array of <c>ceil(stop)</c> values.</returns>
        static member arange (stop: float, ?dtype: DType, ?order: Order, ?backend: StorageBackend) : NDArray =
            NDArray.arange(0.0, stop, 1.0, defaultArg dtype (if Double.IsInteger stop then Config.DefaultInt else Config.DefaultFloat),
                           orderOf order, backendOf backend)

        /// <summary>Evenly spaced values over a closed or half-open interval.</summary>
        /// <param name="start">First value.</param>
        /// <param name="stop">Last value when <paramref name="endpoint"/> is set, otherwise the exclusive bound.</param>
        /// <param name="num">Number of values; must not be negative.</param>
        /// <param name="endpoint">Whether <paramref name="stop"/> is included. Defaults to <c>true</c>.</param>
        /// <param name="dtype">Element dtype; defaults to <see cref="P:YANum.Core.Config.DefaultFloat"/>.</param>
        /// <param name="order">Layout order; immaterial at rank 1.</param>
        /// <param name="backend">Allocator to use.</param>
        /// <returns>A rank-1 array of <paramref name="num"/> values.</returns>
        /// <remarks>When <paramref name="endpoint"/> is set the last element is exactly <paramref name="stop"/>.</remarks>
        /// <exception cref="T:YANum.Core.YANumValueError">The count is negative.</exception>
        static member linspace (start: float, stop: float, num: int, ?endpoint: bool, ?dtype: DType, ?order: Order, ?backend: StorageBackend) : NDArray =
            if num < 0 then Errors.valueError (sprintf "linspace: num must not be negative, got %d" num)
            let inclusive = defaultArg endpoint true
            let dt = defaultArg dtype Config.DefaultFloat
            let a = newDense [| num |] dt (orderOf order) (backendOf backend) false
            let isz = nativeint dt.ItemSize
            let basePtr = a.Storage.Pointer
            let div = if inclusive then num - 1 else num
            let mutable i = 0
            while i < num do
                let v =
                    if div <= 0 then start
                    elif inclusive && i = num - 1 then stop
                    else start + (stop - start) * (float i / float div)
                Fill.writeFloat dt (basePtr + nativeint i * isz) v
                i <- i + 1
            a

        /// <summary>Values evenly spaced on a logarithmic scale.</summary>
        /// <param name="start">Exponent of the first value.</param>
        /// <param name="stop">Exponent of the last value when <paramref name="endpoint"/> is set.</param>
        /// <param name="num">Number of values.</param>
        /// <param name="numBase">The base; defaults to 10.</param>
        /// <param name="endpoint">Whether <paramref name="stop"/> is included. Defaults to <c>true</c>.</param>
        /// <param name="dtype">Element dtype; defaults to <see cref="P:YANum.Core.Config.DefaultFloat"/>.</param>
        /// <param name="order">Layout order; immaterial at rank 1.</param>
        /// <param name="backend">Allocator to use.</param>
        /// <returns>A rank-1 array of <paramref name="num"/> values.</returns>
        static member logspace (start: float, stop: float, num: int, ?numBase: float, ?endpoint: bool, ?dtype: DType, ?order: Order, ?backend: StorageBackend) : NDArray =
            if num < 0 then Errors.valueError (sprintf "logspace: num must not be negative, got %d" num)
            let b = defaultArg numBase 10.0
            let inclusive = defaultArg endpoint true
            let dt = defaultArg dtype Config.DefaultFloat
            let a = newDense [| num |] dt (orderOf order) (backendOf backend) false
            let isz = nativeint dt.ItemSize
            let basePtr = a.Storage.Pointer
            let div = if inclusive then num - 1 else num
            let mutable i = 0
            while i < num do
                let e =
                    if div <= 0 then start
                    elif inclusive && i = num - 1 then stop
                    else start + (stop - start) * (float i / float div)
                Fill.writeFloat dt (basePtr + nativeint i * isz) (Math.Pow(b, e))
                i <- i + 1
            a

        /// <summary>Values evenly spaced on a geometric progression.</summary>
        /// <param name="start">First value; must not be zero.</param>
        /// <param name="stop">Last value when <paramref name="endpoint"/> is set; must not be zero and must share the sign of <paramref name="start"/>.</param>
        /// <param name="num">Number of values.</param>
        /// <param name="endpoint">Whether <paramref name="stop"/> is included. Defaults to <c>true</c>.</param>
        /// <param name="dtype">Element dtype; defaults to <see cref="P:YANum.Core.Config.DefaultFloat"/>.</param>
        /// <param name="order">Layout order; immaterial at rank 1.</param>
        /// <param name="backend">Allocator to use.</param>
        /// <returns>A rank-1 array of <paramref name="num"/> values.</returns>
        /// <exception cref="T:YANum.Core.YANumValueError">An endpoint is zero, or the endpoints have opposite signs.</exception>
        static member geomspace (start: float, stop: float, num: int, ?endpoint: bool, ?dtype: DType, ?order: Order, ?backend: StorageBackend) : NDArray =
            if num < 0 then Errors.valueError (sprintf "geomspace: num must not be negative, got %d" num)
            if start = 0.0 || stop = 0.0 then
                Errors.valueError "geomspace: neither endpoint may be zero"
            if (start < 0.0) <> (stop < 0.0) then
                Errors.valueError (sprintf "geomspace: endpoints %g and %g have opposite signs" start stop)
            let inclusive = defaultArg endpoint true
            let dt = defaultArg dtype Config.DefaultFloat
            let a = newDense [| num |] dt (orderOf order) (backendOf backend) false
            let isz = nativeint dt.ItemSize
            let basePtr = a.Storage.Pointer
            let div = if inclusive then num - 1 else num
            let sign = if start < 0.0 then -1.0 else 1.0
            let ls = Math.Log(abs start)
            let le = Math.Log(abs stop)
            let mutable i = 0
            while i < num do
                let v =
                    if div <= 0 then start
                    elif inclusive && i = num - 1 then stop
                    else sign * Math.Exp(ls + (le - ls) * (float i / float div))
                Fill.writeFloat dt (basePtr + nativeint i * isz) v
                i <- i + 1
            a

        /// <summary>An array whose elements are computed from their indices.</summary>
        /// <param name="shape">The extents.</param>
        /// <param name="f">Called once per element with that element's index.</param>
        /// <param name="order">Layout order.</param>
        /// <param name="backend">Allocator to use.</param>
        /// <returns>A new array owning its data.</returns>
        /// <remarks>
        /// <b>The index array handed to <paramref name="f"/> is reused across calls</b> so that
        /// the loop allocates nothing. Copy it if you need to keep it.
        /// </remarks>
        static member fromFunction<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType>
            (shape: int[], f: Func<int[], 'T>, ?order: Order, ?backend: StorageBackend) : NDArray =
            if isNull (box f) then Errors.valueError "fromFunction: the generator is null"
            let dtype = DTypeOf<'T>.Value
            let a = newDense shape dtype (orderOf order) (backendOf backend) false
            let size = a.Size
            if size > 0L then
                let rank = shape.Length
                let strides = a.Strides
                let idx = Array.zeroCreate<int> rank
                let mutable off = 0n
                let mutable k = 0L
                while k < size do
                    NativePtr.write (a.Storage.PointerAs<'T> off) (f.Invoke idx)
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
                    k <- k + 1L
                if dtype.Code = DTypeCode.Bool then Elem.normalizeBools a.Storage.Pointer (nativeint size)
            a

        /// <summary>A dense copy of an array.</summary>
        /// <param name="source">The array to copy.</param>
        /// <param name="order">Target layout; defaults to <see cref="P:YANum.Core.Config.DefaultOrder"/>.</param>
        /// <returns>A new array owning its data.</returns>
        static member copy (source: NDArray, ?order: Order) : NDArray =
            source.Copy(orderOf order)
