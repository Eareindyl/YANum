namespace YANum.Core

#nowarn "9"
#nowarn "51"

open System
open System.Numerics
open System.Runtime.CompilerServices
open System.Runtime.InteropServices
open System.Text
open System.Threading
open Microsoft.FSharp.NativeInterop

/// <summary>Strided element-level primitives shared by copying, comparison and ingest.</summary>
module internal Elem =

    /// Copies `size` elements of `itemSize` bytes between two strided layouts of the same shape.
    let copyStrided (srcBase: nativeint) (srcOffset: nativeint) (srcStrides: nativeint[])
                    (dstBase: nativeint) (dstOffset: nativeint) (dstStrides: nativeint[])
                    (shape: int[]) (size: nativeint) (itemSize: int) : unit =
        if size > 0n then
            let isz = nativeint itemSize
            let bytes = uint32 itemSize
            let rank = shape.Length
            if rank = 0 then
                Unsafe.CopyBlockUnaligned((dstBase + dstOffset * isz).ToPointer(),
                                          (srcBase + srcOffset * isz).ToPointer(), bytes)
            else
                let idx = Array.zeroCreate<int> rank
                let mutable so = srcOffset
                let mutable dof = dstOffset
                let mutable k = 0n
                while k < size do
                    Unsafe.CopyBlockUnaligned((dstBase + dof * isz).ToPointer(),
                                              (srcBase + so * isz).ToPointer(), bytes)
                    let mutable ax = rank - 1
                    let mutable carrying = true
                    while carrying && ax >= 0 do
                        idx.[ax] <- idx.[ax] + 1
                        so <- so + srcStrides.[ax]
                        dof <- dof + dstStrides.[ax]
                        if idx.[ax] < shape.[ax] then carrying <- false
                        else
                            let back = nativeint shape.[ax]
                            so <- so - srcStrides.[ax] * back
                            dof <- dof - dstStrides.[ax] * back
                            idx.[ax] <- 0
                            ax <- ax - 1
                    k <- k + 1n

    let inline private rd<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType>
        (addr: nativeint) : 'T = NativePtr.read (NativePtr.ofNativeInt<'T> addr)

    /// Value equality for one element, using the dtype's real comparison so that IEEE
    /// semantics survive: NaN is unequal to itself and -0.0 equals +0.0.
    let equalsAt (dtype: DType) (a: nativeint) (b: nativeint) : bool =
        match dtype.Code with
        | DTypeCode.Bool -> (rd<byte> a <> 0uy) = (rd<byte> b <> 0uy)
        | DTypeCode.Int8 -> rd<sbyte> a = rd<sbyte> b
        | DTypeCode.UInt8 -> rd<byte> a = rd<byte> b
        | DTypeCode.Int16 -> rd<int16> a = rd<int16> b
        | DTypeCode.UInt16 -> rd<uint16> a = rd<uint16> b
        | DTypeCode.Int32 -> rd<int32> a = rd<int32> b
        | DTypeCode.UInt32 -> rd<uint32> a = rd<uint32> b
        | DTypeCode.Int64 -> rd<int64> a = rd<int64> b
        | DTypeCode.UInt64 -> rd<uint64> a = rd<uint64> b
        | DTypeCode.Int128 -> rd<Int128> a = rd<Int128> b
        | DTypeCode.UInt128 -> rd<UInt128> a = rd<UInt128> b
        // Half and Complex are structs, so F#'s `=` routes through Equals, which follows
        // Double.Equals and reports NaN as equal to itself. Compare the components as
        // primitives instead, which the JIT lowers to ceq and which is what IEEE requires.
        | DTypeCode.Float16 -> float (rd<Half> a) = float (rd<Half> b)
        | DTypeCode.Float32 -> rd<float32> a = rd<float32> b
        | DTypeCode.Float64 -> rd<float> a = rd<float> b
        | DTypeCode.Decimal -> rd<decimal> a = rd<decimal> b
        | DTypeCode.Complex64 ->
            let ca = rd<Complex32> a
            let cb = rd<Complex32> b
            ca.Real = cb.Real && ca.Imaginary = cb.Imaginary
        | DTypeCode.Complex128 ->
            let ca = rd<Complex> a
            let cb = rd<Complex> b
            ca.Real = cb.Real && ca.Imaginary = cb.Imaginary
        | DTypeCode.Char -> rd<char> a = rd<char> b
        | DTypeCode.DateTime -> rd<DateTime> a = rd<DateTime> b
        | DTypeCode.DateTimeOffset -> rd<DateTimeOffset> a = rd<DateTimeOffset> b
        | DTypeCode.DateOnly -> rd<DateOnly> a = rd<DateOnly> b
        | DTypeCode.TimeOnly -> rd<TimeOnly> a = rd<TimeOnly> b
        | DTypeCode.TimeSpan -> rd<TimeSpan> a = rd<TimeSpan> b
        | DTypeCode.Utf16 | DTypeCode.Ascii ->
            let n = dtype.ItemSize
            let sa = ReadOnlySpan<byte>(NativePtr.toVoidPtr (NativePtr.ofNativeInt<byte> a), n)
            let sb = ReadOnlySpan<byte>(NativePtr.toVoidPtr (NativePtr.ofNativeInt<byte> b), n)
            sa.SequenceEqual sb
        | c -> Errors.typeError (sprintf "array_equal: dtype code %d has no equality kernel" (int c))

    /// Element-wise comparison of two strided layouts that already share dtype and shape.
    let equalStrided (dtype: DType)
                     (aBase: nativeint) (aOffset: nativeint) (aStrides: nativeint[])
                     (bBase: nativeint) (bOffset: nativeint) (bStrides: nativeint[])
                     (shape: int[]) (size: nativeint) : bool =
        if size = 0n then true
        else
            let isz = nativeint dtype.ItemSize
            let rank = shape.Length
            if rank = 0 then equalsAt dtype (aBase + aOffset * isz) (bBase + bOffset * isz)
            else
                let idx = Array.zeroCreate<int> rank
                let mutable ao = aOffset
                let mutable bo = bOffset
                let mutable k = 0n
                let mutable ok = true
                while ok && k < size do
                    ok <- equalsAt dtype (aBase + ao * isz) (bBase + bo * isz)
                    let mutable ax = rank - 1
                    let mutable carrying = true
                    while carrying && ax >= 0 do
                        idx.[ax] <- idx.[ax] + 1
                        ao <- ao + aStrides.[ax]
                        bo <- bo + bStrides.[ax]
                        if idx.[ax] < shape.[ax] then carrying <- false
                        else
                            let back = nativeint shape.[ax]
                            ao <- ao - aStrides.[ax] * back
                            bo <- bo - bStrides.[ax] * back
                            idx.[ax] <- 0
                            ax <- ax - 1
                    k <- k + 1n
                ok

    /// Forces every byte of a Bool buffer to 0 or 1.
    ///
    /// A `bool` byte of 0x02 compares as not-`true` under `=` while `if b` takes the true
    /// branch, and it is reachable through YANum's own MemoryMarshal reinterpretation. Every
    /// ingest path that reinterprets foreign bytes as Bool runs this.
    let normalizeBools (basePtr: nativeint) (count: nativeint) : unit =
        let mutable p = NativePtr.ofNativeInt<byte> basePtr
        let mutable i = 0n
        while i < count do
            if NativePtr.read p <> 0uy then NativePtr.write p 1uy
            p <- NativePtr.add p 1
            i <- i + 1n

/// <summary>
/// An n-dimensional, strided, dtype-tagged array.
/// </summary>
/// <remarks>
/// <para>
/// <b><c>=</c> on two arrays means <c>array_equal</c>, not NumPy's element-wise <c>==</c>.</b>
/// It returns a <c>bool</c>: same dtype, same shape, all elements equal. Element-wise equality
/// is the <c>.=</c> operator, which arrives in Phase 4 and returns a <c>Bool</c> array.
/// </para>
/// <para>
/// <b>Grammar rules for the operator surface</b> (Phases 3 and 4 introduce the operators; the
/// rules are stated here because they are properties of F#'s fixed precedence table):
/// mixed <c>&amp;&amp;&amp;</c>/<c>|||</c> chains require parentheses, because F# gives the
/// bitwise operators and the dot-comparisons one left-associative precedence level; and a
/// comparison whose operand is itself a comparison requires parentheses. Both are satisfied by
/// the habit NumPy already teaches — parenthesise every comparison.
/// </para>
/// <para>
/// All <c>Bool</c> arrays YANum produces contain only 0 or 1; every ingest path that
/// reinterprets foreign bytes normalises them.
/// </para>
/// </remarks>
// F# rejects [<CustomEquality>] on a class (FS0377); the override below is the custom
// equality, and there is deliberately no IComparable, so `<` does not resolve.
[<Sealed; AllowNullLiteral>]
[<StructuredFormatDisplay("{DisplayText}")>]
type NDArray =

    val internal Store : Storage
    val mutable internal Desc : ArrayDesc
    val internal BaseArray : NDArray
    val mutable internal DisposedFlag : int

    internal new (storage: Storage, desc: inref<ArrayDesc>, baseArray: NDArray) =
        { Store = storage; Desc = desc; BaseArray = baseArray; DisposedFlag = 0 }

    /// Builds an array, refreshing the cached flags and taking a storage reference for views.
    static member internal Build (storage: Storage, desc: byref<ArrayDesc>, baseArray: NDArray) : NDArray =
        Desc.refreshFlags &desc storage.Pointer storage.IsWritable (isNull baseArray)
        if not (isNull baseArray) then storage.AddRef()
        new NDArray(storage, &desc, baseArray)

    /// Builds a view of `src` from a descriptor derived from it.
    static member internal View (src: NDArray, desc: byref<ArrayDesc>) : NDArray =
        NDArray.Build(src.Store, &desc, src)

    // -----------------------------------------------------------------------------------
    //  Attributes
    // -----------------------------------------------------------------------------------

    /// <summary>The extents, materialised on demand.</summary>
    member this.Shape : int[] = Desc.shapeArray &this.Desc

    /// <summary>The number of axes, from 0 to 16.</summary>
    member this.Rank : int = this.Desc.Rank

    /// <summary>The total number of elements.</summary>
    member this.Size : int64 = int64 this.Desc.Size

    /// <summary>The element dtype.</summary>
    member this.DType : DType = this.Desc.DType

    /// <summary>The size of one element in bytes.</summary>
    member this.ItemSize : int = this.Desc.DType.ItemSize

    /// <summary>The total number of bytes the elements occupy, ignoring gaps.</summary>
    member this.NBytes : int64 = int64 this.Desc.Size * int64 this.Desc.DType.ItemSize

    /// <summary>The strides, <b>in elements</b>, materialised on demand.</summary>
    member this.Strides : nativeint[] = Desc.stridesArray &this.Desc

    /// <summary>The strides in bytes: the element strides times the item size.</summary>
    member this.StridesBytes : nativeint[] =
        let s = Desc.stridesArray &this.Desc
        let isz = nativeint this.Desc.DType.ItemSize
        for i in 0 .. s.Length - 1 do s.[i] <- s.[i] * isz
        s

    /// <summary>The cached layout facts, computed at construction.</summary>
    member this.Flags : ArrayFlags = this.Desc.Flags

    // Every flag test below is a bitwise mask, not Enum HasFlag: the latter boxes both the
    // receiver and the argument from F#, measured at 48 bytes a call.
    /// <summary>Whether the elements are laid out row-major with no gaps.</summary>
    member this.IsCContiguous : bool = this.Desc.Flags &&& ArrayFlags.CContiguous <> ArrayFlags.None

    /// <summary>Whether the elements are laid out column-major with no gaps.</summary>
    member this.IsFContiguous : bool = this.Desc.Flags &&& ArrayFlags.FContiguous <> ArrayFlags.None

    /// <summary>Whether elements may be written.</summary>
    member this.IsWritable : bool = this.Desc.Flags &&& ArrayFlags.Writable <> ArrayFlags.None

    /// <summary>Whether the first element's address is a multiple of the item size.</summary>
    member this.IsAligned : bool = this.Desc.Flags &&& ArrayFlags.Aligned <> ArrayFlags.None

    /// <summary>The array this one is a view of, or <c>null</c> when it owns its data.</summary>
    member this.Base : NDArray = this.BaseArray

    /// <summary>The array at the end of the view chain: the one that actually owns the storage.</summary>
    member this.RootBase : NDArray =
        let mutable a = this
        while not (isNull a.BaseArray) do a <- a.BaseArray
        a

    /// <summary>Whether this array owns its storage rather than viewing someone else's.</summary>
    member this.OwnsData : bool = isNull this.BaseArray

    /// <summary>Which allocator backs the elements.</summary>
    member this.Backend : StorageBackend = this.Store.Backend

    /// <summary>The underlying storage block. Internal detail exposed for interop and tests.</summary>
    member this.Storage : Storage = this.Store

    /// <summary>Distance from the start of the storage block to element zero, in elements.</summary>
    member this.Offset : nativeint = this.Desc.Offset

    // -----------------------------------------------------------------------------------
    //  Views
    // -----------------------------------------------------------------------------------

    /// <summary>A view with the axis order reversed. O(1); shares storage.</summary>
    /// <returns>The transposed view.</returns>
    member this.Transpose () : NDArray =
        let mutable d = Unchecked.defaultof<ArrayDesc>
        Desc.transposeInto &this.Desc &d
        NDArray.View(this, &d)

    /// <summary>A view with the axis order reversed. O(1); shares storage.</summary>
    member this.T : NDArray = this.Transpose()

    /// <summary>A view with the axes reordered by an explicit permutation. O(1); shares storage.</summary>
    /// <param name="axes">A permutation of <c>0 .. rank-1</c>; negative entries count from the end.</param>
    /// <returns>The permuted view.</returns>
    /// <exception cref="T:YANum.Core.YANumShapeError">The argument is not a permutation of the axes.</exception>
    member this.Permute (axes: int[]) : NDArray =
        let n = this.Desc.Rank
        if isNull (box axes) || axes.Length <> n then
            Errors.shapeError
                (sprintf "permute: expected %d axes for an array of rank %d, got %d"
                    n n (if isNull (box axes) then 0 else axes.Length))
        let perm = Array.zeroCreate<int> n
        let seen = Array.zeroCreate<bool> n
        for i in 0 .. n - 1 do
            let a = Shape.normalizeAxis n axes.[i]
            if seen.[a] then
                Errors.shapeError (sprintf "permute: axis %d appears more than once in the permutation" a)
            seen.[a] <- true
            perm.[i] <- a
        let mutable d = Unchecked.defaultof<ArrayDesc>
        Desc.permuteInto &this.Desc perm &d
        NDArray.View(this, &d)

    /// <summary>A view with two axes exchanged. O(1); shares storage.</summary>
    /// <param name="axis1">First axis; negative values count from the end.</param>
    /// <param name="axis2">Second axis; negative values count from the end.</param>
    /// <returns>The swapped view.</returns>
    member this.SwapAxes (axis1: int, axis2: int) : NDArray =
        let n = this.Desc.Rank
        let a = Shape.normalizeAxis n axis1
        let b = Shape.normalizeAxis n axis2
        let perm = Array.init n id
        perm.[a] <- b
        perm.[b] <- a
        let mutable d = Unchecked.defaultof<ArrayDesc>
        Desc.permuteInto &this.Desc perm &d
        NDArray.View(this, &d)

    /// <summary>A view with one axis moved to a new position. O(1); shares storage.</summary>
    /// <param name="source">Axis to move; negative values count from the end.</param>
    /// <param name="destination">Position it should end up at; negative values count from the end.</param>
    /// <returns>The reordered view.</returns>
    member this.MoveAxis (source: int, destination: int) : NDArray =
        let n = this.Desc.Rank
        let s = Shape.normalizeAxis n source
        let d0 = Shape.normalizeAxis n destination
        let rest = ResizeArray<int>(n)
        for i in 0 .. n - 1 do
            if i <> s then rest.Add i
        rest.Insert(d0, s)
        this.Permute(rest.ToArray())

    /// <summary>A view with a length-1 axis inserted. O(1); shares storage.</summary>
    /// <param name="axis">Insertion position in <c>-rank-1 .. rank</c>.</param>
    /// <returns>The expanded view.</returns>
    /// <exception cref="T:YANum.Core.YANumShapeError">The result would exceed rank 16, or the axis is out of range.</exception>
    member this.ExpandDims (axis: int) : NDArray =
        let n = this.Desc.Rank
        if n + 1 > Shape.MaxRank then
            Errors.shapeError
                (sprintf "expandDims: the result would have rank %d, above YANum's maximum of %d" (n + 1) Shape.MaxRank)
        let a = if axis < 0 then axis + n + 1 else axis
        if a < 0 || a > n then
            Errors.shapeError
                (sprintf "expandDims: axis %d is out of range for an array of rank %d (valid axes are %d..%d)"
                    axis n (-n - 1) n)
        let mutable d = Unchecked.defaultof<ArrayDesc>
        Desc.expandDimsInto &this.Desc a &d
        NDArray.View(this, &d)

    /// <summary>A view with every length-1 axis removed. O(1); shares storage.</summary>
    /// <returns>The squeezed view.</returns>
    member this.Squeeze () : NDArray =
        let mutable d = Unchecked.defaultof<ArrayDesc>
        Desc.squeezeInto &this.Desc null &d
        NDArray.View(this, &d)

    /// <summary>A view with the named length-1 axes removed. O(1); shares storage.</summary>
    /// <param name="axes">Axes to drop; each must have extent 1. Negative values count from the end.</param>
    /// <returns>The squeezed view.</returns>
    /// <exception cref="T:YANum.Core.YANumShapeError">An axis does not have extent 1.</exception>
    member this.Squeeze (axes: int[]) : NDArray =
        let n = this.Desc.Rank
        let norm = Array.map (Shape.normalizeAxis n) axes
        let sh = Desc.shapeSpan &this.Desc
        for a in norm do
            if sh.[a] <> 1 then
                Errors.shapeError
                    (sprintf "squeeze: axis %d has extent %d; only axes of extent 1 can be squeezed" a sh.[a])
        let mutable d = Unchecked.defaultof<ArrayDesc>
        Desc.squeezeInto &this.Desc norm &d
        NDArray.View(this, &d)

    /// <summary>
    /// A reshaped array, as a view when the layout allows and a fresh copy otherwise.
    /// </summary>
    /// <param name="newShape">The requested extents. Exactly one entry may be <c>-1</c>, which is inferred.</param>
    /// <param name="order">Reading order for the reshape; defaults to <see cref="P:YANum.Core.Config.DefaultOrder"/>.</param>
    /// <returns>The reshaped array.</returns>
    /// <remarks>
    /// <b>Whether you get a view or a copy is observable</b>: check <c>OwnsData</c>, or
    /// <c>Base</c>, on the result. A view is returned exactly when
    /// <see cref="M:YANum.Core.Shape.canReshape"/> allows one — that is, when the source is
    /// contiguous in the requested order. Everything else is served by a contiguous copy.
    /// </remarks>
    /// <exception cref="T:YANum.Core.YANumShapeError">The requested shape has a different total size, or more than one <c>-1</c>.</exception>
    member private this.ReshapeCore (newShape: int[], order: Order) : NDArray =
        let size = this.Desc.Size
        // Deliberately loop rather than use tryFindIndex/filter: those allocate a closure and
        // an intermediate array on a path that otherwise allocates only the result object.
        let mutable inferAt = -1
        let mutable inferCount = 0
        for i in 0 .. newShape.Length - 1 do
            if newShape.[i] = -1 then
                inferAt <- i
                inferCount <- inferCount + 1
        let resolved =
            if inferCount = 0 then
                Shape.validate newShape
                newShape
            else
                if inferCount > 1 then
                    Errors.shapeError "reshape: at most one extent may be -1"
                let r = Array.copy newShape
                r.[inferAt] <- 1
                Shape.validate r
                let known = Shape.size r
                if known = 0n || size % known <> 0n then
                    Errors.shapeError
                        (sprintf "reshape: cannot infer the -1 extent for shape %s from an array of %d elements"
                            (Errors.formatShape newShape) (int64 size))
                r.[inferAt] <- int (size / known)
                r
        if Shape.size resolved <> size then
            Errors.shapeError
                (sprintf "reshape: cannot reshape an array of %d elements (shape %s) into shape %s"
                    (int64 size) (Errors.formatShape (Desc.shapeArray &this.Desc)) (Errors.formatShape resolved))
        // The cached flags already answer Shape.canReshape without materialising the shape
        // and stride arrays, and an array of size <= 1 carries both contiguity flags.
        let viewPossible =
            match order with
            | C -> this.IsCContiguous
            | F -> this.IsFContiguous
            | K -> this.IsCContiguous || this.IsFContiguous
        if viewPossible then
            let mutable d = Unchecked.defaultof<ArrayDesc>
            Desc.reshapeInto &this.Desc resolved order &d
            NDArray.View(this, &d)
        else
            // No stride-only view exists, so the reshape is served by a contiguous copy. The
            // result owns that copy outright rather than viewing a temporary.
            let copied = this.Copy order
            let mutable d = Unchecked.defaultof<ArrayDesc>
            Desc.reshapeInto &copied.Desc resolved order &d
            NDArray.Build(copied.Store, &d, null)

    /// <summary>A reshaped array using <see cref="P:YANum.Core.Config.DefaultOrder"/>.</summary>
    /// <param name="newShape">The requested extents. Exactly one entry may be <c>-1</c>.</param>
    /// <returns>The reshaped array — a view when possible, a copy otherwise.</returns>
    member this.Reshape (newShape: int[]) : NDArray =
        this.ReshapeCore(newShape, Config.DefaultOrder)

    /// <summary>A reshaped array using an explicit order.</summary>
    /// <param name="newShape">The requested extents. Exactly one entry may be <c>-1</c>.</param>
    /// <param name="order">Reading order for the reshape.</param>
    /// <returns>The reshaped array — a view when possible, a copy otherwise.</returns>
    member this.Reshape (newShape: int[], order: Order) : NDArray =
        this.ReshapeCore(newShape, order)

    /// <summary>A rank-1 array of every element, as a view when possible.</summary>
    /// <param name="order">Reading order; defaults to <see cref="P:YANum.Core.Config.DefaultOrder"/>.</param>
    /// <returns>The flattened array.</returns>
    member this.Ravel (order: Order) : NDArray = this.ReshapeCore([| int this.Desc.Size |], order)

    /// <summary>A rank-1 array of every element, as a view when possible.</summary>
    /// <returns>The flattened array.</returns>
    member this.Ravel () : NDArray = this.Ravel Config.DefaultOrder

    /// <summary>A dense copy of the elements in the requested order.</summary>
    /// <param name="order">Target layout; <c>K</c> keeps an F-contiguous source in F order and everything else in C order.</param>
    /// <returns>A new array owning its data.</returns>
    member this.Copy (order: Order) : NDArray =
        let shape = Desc.shapeArray &this.Desc
        let srcStrides = Desc.stridesArray &this.Desc
        let dtype = this.Desc.DType
        let effective =
            match order with
            | K -> if this.IsFContiguous && not this.IsCContiguous then F else C
            | o -> o
        let size = this.Desc.Size
        let bytes = unativeint size * unativeint dtype.ItemSize
        let storage = Storage.AllocateUninitialized(bytes, this.Store.Backend)
        let mutable d = Unchecked.defaultof<ArrayDesc>
        Desc.initDense &d shape effective dtype
        let dstStrides = Desc.stridesArray &d
        Elem.copyStrided this.Store.Pointer this.Desc.Offset srcStrides
                         storage.Pointer 0n dstStrides shape size dtype.ItemSize
        NDArray.Build(storage, &d, null)

    /// <summary>A dense copy in <see cref="P:YANum.Core.Config.DefaultOrder"/>.</summary>
    /// <returns>A new array owning its data.</returns>
    member this.Copy () : NDArray = this.Copy Config.DefaultOrder

    /// <summary>This array if it is already C-contiguous, otherwise a C-contiguous copy.</summary>
    /// <returns>A C-contiguous array with the same shape, dtype and values.</returns>
    member this.AsContiguous () : NDArray = if this.IsCContiguous then this else this.Copy C

    /// <summary>This array if it is already F-contiguous, otherwise an F-contiguous copy.</summary>
    /// <returns>An F-contiguous array with the same shape, dtype and values.</returns>
    member this.AsFortran () : NDArray = if this.IsFContiguous then this else this.Copy F

    // -----------------------------------------------------------------------------------
    //  Element access
    // -----------------------------------------------------------------------------------

    member private this.CheckElementType<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType>
        (op: string) : unit =
        let want = DTypeOf<'T>.Value
        if want.Code <> this.Desc.DType.Code then
            Errors.typeError
                (sprintf "%s: this array has dtype %s, but the element type '%s' maps to dtype %s"
                    op this.Desc.DType.Name typeof<'T>.Name want.Name)

    /// <summary>Reads one element.</summary>
    /// <param name="idx">One index per axis; negative values count from the end.</param>
    /// <returns>The element.</returns>
    /// <exception cref="T:YANum.Core.YANumTypeError">The element type does not match the array's dtype.</exception>
    /// <exception cref="T:YANum.Core.YANumShapeError">The wrong number of indices was supplied.</exception>
    member this.GetItem<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType>
        (idx: int[]) : 'T =
        this.CheckElementType<'T> "getItem"
        let off = Desc.offsetOf &this.Desc idx Config.BoundsCheckActive "getItem"
        NativePtr.read (this.Store.PointerAs<'T> off)

    /// <summary>Writes one element.</summary>
    /// <param name="idx">One index per axis; negative values count from the end.</param>
    /// <param name="value">The value to store.</param>
    /// <exception cref="T:YANum.Core.YANumTypeError">The element type does not match the array's dtype.</exception>
    /// <exception cref="T:YANum.Core.YANumValueError">The array is read-only.</exception>
    member this.SetItem<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType>
        (idx: int[], value: 'T) : unit =
        this.CheckElementType<'T> "setItem"
        if not this.IsWritable then Errors.valueError "setItem: this array is read-only"
        let off = Desc.offsetOf &this.Desc idx Config.BoundsCheckActive "setItem"
        NativePtr.write (this.Store.PointerAs<'T> off) value

    /// <summary>The byte address of one element. Used by interop and by the string accessors.</summary>
    /// <param name="idx">One index per axis; negative values count from the end.</param>
    /// <param name="op">Operation name for error messages.</param>
    /// <returns>The absolute address of the element.</returns>
    member internal this.AddressOf (idx: int[], op: string) : nativeint =
        let off = Desc.offsetOf &this.Desc idx Config.BoundsCheckActive op
        this.Store.Pointer + off * nativeint this.Desc.DType.ItemSize

    // -----------------------------------------------------------------------------------
    //  String access
    // -----------------------------------------------------------------------------------

    member private this.RequireString (op: string) : unit =
        if not this.Desc.DType.IsString then
            Errors.typeError
                (sprintf "%s: this array has dtype %s; string access needs utf16<n> or ascii<n>"
                    op this.Desc.DType.Name)

    /// <summary>Reads one fixed-width string element, without its NUL padding.</summary>
    /// <param name="idx">One index per axis; negative values count from the end.</param>
    /// <returns>The stored value.</returns>
    /// <exception cref="T:YANum.Core.YANumTypeError">The array does not have a string dtype.</exception>
    member this.GetString (idx: int[]) : string =
        this.RequireString "getString"
        Printing.readStringElement this.Desc.DType (this.AddressOf(idx, "getString"))

    /// <summary>
    /// Writes one fixed-width string element, NUL-padding the remainder of the slot.
    /// </summary>
    /// <param name="idx">One index per axis; negative values count from the end.</param>
    /// <param name="value">The value to store; <c>null</c> stores the empty string.</param>
    /// <exception cref="T:YANum.Core.YANumOverflowError">
    /// The value is longer than the dtype width and
    /// <see cref="P:YANum.Core.Config.StringOverflow"/> is <c>Throw</c>.
    /// </exception>
    /// <exception cref="T:YANum.Core.YANumTypeError">The array does not have a string dtype.</exception>
    member this.SetString (idx: int[], value: string) : unit =
        this.RequireString "setString"
        if not this.IsWritable then Errors.valueError "setString: this array is read-only"
        let dtype = this.Desc.DType
        let w = dtype.Width
        let s = if isNull value then "" else value
        let s =
            if s.Length <= w then s
            else
                match Config.StringOverflow with
                | Truncate -> s.Substring(0, w)
                | Throw ->
                    Errors.overflowError
                        (sprintf "setString: the value is %d characters but the dtype is %s, which holds %d"
                            s.Length dtype.Name w)
        let addr = this.AddressOf(idx, "setString")
        match dtype.Code with
        | DTypeCode.Utf16 ->
            let dst = Span<char>(NativePtr.toVoidPtr (NativePtr.ofNativeInt<char> addr), w)
            dst.Clear()
            s.AsSpan().CopyTo(dst)
        | _ ->
            let dst = Span<byte>(NativePtr.toVoidPtr (NativePtr.ofNativeInt<byte> addr), w)
            dst.Clear()
            Encoding.ASCII.GetBytes(s.AsSpan(), dst) |> ignore

    /// <summary>A read-only span over one <c>utf16&lt;n&gt;</c> element, padding included.</summary>
    /// <param name="idx">One index per axis; negative values count from the end.</param>
    /// <returns>A span of exactly <c>Width</c> UTF-16 code units.</returns>
    /// <exception cref="T:YANum.Core.YANumTypeError">The array's dtype is not <c>utf16&lt;n&gt;</c>.</exception>
    member this.SpanOf (idx: int[]) : ReadOnlySpan<char> =
        if this.Desc.DType.Code <> DTypeCode.Utf16 then
            (Errors.typeError
                (sprintf "spanOf: a ReadOnlySpan<char> needs dtype utf16<n>, but this array is %s"
                    this.Desc.DType.Name) : unit)
        let addr = this.AddressOf(idx, "spanOf")
        ReadOnlySpan<char>(NativePtr.toVoidPtr (NativePtr.ofNativeInt<char> addr), this.Desc.DType.Width)

    // -----------------------------------------------------------------------------------
    //  Interop
    // -----------------------------------------------------------------------------------

    /// <summary>Copies every element into a fresh one-dimensional array, in C order.</summary>
    /// <returns>A new array of <c>Size</c> elements.</returns>
    /// <exception cref="T:YANum.Core.YANumTypeError">The element type does not match the array's dtype.</exception>
    /// <exception cref="T:YANum.Core.YANumValueError">The array has more than <c>Int32.MaxValue</c> elements.</exception>
    member this.ToArray<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType> () : 'T[] =
        this.CheckElementType<'T> "toArray"
        let size = this.Desc.Size
        if size > nativeint Int32.MaxValue then
            Errors.valueError (sprintf "toArray: %d elements do not fit a .NET array" (int64 size))
        let result = Array.zeroCreate<'T> (int size)
        if size > 0n then
            let shape = Desc.shapeArray &this.Desc
            let srcStrides = Desc.stridesArray &this.Desc
            let dstStrides = Shape.cStrides shape
            use p = fixed result
            Elem.copyStrided this.Store.Pointer this.Desc.Offset srcStrides
                             (NativePtr.toNativeInt p) 0n dstStrides shape size this.Desc.DType.ItemSize
        result

    /// <summary>Copies a rank-2 array into a fresh two-dimensional .NET array.</summary>
    /// <returns>A new <c>'T[,]</c> with the same extents.</returns>
    /// <exception cref="T:YANum.Core.YANumShapeError">The array is not rank 2.</exception>
    member this.ToArray2D<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType> () : 'T[,] =
        this.CheckElementType<'T> "toArray2D"
        if this.Desc.Rank <> 2 then
            Errors.shapeError (sprintf "toArray2D: this array has rank %d, not 2" this.Desc.Rank)
        let sh = Desc.shapeSpan &this.Desc
        let rows = sh.[0]
        let cols = sh.[1]
        let result = Array2D.zeroCreate<'T> rows cols
        if this.Desc.Size > 0n then
            let shape = [| rows; cols |]
            let srcStrides = Desc.stridesArray &this.Desc
            let dstStrides = Shape.cStrides shape
            let handle = GCHandle.Alloc(result, GCHandleType.Pinned)
            try
                Elem.copyStrided this.Store.Pointer this.Desc.Offset srcStrides
                                 (handle.AddrOfPinnedObject()) 0n dstStrides shape this.Desc.Size
                                 this.Desc.DType.ItemSize
            finally
                handle.Free()
        result

    /// <summary>A span aliasing the elements. Contiguous arrays only.</summary>
    /// <returns>A span over the elements in memory order.</returns>
    /// <exception cref="T:YANum.Core.YANumShapeError">The array is not contiguous.</exception>
    /// <exception cref="T:YANum.Core.YANumTypeError">The element type does not match the array's dtype.</exception>
    member this.AsSpan<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType> () : Span<'T> =
        this.CheckElementType<'T> "asSpan"
        if not (this.IsCContiguous || this.IsFContiguous) then
            (Errors.shapeError
                "asSpan: this array is not contiguous; call AsContiguous() first, which copies" : unit)
        this.Store.AsSpan<'T>(this.Desc.Offset, this.Desc.Size)

    /// <summary>Copies every string element into a fresh array, in C order.</summary>
    /// <returns>A new array of <c>Size</c> strings, padding removed.</returns>
    /// <exception cref="T:YANum.Core.YANumTypeError">The array does not have a string dtype.</exception>
    member this.ToStringArray () : string[] =
        this.RequireString "toStringArray"
        let size = this.Desc.Size
        if size > nativeint Int32.MaxValue then
            Errors.valueError (sprintf "toStringArray: %d elements do not fit a .NET array" (int64 size))
        let result = Array.zeroCreate<string> (int size)
        if size > 0n then
            let dtype = this.Desc.DType
            let isz = nativeint dtype.ItemSize
            let shape = Desc.shapeArray &this.Desc
            let strides = Desc.stridesArray &this.Desc
            let rank = shape.Length
            let basePtr = this.Store.Pointer
            if rank = 0 then
                result.[0] <- Printing.readStringElement dtype (basePtr + this.Desc.Offset * isz)
            else
                let idx = Array.zeroCreate<int> rank
                let mutable off = this.Desc.Offset
                let mutable k = 0
                while k < int size do
                    result.[k] <- Printing.readStringElement dtype (basePtr + off * isz)
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
                    k <- k + 1
        result

    // -----------------------------------------------------------------------------------
    //  Equality, printing, lifetime
    // -----------------------------------------------------------------------------------

    /// <summary>
    /// <c>array_equal</c> semantics: same dtype, same shape, every element equal.
    /// </summary>
    /// <param name="other">The array to compare against.</param>
    /// <returns><c>true</c> when the two arrays are element-wise identical.</returns>
    /// <remarks>
    /// Comparison uses the dtype's real element equality, not <c>memcmp</c>, except where
    /// <see cref="P:YANum.Core.DType.MemcmpLegal"/> is set and both layouts are densely packed.
    /// So two <c>NaN</c> arrays are <b>not</b> equal, while <c>-0.0</c> and <c>+0.0</c> arrays
    /// <b>are</b>, and <c>1.0m</c> equals <c>1.00m</c>.
    /// </remarks>
    member this.ArrayEqual (other: NDArray) : bool =
        if isNull (box other) then false
        elif Object.ReferenceEquals(this, other) then
            // An array is equal to itself only if it holds no NaN; fall through to the real
            // comparison rather than short-circuiting, so IEEE semantics survive.
            this.ElementsEqual other
        else
            // Field-by-field, never `let a = this.Desc`: that would copy 224 bytes.
            if this.Desc.DType <> other.Desc.DType then false
            elif this.Desc.Rank <> other.Desc.Rank then false
            elif this.Desc.Size <> other.Desc.Size then false
            else
                let sa = Desc.shapeSpan &this.Desc
                let sb = Desc.shapeSpan &other.Desc
                let mutable same = true
                for i in 0 .. this.Desc.Rank - 1 do
                    if sa.[i] <> sb.[i] then same <- false
                if not same then false else this.ElementsEqual other

    member private this.ElementsEqual (other: NDArray) : bool =
        let dtype = this.Desc.DType
        let size = this.Desc.Size
        if size = 0n then true
        elif dtype.MemcmpLegal
             && ((this.IsCContiguous && other.IsCContiguous) || (this.IsFContiguous && other.IsFContiguous))
             && size <= nativeint Int32.MaxValue then
            let n = int size * dtype.ItemSize
            let sa = ReadOnlySpan<byte>(NativePtr.toVoidPtr (NativePtr.ofNativeInt<byte>
                        (this.Store.Pointer + this.Desc.Offset * nativeint dtype.ItemSize)), n)
            let sb = ReadOnlySpan<byte>(NativePtr.toVoidPtr (NativePtr.ofNativeInt<byte>
                        (other.Store.Pointer + other.Desc.Offset * nativeint dtype.ItemSize)), n)
            sa.SequenceEqual sb
        else
            Elem.equalStrided dtype
                this.Store.Pointer this.Desc.Offset (Desc.stridesArray &this.Desc)
                other.Store.Pointer other.Desc.Offset (Desc.stridesArray &other.Desc)
                (Desc.shapeArray &this.Desc) size

    /// <summary><c>array_equal</c> semantics against a boxed value.</summary>
    /// <param name="o">The value to compare against.</param>
    /// <returns><c>true</c> when <paramref name="o"/> is an <see cref="T:YANum.Core.NDArray"/> that compares equal.</returns>
    override this.Equals (o: obj) : bool =
        match o with
        | :? NDArray as a -> this.ArrayEqual a
        | _ -> false

    /// <summary>Hashes the shape and dtype only.</summary>
    /// <returns>The hash code.</returns>
    /// <remarks>
    /// <b>Collisions are expected and frequent</b>: every <c>float64</c> array of shape
    /// <c>[1000]</c> hashes identically. Hashing the contents would be O(n) and is not a
    /// sensible default for a numerical array. Do not use arrays as dictionary keys on a hot
    /// path.
    /// </remarks>
    override this.GetHashCode () : int =
        let mutable h = HashCode()
        h.Add(int this.Desc.DType.Code)
        h.Add(this.Desc.DType.Width)
        h.Add(this.Desc.Rank)
        let sh = Desc.shapeSpan &this.Desc
        for i in 0 .. this.Desc.Rank - 1 do h.Add(sh.[i])
        h.ToHashCode()

    interface IEquatable<NDArray> with
        member this.Equals (other: NDArray) = this.ArrayEqual other

    /// <summary>A repr header naming shape and dtype, followed by the values.</summary>
    /// <returns>The rendered array.</returns>
    override this.ToString () : string = Printing.render this.Store &this.Desc

    /// <summary>Backs <c>%A</c> formatting through <c>StructuredFormatDisplay</c>.</summary>
    member this.DisplayText : string = this.ToString()

    /// <summary>
    /// Drops this array's reference to its storage.
    /// </summary>
    /// <remarks>
    /// A no-op for the managed backend, where the GC owns the lifetime. For the native backend
    /// this drops one reference count; the block is freed when the last one goes.
    /// <b>Disposing the owner of a native-backed array while views are alive is a
    /// use-after-free</b> once the count reaches zero. Set
    /// <see cref="P:YANum.Core.Config.CheckAliasing"/> to have that asserted.
    /// </remarks>
    member this.Dispose () : unit =
        if Interlocked.Exchange(&this.DisposedFlag, 1) = 0 then
            if this.OwnsData then this.Store.Dispose() else this.Store.Release()

    interface IDisposable with
        member this.Dispose () = this.Dispose()
