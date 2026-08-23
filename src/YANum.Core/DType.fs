namespace YANum.Core

// ---------------------------------------------------------------------------------------
//  THE CANONICAL GENERIC CONSTRAINT SET  (Phase 0, spikes 0.9 / 0.10)
//
//  `'T : unmanaged` alone does NOT compile against MemoryMarshal.Cast / MemoryMarshal.AsBytes:
//  the BCL signatures demand `struct` plus a default constructor, and F# does not infer those
//  from `unmanaged`. Every generic signature in YANum that reinterprets element bytes uses,
//  VERBATIM:
//
//      when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType
//
//  F# has no constraint abbreviation. Consistency is by convention and review. Deviating
//  produces error FS0340 at the signature boundary, which is at least loud.
// ---------------------------------------------------------------------------------------

open System
open System.Numerics
open System.Runtime.CompilerServices

/// <summary>Coarse classification of a dtype, used to drive promotion and kernel selection.</summary>
type DTypeKind =
    /// <summary>The single-byte boolean dtype.</summary>
    | Boolean = 0
    /// <summary>Two's-complement signed integers.</summary>
    | SignedInt = 1
    /// <summary>Unsigned integers.</summary>
    | UnsignedInt = 2
    /// <summary>IEEE-754 binary floating point.</summary>
    | Float = 3
    /// <summary>The .NET base-10 <see cref="T:System.Decimal"/>.</summary>
    | Decimal = 4
    /// <summary>Complex numbers with floating-point components.</summary>
    | Complex = 5
    /// <summary>UTF-16 code units treated as a scalar dtype.</summary>
    | Char = 6
    /// <summary>A point on a timeline (date and/or time of day).</summary>
    | Instant = 7
    /// <summary>An elapsed amount of time.</summary>
    | Duration = 8
    /// <summary>Fixed-width string dtypes.</summary>
    | Str = 9

/// <summary>
/// The dense, byte-backed dtype discriminator. All kernel-table indexing uses this value
/// alone; the fixed-width string dtypes carry their width beside it in
/// <see cref="T:YANum.Core.DType"/>.
/// </summary>
type DTypeCode =
    /// <summary>1-byte boolean, normalised to 0 or 1.</summary>
    | Bool = 0uy
    /// <summary>Signed 8-bit integer.</summary>
    | Int8 = 1uy
    /// <summary>Unsigned 8-bit integer. Also spelled <c>Byte</c>.</summary>
    | UInt8 = 2uy
    /// <summary>Signed 16-bit integer.</summary>
    | Int16 = 3uy
    /// <summary>Unsigned 16-bit integer.</summary>
    | UInt16 = 4uy
    /// <summary>Signed 32-bit integer.</summary>
    | Int32 = 5uy
    /// <summary>Unsigned 32-bit integer.</summary>
    | UInt32 = 6uy
    /// <summary>Signed 64-bit integer.</summary>
    | Int64 = 7uy
    /// <summary>Unsigned 64-bit integer.</summary>
    | UInt64 = 8uy
    /// <summary>Signed 128-bit integer.</summary>
    | Int128 = 9uy
    /// <summary>Unsigned 128-bit integer.</summary>
    | UInt128 = 10uy
    /// <summary>IEEE-754 binary16.</summary>
    | Float16 = 11uy
    /// <summary>IEEE-754 binary32.</summary>
    | Float32 = 12uy
    /// <summary>IEEE-754 binary64.</summary>
    | Float64 = 13uy
    /// <summary>The .NET 128-bit base-10 decimal.</summary>
    | Decimal = 14uy
    /// <summary>Complex with two binary32 components (NumPy naming: total bits).</summary>
    | Complex64 = 15uy
    /// <summary>Complex with two binary64 components (NumPy naming: total bits).</summary>
    | Complex128 = 16uy
    /// <summary>A single UTF-16 code unit.</summary>
    | Char = 17uy
    /// <summary>A <see cref="T:System.DateTime"/>, kind preserved.</summary>
    | DateTime = 18uy
    /// <summary>A <see cref="T:System.DateTimeOffset"/>.</summary>
    | DateTimeOffset = 19uy
    /// <summary>A <see cref="T:System.DateOnly"/>.</summary>
    | DateOnly = 20uy
    /// <summary>A <see cref="T:System.TimeOnly"/>.</summary>
    | TimeOnly = 21uy
    /// <summary>A <see cref="T:System.TimeSpan"/>.</summary>
    | TimeSpan = 22uy
    /// <summary>Fixed-width UTF-16 string of <c>Width</c> code units.</summary>
    | Utf16 = 23uy
    /// <summary>Fixed-width ASCII string of <c>Width</c> bytes.</summary>
    | Ascii = 24uy

/// <summary>Flat metadata tables indexed by <see cref="T:YANum.Core.DTypeCode"/>.</summary>
module internal DTypeTables =

    [<Literal>]
    let Count = 25

    /// Byte size of one element; 0 for the two width-parameterised string dtypes.
    let itemSize : int[] =
        [| 1; 1; 1; 2; 2; 4; 4; 8; 8; 16; 16; 2; 4; 8; 16; 8; 16; 2; 8; 16; 4; 8; 8; 0; 0 |]

    /// Preferred alignment of one element, in bytes.
    let alignment : int[] =
        [| 1; 1; 1; 2; 2; 4; 4; 8; 8; 16; 16; 2; 4; 8; 8; 4; 8; 2; 8; 8; 4; 8; 8; 2; 1 |]

    let kind : DTypeKind[] =
        [| DTypeKind.Boolean; DTypeKind.SignedInt; DTypeKind.UnsignedInt
           DTypeKind.SignedInt; DTypeKind.UnsignedInt
           DTypeKind.SignedInt; DTypeKind.UnsignedInt
           DTypeKind.SignedInt; DTypeKind.UnsignedInt
           DTypeKind.SignedInt; DTypeKind.UnsignedInt
           DTypeKind.Float; DTypeKind.Float; DTypeKind.Float
           DTypeKind.Decimal
           DTypeKind.Complex; DTypeKind.Complex
           DTypeKind.Char
           DTypeKind.Instant; DTypeKind.Instant; DTypeKind.Instant; DTypeKind.Instant
           DTypeKind.Duration
           DTypeKind.Str; DTypeKind.Str |]

    let simd : bool[] =
        [| true; true; true; true; true; true; true; true; true
           false; false
           false; true; true
           false
           false; false
           true
           false; false; false; false; false
           false; false |]

    /// Phase 0 spike 0.9: byte comparison agrees with value equality only for these.
    let memcmpLegal : bool[] =
        [| true; true; true; true; true; true; true; true; true; true; true
           false; false; false
           false
           false; false
           true
           false; false
           true; true; true
           true; true |]

    let mantissaBits : int[] =
        [| 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 11; 24; 53; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0; 0 |]

    let names : string[] =
        [| "bool"; "int8"; "uint8"; "int16"; "uint16"; "int32"; "uint32"; "int64"; "uint64"
           "int128"; "uint128"; "float16"; "float32"; "float64"; "decimal"
           "complex64"; "complex128"; "char"
           "datetime"; "datetimeoffset"; "dateonly"; "timeonly"; "timespan"
           "utf16"; "ascii" |]

    let clrTypes : Type[] =
        [| typeof<bool>; typeof<sbyte>; typeof<byte>; typeof<int16>; typeof<uint16>
           typeof<int32>; typeof<uint32>; typeof<int64>; typeof<uint64>
           typeof<Int128>; typeof<UInt128>
           typeof<Half>; typeof<float32>; typeof<float>
           typeof<decimal>
           typeof<Complex32>; typeof<Complex>
           typeof<char>
           typeof<DateTime>; typeof<DateTimeOffset>; typeof<DateOnly>; typeof<TimeOnly>
           typeof<TimeSpan>
           typeof<char>; typeof<byte> |]

/// <summary>
/// A dtype: a dense <see cref="T:YANum.Core.DTypeCode"/> plus a width that is meaningful
/// only for the fixed-width string dtypes.
/// </summary>
/// <remarks>
/// <para>
/// This is a parameterised struct rather than a flat enum because <c>Utf16</c> and
/// <c>Ascii</c> carry an element count. Everything else has <c>Width = 0</c>.
/// </para>
/// <para>
/// <b>Complex naming follows NumPy, not .NET.</b> The number is the total bit width, so
/// <c>Complex64</c> is a pair of <c>float32</c> and <c>Complex128</c> is a pair of
/// <c>float64</c>. This is deliberate and is the opposite of the .NET convention where the
/// number would name the component.
/// </para>
/// </remarks>
[<Struct; StructuralEquality; StructuralComparison>]
type DType =

    /// <summary>The dense discriminator. Kernel tables are indexed by this alone.</summary>
    val Code : DTypeCode

    /// <summary>Element count for <c>Utf16</c>/<c>Ascii</c>; 0 for every fixed dtype.</summary>
    val Width : int

    internal new (code: DTypeCode, width: int) = { Code = code; Width = width }

    /// <summary>Size of one element in bytes.</summary>
    /// <remarks>
    /// A method rather than a plain table lookup because the string dtypes derive their size
    /// from <see cref="P:YANum.Core.DType.Width"/>.
    /// </remarks>
    member this.ItemSize : int =
        match this.Code with
        | DTypeCode.Utf16 -> 2 * this.Width
        | DTypeCode.Ascii -> this.Width
        | c -> DTypeTables.itemSize.[int c]

    /// <summary>The classification used by promotion and kernel selection.</summary>
    member this.Kind : DTypeKind = DTypeTables.kind.[int this.Code]

    /// <summary>The CLR element type. For string dtypes this is the element of the backing buffer.</summary>
    member this.ClrType : Type = DTypeTables.clrTypes.[int this.Code]

    /// <summary>Preferred element alignment in bytes.</summary>
    member this.Alignment : int = DTypeTables.alignment.[int this.Code]

    /// <summary>Whether hardware SIMD kernels are available for this dtype.</summary>
    member this.IsSimdEligible : bool = DTypeTables.simd.[int this.Code]

    /// <summary>
    /// Whether byte comparison agrees with value equality for this dtype.
    /// </summary>
    /// <remarks>
    /// This governs <b>only</b> the structural fast path of <c>array_equal</c>,
    /// <c>NDArray.Equals</c> and cache keys. It never governs element-wise <c>.=</c>, which
    /// always runs the dtype's real equality kernel so that IEEE semantics survive: floats
    /// fail memcmp in both directions, since <c>-0.0</c> and <c>+0.0</c> differ in bytes but
    /// compare equal, while two <c>NaN</c>s share bytes but compare unequal.
    /// </remarks>
    member this.MemcmpLegal : bool = DTypeTables.memcmpLegal.[int this.Code]

    /// <summary>Mantissa bits for floating dtypes (11, 24 or 53); 0 for everything else.</summary>
    member this.MantissaBits : int = DTypeTables.mantissaBits.[int this.Code]

    /// <summary>Whether this dtype participates in arithmetic promotion.</summary>
    member this.IsNumeric : bool =
        match this.Kind with
        | DTypeKind.Boolean | DTypeKind.SignedInt | DTypeKind.UnsignedInt
        | DTypeKind.Float | DTypeKind.Decimal | DTypeKind.Complex -> true
        | _ -> false

    /// <summary>Whether this is one of the two fixed-width string dtypes.</summary>
    member this.IsString : bool = this.Kind = DTypeKind.Str

    /// <summary>Whether this is an integral dtype (signed or unsigned, any width).</summary>
    member this.IsInteger : bool =
        this.Kind = DTypeKind.SignedInt || this.Kind = DTypeKind.UnsignedInt

    /// <summary>Value bit width for integral dtypes; 0 otherwise.</summary>
    member this.IntegerBits : int =
        if this.IsInteger then 8 * DTypeTables.itemSize.[int this.Code] else 0

    /// <summary>The canonical lowercase name, e.g. <c>float64</c>, <c>uint8</c>, <c>utf16&lt;16&gt;</c>.</summary>
    member this.Name : string =
        match this.Code with
        | DTypeCode.Utf16 -> "utf16<" + string this.Width + ">"
        | DTypeCode.Ascii -> "ascii<" + string this.Width + ">"
        | c -> DTypeTables.names.[int c]

    /// <summary>Returns <see cref="P:YANum.Core.DType.Name"/>.</summary>
    /// <returns>The canonical dtype name.</returns>
    override this.ToString() : string = this.Name

    /// <summary>The 1-byte boolean dtype, always stored as 0 or 1.</summary>
    static member Bool = DType(DTypeCode.Bool, 0)
    /// <summary>Signed 8-bit integer.</summary>
    static member Int8 = DType(DTypeCode.Int8, 0)
    /// <summary>Unsigned 8-bit integer.</summary>
    static member UInt8 = DType(DTypeCode.UInt8, 0)
    /// <summary>Unsigned 8-bit integer. The same value as <see cref="P:YANum.Core.DType.UInt8"/>.</summary>
    static member Byte = DType(DTypeCode.UInt8, 0)
    /// <summary>Signed 16-bit integer.</summary>
    static member Int16 = DType(DTypeCode.Int16, 0)
    /// <summary>Unsigned 16-bit integer.</summary>
    static member UInt16 = DType(DTypeCode.UInt16, 0)
    /// <summary>Signed 32-bit integer.</summary>
    static member Int32 = DType(DTypeCode.Int32, 0)
    /// <summary>Unsigned 32-bit integer.</summary>
    static member UInt32 = DType(DTypeCode.UInt32, 0)
    /// <summary>Signed 64-bit integer.</summary>
    static member Int64 = DType(DTypeCode.Int64, 0)
    /// <summary>Unsigned 64-bit integer.</summary>
    static member UInt64 = DType(DTypeCode.UInt64, 0)
    /// <summary>Signed 128-bit integer. No SIMD and no hardware arithmetic.</summary>
    static member Int128 = DType(DTypeCode.Int128, 0)
    /// <summary>Unsigned 128-bit integer. No SIMD and no hardware arithmetic.</summary>
    static member UInt128 = DType(DTypeCode.UInt128, 0)
    /// <summary>IEEE-754 binary16 (<see cref="T:System.Half"/>).</summary>
    static member Float16 = DType(DTypeCode.Float16, 0)
    /// <summary>IEEE-754 binary32.</summary>
    static member Float32 = DType(DTypeCode.Float32, 0)
    /// <summary>IEEE-754 binary64.</summary>
    static member Float64 = DType(DTypeCode.Float64, 0)
    /// <summary>The .NET 128-bit base-10 decimal.</summary>
    static member Decimal = DType(DTypeCode.Decimal, 0)
    /// <summary>Complex with two <c>float32</c> components. NumPy naming: the number is total bits.</summary>
    /// <remarks>This is <b>not</b> the .NET convention; <c>Complex64</c> is a pair of binary32.</remarks>
    static member Complex64 = DType(DTypeCode.Complex64, 0)
    /// <summary>Complex with two <c>float64</c> components. NumPy naming: the number is total bits.</summary>
    /// <remarks>This is <b>not</b> the .NET convention; <c>Complex128</c> is a pair of binary64.</remarks>
    static member Complex128 = DType(DTypeCode.Complex128, 0)
    /// <summary>A single UTF-16 code unit. Promotes as <c>UInt16</c>.</summary>
    static member Char = DType(DTypeCode.Char, 0)
    /// <summary>A <see cref="T:System.DateTime"/>; <c>Kind</c> survives storage and retrieval.</summary>
    static member DateTime = DType(DTypeCode.DateTime, 0)
    /// <summary>A <see cref="T:System.DateTimeOffset"/>.</summary>
    static member DateTimeOffset = DType(DTypeCode.DateTimeOffset, 0)
    /// <summary>A <see cref="T:System.DateOnly"/>.</summary>
    static member DateOnly = DType(DTypeCode.DateOnly, 0)
    /// <summary>A <see cref="T:System.TimeOnly"/>.</summary>
    static member TimeOnly = DType(DTypeCode.TimeOnly, 0)
    /// <summary>A <see cref="T:System.TimeSpan"/>.</summary>
    static member TimeSpan = DType(DTypeCode.TimeSpan, 0)

    /// <summary>A fixed-width UTF-16 string dtype.</summary>
    /// <param name="width">Number of UTF-16 code units per element; must be positive.</param>
    /// <returns>The dtype occupying <c>2 * width</c> bytes per element.</returns>
    /// <exception cref="T:YANum.Core.YANumValueError">The width is not positive.</exception>
    static member Utf16 (width: int) : DType =
        if width <= 0 then
            Errors.valueError (sprintf "DType.Utf16: width must be positive, got %d" width)
        else DType(DTypeCode.Utf16, width)

    /// <summary>A fixed-width ASCII string dtype.</summary>
    /// <param name="width">Number of bytes per element; must be positive.</param>
    /// <returns>The dtype occupying <c>width</c> bytes per element.</returns>
    /// <exception cref="T:YANum.Core.YANumValueError">The width is not positive.</exception>
    static member Ascii (width: int) : DType =
        if width <= 0 then
            Errors.valueError (sprintf "DType.Ascii: width must be positive, got %d" width)
        else DType(DTypeCode.Ascii, width)

    /// <summary>Builds a dtype from a code and a width, validating both.</summary>
    /// <param name="code">The dtype code.</param>
    /// <param name="width">Element count for string dtypes; ignored (and forced to 0) otherwise.</param>
    /// <returns>The dtype.</returns>
    /// <exception cref="T:YANum.Core.YANumValueError">The code is out of range, or a string width is not positive.</exception>
    static member Create (code: DTypeCode, width: int) : DType =
        let c = int code
        if c < 0 || c >= DTypeTables.Count then
            Errors.valueError (sprintf "DType.Create: dtype code %d is out of range 0..%d" c (DTypeTables.Count - 1))
        else
            match code with
            | DTypeCode.Utf16 -> DType.Utf16 width
            | DTypeCode.Ascii -> DType.Ascii width
            | _ -> DType(code, 0)

    /// <summary>Builds a fixed (non-string) dtype from its code.</summary>
    /// <param name="code">The dtype code; must not be <c>Utf16</c> or <c>Ascii</c>.</param>
    /// <returns>The dtype.</returns>
    /// <exception cref="T:YANum.Core.YANumValueError">The code names a width-parameterised string dtype.</exception>
    static member OfCode (code: DTypeCode) : DType =
        match code with
        | DTypeCode.Utf16 | DTypeCode.Ascii ->
            Errors.valueError
                (sprintf "DType.OfCode: %s is width-parameterised; use DType.Utf16 n or DType.Ascii n"
                    DTypeTables.names.[int code])
        | _ -> DType.Create(code, 0)

/// <summary>Constants over <see cref="T:YANum.Core.DTypeCode"/>.</summary>
[<CompilationRepresentation(CompilationRepresentationFlags.ModuleSuffix)>]
module DTypeCode =

    /// <summary>The number of dtype codes. Sizes every flat kernel table.</summary>
    [<Literal>]
    let Count = 25

    /// <summary>Every dtype code, in declaration order.</summary>
    let all : DTypeCode[] = Array.init Count (fun i -> LanguagePrimitives.EnumOfValue<byte, DTypeCode>(byte i))

/// <summary>Reflection-based resolution from a CLR type to its dtype. Call sites cache the result.</summary>
module DTypeTable =

    /// <summary>Maps a CLR element type to its dtype.</summary>
    /// <param name="t">The CLR type.</param>
    /// <returns>The corresponding dtype.</returns>
    /// <exception cref="T:YANum.Core.YANumTypeError">The type has no dtype.</exception>
    let resolve (t: Type) : DType =
        if   t = typeof<bool>           then DType.Bool
        elif t = typeof<sbyte>          then DType.Int8
        elif t = typeof<byte>           then DType.UInt8
        elif t = typeof<int16>          then DType.Int16
        elif t = typeof<uint16>         then DType.UInt16
        elif t = typeof<int32>          then DType.Int32
        elif t = typeof<uint32>         then DType.UInt32
        elif t = typeof<int64>          then DType.Int64
        elif t = typeof<uint64>         then DType.UInt64
        elif t = typeof<Int128>         then DType.Int128
        elif t = typeof<UInt128>        then DType.UInt128
        elif t = typeof<Half>           then DType.Float16
        elif t = typeof<float32>        then DType.Float32
        elif t = typeof<float>          then DType.Float64
        elif t = typeof<decimal>        then DType.Decimal
        elif t = typeof<Complex32>      then DType.Complex64
        elif t = typeof<Complex>        then DType.Complex128
        elif t = typeof<char>           then DType.Char
        elif t = typeof<DateTime>       then DType.DateTime
        elif t = typeof<DateTimeOffset> then DType.DateTimeOffset
        elif t = typeof<DateOnly>       then DType.DateOnly
        elif t = typeof<TimeOnly>       then DType.TimeOnly
        elif t = typeof<TimeSpan>       then DType.TimeSpan
        else
            Errors.typeError
                (sprintf "no YANum dtype corresponds to the CLR type '%s'; supported element types are \
                          bool, sbyte, byte, int16, uint16, int32, uint32, int64, uint64, Int128, UInt128, \
                          Half, float32, float, decimal, Complex32, Complex, char, DateTime, DateTimeOffset, \
                          DateOnly, TimeOnly and TimeSpan" t.FullName)

/// <summary>
/// The compile-time bridge from an element type to its dtype.
/// </summary>
/// <remarks>
/// <para>
/// <b>This does not constant-fold and cannot be made to</b> (Phase 0, spike 0.10): F# never
/// emits <c>initonly</c> static fields, so the JIT cannot promote the read to a constant.
/// The measured cost is about 0.06 ns per access.
/// </para>
/// <para>
/// <b>Read it once per operation and hoist it out of every loop.</b> A per-element read is a
/// defect and is treated as one in review.
/// </para>
/// </remarks>
type DTypeOf<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType> private () =

    /// <summary>The dtype of <c>'T</c>, resolved once per closed generic instantiation.</summary>
    static member val Value : DType = DTypeTable.resolve typeof<'T> with get
