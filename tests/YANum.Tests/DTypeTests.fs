namespace YANum.Tests

open System
open System.Numerics
open Xunit
open YANum.Core

module DTypeTests =

    // -----------------------------------------------------------------------------------
    //  Metadata table (specification section 6.2)
    // -----------------------------------------------------------------------------------

    /// dtype, name, item size, kind, SIMD-eligible, memcmp-legal
    let table : obj[] seq =
        [ [| box DType.Bool;           box "bool";           box 1;  box DTypeKind.Boolean;     box true;  box true  |]
          [| box DType.Int8;           box "int8";           box 1;  box DTypeKind.SignedInt;   box true;  box true  |]
          [| box DType.UInt8;          box "uint8";          box 1;  box DTypeKind.UnsignedInt; box true;  box true  |]
          [| box DType.Int16;          box "int16";          box 2;  box DTypeKind.SignedInt;   box true;  box true  |]
          [| box DType.UInt16;         box "uint16";         box 2;  box DTypeKind.UnsignedInt; box true;  box true  |]
          [| box DType.Int32;          box "int32";          box 4;  box DTypeKind.SignedInt;   box true;  box true  |]
          [| box DType.UInt32;         box "uint32";         box 4;  box DTypeKind.UnsignedInt; box true;  box true  |]
          [| box DType.Int64;          box "int64";          box 8;  box DTypeKind.SignedInt;   box true;  box true  |]
          [| box DType.UInt64;         box "uint64";         box 8;  box DTypeKind.UnsignedInt; box true;  box true  |]
          [| box DType.Int128;         box "int128";         box 16; box DTypeKind.SignedInt;   box false; box true  |]
          [| box DType.UInt128;        box "uint128";        box 16; box DTypeKind.UnsignedInt; box false; box true  |]
          [| box DType.Float16;        box "float16";        box 2;  box DTypeKind.Float;       box false; box false |]
          [| box DType.Float32;        box "float32";        box 4;  box DTypeKind.Float;       box true;  box false |]
          [| box DType.Float64;        box "float64";        box 8;  box DTypeKind.Float;       box true;  box false |]
          [| box DType.Decimal;        box "decimal";        box 16; box DTypeKind.Decimal;     box false; box false |]
          [| box DType.Complex64;      box "complex64";      box 8;  box DTypeKind.Complex;     box false; box false |]
          [| box DType.Complex128;     box "complex128";     box 16; box DTypeKind.Complex;     box false; box false |]
          [| box DType.Char;           box "char";           box 2;  box DTypeKind.Char;        box true;  box true  |]
          [| box DType.DateTime;       box "datetime";       box 8;  box DTypeKind.Instant;     box false; box false |]
          [| box DType.DateTimeOffset; box "datetimeoffset"; box 16; box DTypeKind.Instant;     box false; box false |]
          [| box DType.DateOnly;       box "dateonly";       box 4;  box DTypeKind.Instant;     box false; box true  |]
          [| box DType.TimeOnly;       box "timeonly";       box 8;  box DTypeKind.Instant;     box false; box true  |]
          [| box DType.TimeSpan;       box "timespan";       box 8;  box DTypeKind.Duration;    box false; box true  |] ]

    [<Theory; MemberData(nameof table)>]
    let ``dtype metadata matches the specification table``
        (dt: DType) (name: string) (itemSize: int) (kind: DTypeKind) (simd: bool) (memcmp: bool) =
        Assert.Equal(name, dt.Name)
        Assert.Equal(itemSize, dt.ItemSize)
        Assert.Equal(kind, dt.Kind)
        Assert.Equal(simd, dt.IsSimdEligible)
        Assert.Equal(memcmp, dt.MemcmpLegal)

    [<Fact>]
    let ``there are exactly 25 dtype codes`` () =
        Assert.Equal(25, DTypeCode.Count)
        Assert.Equal(25, DTypeCode.all.Length)
        Assert.Equal(25, Generators.allDTypes.Length)

    [<Fact>]
    let ``every fixed dtype item size agrees with the CLR element size`` () =
        // Unsafe.SizeOf, not Marshal.SizeOf: the latter reports the marshalled size and
        // refuses DateTime outright, while YANum stores the in-memory representation.
        let sizeOf = typeof<System.Runtime.CompilerServices.Unsafe>.GetMethod "SizeOf"
        for dt in Generators.fixedDTypes do
            let actual = sizeOf.MakeGenericMethod(dt.ClrType).Invoke(null, [||]) :?> int
            Assert.Equal(dt.ItemSize, actual)

    [<Fact>]
    let ``string dtypes size from their width`` () =
        Assert.Equal(2, (DType.Utf16 1).ItemSize)
        Assert.Equal(32, (DType.Utf16 16).ItemSize)
        Assert.Equal(1, (DType.Ascii 1).ItemSize)
        Assert.Equal(16, (DType.Ascii 16).ItemSize)
        Assert.Equal("utf16<16>", (DType.Utf16 16).Name)
        Assert.Equal("ascii<4>", (DType.Ascii 4).Name)
        Assert.True (DType.Utf16 3).IsString
        Assert.True (DType.Ascii 3).IsString
        Assert.True (DType.Utf16 3).MemcmpLegal

    [<Fact>]
    let ``a non-positive string width is rejected`` () =
        Assert.Throws<YANumValueError>(fun () -> DType.Utf16 0 |> ignore) |> ignore
        Assert.Throws<YANumValueError>(fun () -> DType.Ascii -1 |> ignore) |> ignore

    [<Fact>]
    let ``UInt8 and Byte are the same value`` () =
        Assert.Equal(DType.UInt8, DType.Byte)

    [<Fact>]
    let ``complex naming follows NumPy: the number is total bits`` () =
        Assert.Equal(typeof<Complex32>, DType.Complex64.ClrType)
        Assert.Equal(8, DType.Complex64.ItemSize)
        Assert.Equal(typeof<Complex>, DType.Complex128.ClrType)
        Assert.Equal(16, DType.Complex128.ItemSize)

    [<Fact>]
    let ``mantissa bits are 11 24 53`` () =
        Assert.Equal(11, DType.Float16.MantissaBits)
        Assert.Equal(24, DType.Float32.MantissaBits)
        Assert.Equal(53, DType.Float64.MantissaBits)
        Assert.Equal(0, DType.Int32.MantissaBits)

    [<Fact>]
    let ``resolve rejects an unsupported CLR type by name`` () =
        let ex = Assert.Throws<YANumTypeError>(fun () -> DTypeTable.resolve typeof<System.Guid> |> ignore)
        Assert.Contains("Guid", ex.Message)

    // -----------------------------------------------------------------------------------
    //  Round-tripping every dtype
    // -----------------------------------------------------------------------------------

    let private roundTrip<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType>
        (expected: DType) (values: 'T[]) =
        Assert.Equal(expected, DTypeOf<'T>.Value)

        // rank 1: 'T[] -> NDArray -> 'T[] is bit-exact
        let a = NDArray.ofArray values
        Assert.Equal(expected, a.DType)
        Assert.Equal(int64 values.Length, a.Size)
        Assert.Equal<byte[]>(Bytes.ofArray values, Bytes.ofArray (a.ToArray<'T>()))

        // element get and set, including negative indices
        Assert.Equal<byte[]>(Bytes.ofValue values.[0], Bytes.ofValue (a.GetItem<'T> [| 0 |]))
        Assert.Equal<byte[]>(Bytes.ofValue values.[values.Length - 1], Bytes.ofValue (a.GetItem<'T> [| -1 |]))
        a.SetItem([| 0 |], values.[values.Length - 1])
        Assert.Equal<byte[]>(Bytes.ofValue values.[values.Length - 1], Bytes.ofValue (a.GetItem<'T> [| 0 |]))

        // rank 2 in both orders: 'T[,] -> NDArray -> 'T[,] is bit-exact
        let n = values.Length
        let src2 = Array2D.init 2 n (fun i j -> values.[(i + j) % n])
        for order in [ C; F ] do
            let m = NDArray.ofArray2D (src2, order = order)
            Assert.Equal(expected, m.DType)
            Assert.Equal<int[]>([| 2; n |], m.Shape)
            if order = C then Assert.True m.IsCContiguous else Assert.True m.IsFContiguous
            let back = m.ToArray2D<'T>()
            for i in 0 .. 1 do
                for j in 0 .. n - 1 do
                    Assert.Equal<byte[]>(Bytes.ofValue src2.[i, j], Bytes.ofValue back.[i, j])
            // transpose, copy and reshape preserve every byte
            Assert.Equal<byte[]>(Bytes.ofArray (m.ToArray<'T>()), Bytes.ofArray (m.Copy(order).ToArray<'T>()))
            Assert.Equal<byte[]>(Bytes.ofArray (m.ToArray<'T>()), Bytes.ofArray (m.T.T.ToArray<'T>()))
            Assert.Equal<byte[]>(Bytes.ofArray (m.ToArray<'T>()),
                                 Bytes.ofArray (m.Reshape([| n; 2 |]).Reshape([| 2; n |]).ToArray<'T>()))
            // printing must work for every dtype in both orders
            Assert.False(String.IsNullOrWhiteSpace(m.ToString()))

    [<Fact>]
    let ``round-trip bool`` () =
        roundTrip DType.Bool [| true; false; true |]
    [<Fact>]
    let ``round-trip int8`` () =
        roundTrip DType.Int8 [| -128y; 0y; 127y |]
    [<Fact>]
    let ``round-trip uint8`` () =
        roundTrip DType.UInt8 [| 0uy; 128uy; 255uy |]
    [<Fact>]
    let ``round-trip int16`` () =
        roundTrip DType.Int16 [| Int16.MinValue; 0s; Int16.MaxValue |]
    [<Fact>]
    let ``round-trip uint16`` () =
        roundTrip DType.UInt16 [| 0us; 40000us; UInt16.MaxValue |]
    [<Fact>]
    let ``round-trip int32`` () =
        roundTrip DType.Int32 [| Int32.MinValue; 0; Int32.MaxValue |]
    [<Fact>]
    let ``round-trip uint32`` () =
        roundTrip DType.UInt32 [| 0u; 3000000000u; UInt32.MaxValue |]
    [<Fact>]
    let ``round-trip int64`` () =
        roundTrip DType.Int64 [| Int64.MinValue; 0L; Int64.MaxValue |]
    [<Fact>]
    let ``round-trip uint64`` () =
        roundTrip DType.UInt64 [| 0UL; 12345UL; UInt64.MaxValue |]
    [<Fact>]
    let ``round-trip int128`` () =
        roundTrip DType.Int128 [| Int128.MinValue; Int128.Zero; Int128.MaxValue |]
    [<Fact>]
    let ``round-trip uint128`` () =
        roundTrip DType.UInt128 [| UInt128.Zero; UInt128.One; UInt128.MaxValue |]
    [<Fact>]
    let ``round-trip float16`` () =
        roundTrip DType.Float16 [| Half.MinValue; Half.Epsilon; Half.NaN |]
    [<Fact>]
    let ``round-trip float32`` () =
        roundTrip DType.Float32 [| Single.NegativeInfinity; -0.0f; Single.NaN |]
    [<Fact>]
    let ``round-trip float64`` () =
        roundTrip DType.Float64 [| Double.NegativeInfinity; -0.0; Double.NaN |]
    [<Fact>]
    let ``round-trip decimal`` () =
        roundTrip DType.Decimal [| Decimal.MinValue; 0m; 1.00m |]

    [<Fact>]
    let ``round-trip complex64`` () =
        roundTrip DType.Complex64 [| Complex32(1.5f, -2.5f); Complex32(0.0f, 0.0f); Complex32(Single.NaN, 1.0f) |]

    [<Fact>]
    let ``round-trip complex128`` () =
        roundTrip DType.Complex128 [| Complex(1.5, -2.5); Complex(0.0, 0.0); Complex(Double.NaN, 1.0) |]

    [<Fact>]
    let ``round-trip char`` () =
        roundTrip DType.Char [| 'a'; 'é'; ' ' |]

    [<Fact>]
    let ``round-trip datetime`` () =
        roundTrip DType.DateTime [| DateTime.MinValue; DateTime(2024, 2, 29, 13, 45, 30); DateTime.MaxValue |]

    [<Fact>]
    let ``round-trip datetimeoffset`` () =
        roundTrip DType.DateTimeOffset
            [| DateTimeOffset.MinValue
               DateTimeOffset(DateTime(2024, 2, 29, 13, 45, 30), TimeSpan.FromHours 5.5)
               DateTimeOffset.MaxValue |]

    [<Fact>]
    let ``round-trip dateonly`` () =
        roundTrip DType.DateOnly [| DateOnly.MinValue; DateOnly(2024, 2, 29); DateOnly.MaxValue |]

    [<Fact>]
    let ``round-trip timeonly`` () =
        roundTrip DType.TimeOnly [| TimeOnly.MinValue; TimeOnly(13, 45, 30); TimeOnly.MaxValue |]

    [<Fact>]
    let ``round-trip timespan`` () =
        roundTrip DType.TimeSpan [| TimeSpan.MinValue; TimeSpan.Zero; TimeSpan.MaxValue |]

    [<Fact>]
    let ``DateTime Kind survives the round trip`` () =
        for kind in [ DateTimeKind.Utc; DateTimeKind.Local; DateTimeKind.Unspecified ] do
            let v = DateTime(2024, 2, 29, 13, 45, 30, kind)
            let a = NDArray.ofArray [| v |]
            let back = a.GetItem<DateTime> [| 0 |]
            Assert.Equal(kind, back.Kind)
            Assert.Equal(v.Ticks, back.Ticks)
            Assert.Equal<byte[]>(Bytes.ofValue v, Bytes.ofValue back)

    [<Fact>]
    let ``an element type that disagrees with the dtype is refused`` () =
        let a = NDArray.ofArray [| 1.0; 2.0 |]
        Assert.Throws<YANumTypeError>(fun () -> a.GetItem<int64> [| 0 |] |> ignore) |> ignore
        Assert.Throws<YANumTypeError>(fun () -> a.ToArray<float32>() |> ignore) |> ignore
