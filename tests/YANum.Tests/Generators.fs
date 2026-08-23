namespace YANum.Tests

open System
open System.Runtime.InteropServices
open FsCheck
open FsCheck.FSharp
open YANum.Core

/// A shape drawn from ranks 0..5 with extents 0..4.
type ArrayShape = ArrayShape of int[]

/// A shape with no zero extent, for tests that need a non-empty array.
type NonEmptyShape = NonEmptyShape of int[]

/// <summary>
/// Shared FsCheck generators and dtype tables. Phases 2 to 5 reuse these heavily, so they live
/// here rather than beside any one test.
/// </summary>
module Generators =

    /// The 23 dtypes that carry no width.
    let fixedDTypes : DType[] =
        [| DType.Bool; DType.Int8; DType.UInt8; DType.Int16; DType.UInt16
           DType.Int32; DType.UInt32; DType.Int64; DType.UInt64
           DType.Int128; DType.UInt128
           DType.Float16; DType.Float32; DType.Float64; DType.Decimal
           DType.Complex64; DType.Complex128; DType.Char
           DType.DateTime; DType.DateTimeOffset; DType.DateOnly; DType.TimeOnly; DType.TimeSpan |]

    /// All 25 dtype codes, with the two string dtypes at a representative width.
    let allDTypes : DType[] =
        Array.append fixedDTypes [| DType.Utf16 4; DType.Ascii 4 |]

    /// The string dtypes at several widths, for the round-trip and promotion tests.
    let stringDTypes : DType[] =
        [| DType.Utf16 1; DType.Utf16 4; DType.Utf16 16; DType.Ascii 1; DType.Ascii 4; DType.Ascii 16 |]

    /// Every dtype code, including the two width-parameterised ones at width 4.
    let allCodes : DTypeCode[] = DTypeCode.all

    /// Builds a dtype from a code, giving string codes a width.
    let ofCode (width: int) (c: DTypeCode) : DType =
        match c with
        | DTypeCode.Utf16 -> DType.Utf16 width
        | DTypeCode.Ascii -> DType.Ascii width
        | _ -> DType.OfCode c

    let genOrder : Gen<Order> = Gen.elements [ C; F ]

    let genOrderK : Gen<Order> = Gen.elements [ C; F; K ]

    let genBackend : Gen<StorageBackend> = Gen.elements [ Managed; Native ]

    let genDType : Gen<DType> = Gen.elements (List.ofArray allDTypes)

    let genFixedDType : Gen<DType> = Gen.elements (List.ofArray fixedDTypes)

    let genShape : Gen<int[]> =
        gen {
            let! rank = Gen.choose (0, 5)
            let! extents = Gen.arrayOfLength rank (Gen.choose (0, 4))
            return extents
        }

    let genNonEmptyShape : Gen<int[]> =
        gen {
            let! rank = Gen.choose (0, 5)
            let! extents = Gen.arrayOfLength rank (Gen.choose (1, 4))
            return extents
        }

    let arbShape : Arbitrary<ArrayShape> = Arb.fromGen (Gen.map ArrayShape genShape)
    let arbNonEmptyShape : Arbitrary<NonEmptyShape> = Arb.fromGen (Gen.map NonEmptyShape genNonEmptyShape)
    let arbOrder : Arbitrary<Order> = Arb.fromGen genOrder
    let arbDType : Arbitrary<DType> = Arb.fromGen genDType

    /// Registered with FsCheck.Xunit through [<Property(Arbitrary = ...)>].
    type Arbitraries =
        static member ArrayShape () = arbShape
        static member NonEmptyShape () = arbNonEmptyShape
        static member Order () = arbOrder
        static member DType () = arbDType

/// Byte-level helpers used by the bit-exactness assertions.
module Bytes =

    let ofValue<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType>
        (v: 'T) : byte[] =
        MemoryMarshal.AsBytes(ReadOnlySpan<'T>([| v |])).ToArray()

    let ofArray<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType>
        (v: 'T[]) : byte[] =
        MemoryMarshal.AsBytes(ReadOnlySpan<'T>(v)).ToArray()
