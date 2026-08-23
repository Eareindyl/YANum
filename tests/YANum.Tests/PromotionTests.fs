namespace YANum.Tests

open System
open Xunit
open YANum.Core

module PromotionTests =

    let private codes = DTypeCode.all
    let private dt (c: DTypeCode) = Generators.ofCode 4 c

    /// After rules P2 and P3, Char behaves as UInt16 and Bool as UInt8, so both count as
    /// unsigned for the purpose of characterising where the lattice loses associativity.
    let private isUnsignedish (c: DTypeCode) =
        let k = (dt c).Kind
        k = DTypeKind.UnsignedInt || c = DTypeCode.Bool || c = DTypeCode.Char

    let private isSigned (c: DTypeCode) = (dt c).Kind = DTypeKind.SignedInt

    /// The documented corner: a triple that mixes a signed integer with an unsigned one.
    /// Rules P6 and P7 are both defined there, and they disagree about association.
    let private mixesSignedness (a: DTypeCode) (b: DTypeCode) (c: DTypeCode) =
        let all = [ a; b; c ]
        List.exists isSigned all && List.exists isUnsignedish all

    // -----------------------------------------------------------------------------------
    //  Section 7.2 properties
    // -----------------------------------------------------------------------------------

    [<Fact>]
    let ``promotion is commutative over all 25 by 25 code pairs`` () =
        for a in codes do
            for b in codes do
                let x = Promotion.tryPromote (dt a) (dt b)
                let y = Promotion.tryPromote (dt b) (dt a)
                Assert.True((x = y), sprintf "promote %A %A gave %A but the reverse gave %A" a b x y)

    [<Fact>]
    let ``promotion is commutative over a sample of string widths`` () =
        let widths = [ 1; 2; 4; 16 ]
        let strDTypes =
            [ for w in widths do
                yield DType.Utf16 w
                yield DType.Ascii w ]
        let others = strDTypes @ [ DType.Float64; DType.Int32; DType.Bool; DType.DateTime ]
        for a in strDTypes do
            for b in others do
                Assert.True((Promotion.tryPromote a b = Promotion.tryPromote b a),
                            sprintf "promote %s %s is not commutative" a.Name b.Name)

    [<Fact>]
    let ``promotion is idempotent for all 25 dtypes`` () =
        for c in codes do
            let d = dt c
            Assert.Equal(d, Promotion.promote d d)
        for d in Generators.stringDTypes do
            Assert.Equal(d, Promotion.promote d d)

    [<Fact>]
    let ``promotion is associative outside the mixed-signedness corner`` () =
        let mutable checked' = 0
        for a in codes do
            for b in codes do
                for c in codes do
                    if not (mixesSignedness a b c) then
                        match Promotion.tryPromote (dt a) (dt b) with
                        | ValueSome ab ->
                            match Promotion.tryPromote ab (dt c),
                                  Promotion.tryPromote (dt b) (dt c) with
                            | ValueSome left, ValueSome bc ->
                                match Promotion.tryPromote (dt a) bc with
                                | ValueSome right ->
                                    checked' <- checked' + 1
                                    Assert.True((left = right),
                                        sprintf "(%A,%A,%A): left association gave %s, right gave %s"
                                            a b c left.Name right.Name)
                                | ValueNone -> ()
                            | _ -> ()
                        | ValueNone -> ()
        Assert.True(checked' > 1000, sprintf "only %d triples were actually exercised" checked')

    /// <summary>
    /// The lattice is <b>not</b> fully associative, and cannot be while P7 reproduces NumPy's
    /// int-with-float table and P6 falls back to <c>Float64</c>. This test pins the exact set
    /// of violations so that a change to either rule shows up immediately, and asserts that
    /// every one of them lives in the documented mixed-signedness corner.
    /// </summary>
    [<Fact>]
    let ``every associativity violation mixes signed and unsigned integers`` () =
        let violations = ResizeArray<DTypeCode * DTypeCode * DTypeCode>()
        for a in codes do
            for b in codes do
                for c in codes do
                    match Promotion.tryPromote (dt a) (dt b) with
                    | ValueSome ab ->
                        match Promotion.tryPromote ab (dt c), Promotion.tryPromote (dt b) (dt c) with
                        | ValueSome left, ValueSome bc ->
                            match Promotion.tryPromote (dt a) bc with
                            | ValueSome right when left <> right -> violations.Add((a, b, c))
                            | _ -> ()
                        | _ -> ()
                    | ValueNone -> ()
        for (a, b, c) in violations do
            Assert.True(mixesSignedness a b c,
                sprintf "(%A,%A,%A) breaks associativity outside the documented corner" a b c)
        Assert.Equal(72, violations.Count)

    [<Fact>]
    let ``every promotion failure is a YANumTypeError naming both dtypes`` () =
        for a in codes do
            for b in codes do
                if (Promotion.tryPromote (dt a) (dt b)).IsNone then
                    let ex = Assert.Throws<YANumTypeError>(fun () -> Promotion.promote (dt a) (dt b) |> ignore)
                    Assert.Contains((dt a).Name, ex.Message)
                    Assert.Contains((dt b).Name, ex.Message)

    // -----------------------------------------------------------------------------------
    //  Individual rules
    // -----------------------------------------------------------------------------------

    [<Fact>]
    let ``P2 Char promotes as UInt16`` () =
        Assert.Equal(DType.UInt16, Promotion.promote DType.Char DType.UInt8)
        Assert.Equal(DType.Int32, Promotion.promote DType.Char DType.Int16)
        Assert.Equal(DType.Char, Promotion.promote DType.Char DType.Char)

    [<Fact>]
    let ``P3 Bool promotes as UInt8`` () =
        Assert.Equal(DType.UInt8, Promotion.promote DType.Bool DType.UInt8)
        Assert.Equal(DType.Int16, Promotion.promote DType.Bool DType.Int8)
        Assert.Equal(DType.Bool, Promotion.promote DType.Bool DType.Bool)

    [<Fact>]
    let ``P4 and P5 take the wider integer of the same signedness`` () =
        Assert.Equal(DType.Int64, Promotion.promote DType.Int8 DType.Int64)
        Assert.Equal(DType.Int128, Promotion.promote DType.Int128 DType.Int32)
        Assert.Equal(DType.UInt32, Promotion.promote DType.UInt8 DType.UInt32)
        Assert.Equal(DType.UInt128, Promotion.promote DType.UInt128 DType.UInt8)

    [<Fact>]
    let ``P6 finds the smallest signed integer that covers both`` () =
        Assert.Equal(DType.Int16, Promotion.promote DType.UInt8 DType.Int8)
        Assert.Equal(DType.Int32, Promotion.promote DType.UInt16 DType.Int16)
        Assert.Equal(DType.Int64, Promotion.promote DType.UInt32 DType.Int32)
        Assert.Equal(DType.Int64, Promotion.promote DType.UInt32 DType.Int64)

    [<Fact>]
    let ``P6 falls back to Float64 when no signed integer covers both`` () =
        Assert.Equal(DType.Float64, Promotion.promote DType.UInt64 DType.Int64)
        Assert.Equal(DType.Float64, Promotion.promote DType.UInt64 DType.Int8)
        Assert.Equal(DType.Float64, Promotion.promote DType.UInt128 DType.Int64)
        Assert.Equal(DType.Float64, Promotion.promote DType.UInt128 DType.Int128)

    /// Int128 is preserved when it is already an operand, but is never used to widen out of a
    /// 64-bit unsigned: that is what keeps YANum aligned with NumPy on the UInt64 cells.
    [<Fact>]
    let ``P6 keeps an existing Int128 rather than dropping to Float64`` () =
        Assert.Equal(DType.Int128, Promotion.promote DType.UInt8 DType.Int128)
        Assert.Equal(DType.Int128, Promotion.promote DType.UInt64 DType.Int128)

    [<Fact>]
    let ``P6 raises instead of falling back under strict integer promotion`` () =
        Assert.Equal(DType.Float64, Promotion.promote DType.UInt64 DType.Int64)
        use _ = Config.push (fun c -> c.StrictIntegerPromotion <- true)
        let ex = Assert.Throws<YANumTypeError>(fun () -> Promotion.promote DType.UInt64 DType.Int64 |> ignore)
        Assert.Contains("uint64", ex.Message)
        Assert.Contains("int64", ex.Message)
        Assert.Contains("StrictIntegerPromotion", ex.Message)
        // the cells that do have an exact answer are unaffected
        Assert.Equal(DType.Int16, Promotion.promote DType.UInt8 DType.Int8)

    [<Fact>]
    let ``P7 reproduces NumPy's int-with-float table`` () =
        Assert.Equal(DType.Float16, Promotion.promote DType.Int8 DType.Float16)
        Assert.Equal(DType.Float32, Promotion.promote DType.Int16 DType.Float16)
        Assert.Equal(DType.Float64, Promotion.promote DType.Int32 DType.Float32)
        Assert.Equal(DType.Float64, Promotion.promote DType.Int64 DType.Float32)
        Assert.Equal(DType.Float64, Promotion.promote DType.Int32 DType.Float64)
        Assert.Equal(DType.Float64, Promotion.promote DType.Int128 DType.Float16)

    [<Fact>]
    let ``P8 takes the wider float`` () =
        Assert.Equal(DType.Float64, Promotion.promote DType.Float16 DType.Float64)
        Assert.Equal(DType.Float32, Promotion.promote DType.Float16 DType.Float32)

    [<Fact>]
    let ``P9 lifts a float against a complex to the matching complex`` () =
        Assert.Equal(DType.Complex64, Promotion.promote DType.Float32 DType.Complex64)
        Assert.Equal(DType.Complex64, Promotion.promote DType.Float16 DType.Complex64)
        Assert.Equal(DType.Complex128, Promotion.promote DType.Float64 DType.Complex64)
        Assert.Equal(DType.Complex128, Promotion.promote DType.Float32 DType.Complex128)

    [<Fact>]
    let ``P10 takes the wider complex`` () =
        Assert.Equal(DType.Complex128, Promotion.promote DType.Complex64 DType.Complex128)

    [<Fact>]
    let ``P11 promotes an integer through the complex component type`` () =
        Assert.Equal(DType.Complex64, Promotion.promote DType.Int8 DType.Complex64)
        Assert.Equal(DType.Complex128, Promotion.promote DType.Int32 DType.Complex64)
        Assert.Equal(DType.Complex128, Promotion.promote DType.Int64 DType.Complex128)

    [<Fact>]
    let ``P12 Decimal absorbs integers up to 64 bits`` () =
        Assert.Equal(DType.Decimal, Promotion.promote DType.Decimal DType.Bool)
        Assert.Equal(DType.Decimal, Promotion.promote DType.Decimal DType.Int8)
        Assert.Equal(DType.Decimal, Promotion.promote DType.Decimal DType.UInt64)
        Assert.Equal(DType.Decimal, Promotion.promote DType.Decimal DType.Char)

    [<Fact>]
    let ``P13 Decimal against a float complex or 128-bit integer raises`` () =
        for other in [ DType.Float16; DType.Float32; DType.Float64
                       DType.Complex64; DType.Complex128; DType.Int128; DType.UInt128 ] do
            Assert.Throws<YANumTypeError>(fun () -> Promotion.promote DType.Decimal other |> ignore) |> ignore

    [<Fact>]
    let ``P14 and P15 keep temporal dtypes to themselves`` () =
        let temporal = [ DType.DateTime; DType.DateTimeOffset; DType.DateOnly; DType.TimeOnly; DType.TimeSpan ]
        for a in temporal do
            Assert.Equal(a, Promotion.promote a a)
            for b in temporal do
                if a <> b then
                    Assert.Throws<YANumTypeError>(fun () -> Promotion.promote a b |> ignore) |> ignore
            for b in [ DType.Int64; DType.Float64; DType.Bool; DType.Utf16 4 ] do
                Assert.Throws<YANumTypeError>(fun () -> Promotion.promote a b |> ignore) |> ignore

    [<Fact>]
    let ``P16 to P18 take the wider string, widening Ascii to Utf16`` () =
        Assert.Equal(DType.Utf16 8, Promotion.promote (DType.Utf16 3) (DType.Utf16 8))
        Assert.Equal(DType.Ascii 8, Promotion.promote (DType.Ascii 8) (DType.Ascii 3))
        Assert.Equal(DType.Utf16 8, Promotion.promote (DType.Ascii 8) (DType.Utf16 3))
        Assert.Equal(DType.Utf16 8, Promotion.promote (DType.Utf16 3) (DType.Ascii 8))

    [<Fact>]
    let ``promotion is not concatenation`` () =
        // Utf16 a + Utf16 b -> Utf16 (a+b) belongs to Signature.resolve in Phase 3, not here.
        Assert.Equal(DType.Utf16 5, Promotion.promote (DType.Utf16 5) (DType.Utf16 3))

    [<Fact>]
    let ``P19 a string against a non-string raises`` () =
        for other in [ DType.Bool; DType.Int64; DType.Float64; DType.Char; DType.Decimal; DType.DateTime ] do
            Assert.Throws<YANumTypeError>(fun () -> Promotion.promote (DType.Utf16 4) other |> ignore) |> ignore
            Assert.Throws<YANumTypeError>(fun () -> Promotion.promote (DType.Ascii 4) other |> ignore) |> ignore

    [<Fact>]
    let ``canPromote agrees with promote`` () =
        for a in codes do
            for b in codes do
                let ok = Promotion.canPromote (dt a) (dt b)
                if ok then Promotion.promote (dt a) (dt b) |> ignore
                else Assert.Throws<YANumTypeError>(fun () -> Promotion.promote (dt a) (dt b) |> ignore) |> ignore
