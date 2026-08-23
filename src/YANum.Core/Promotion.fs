namespace YANum.Core

open System

/// <summary>
/// The common-type lattice: given two dtypes, the single dtype both can be represented in.
/// </summary>
/// <remarks>
/// <para>
/// This is <b>not</b> the same question as "what dtype does this ufunc return"; that is
/// <c>Signature.resolve</c>, which arrives in Phase 3. In particular string concatenation
/// (<c>Utf16 a</c> + <c>Utf16 b</c> giving <c>Utf16 (a+b)</c>) is a ufunc-specific result
/// type. <c>promote</c> returns <c>Utf16 (max a b)</c>, which is the right answer for
/// comparison and copy.
/// </para>
/// <para>
/// The non-string part of the lattice is memoised in two flat 25x25 tables indexed by
/// <see cref="T:YANum.Core.DTypeCode"/> — one for the default integer rule and one for
/// <see cref="P:YANum.Core.Config.StrictIntegerPromotion"/>. String widths are handled by a
/// small special case outside the tables.
/// </para>
/// </remarks>
module Promotion =

    /// The result ladder for rule P6.
    ///
    /// `Int128` is present but is never reached by *escalation*: see `p6Allows` below. YANum
    /// matches NumPy by refusing to promote a 64-bit unsigned into a 128-bit signed integer,
    /// because `Int128` has neither SIMD nor hardware arithmetic and the choice is viral
    /// through the rest of the pipeline. `UInt64` and `UInt128` against a signed integer of at
    /// most 64 bits therefore fall back to `Float64`, or raise under strict integer promotion.
    let private signedLadder =
        [| DTypeCode.Int8; DTypeCode.Int16; DTypeCode.Int32; DTypeCode.Int64; DTypeCode.Int128 |]

    /// P6 admits `Int128` as a result only when the signed operand is already `Int128`; it is
    /// never used to widen out of a 64-bit unsigned. Without this the lattice loses information
    /// gratuitously (`Int128` against `UInt8` would fall all the way to `Float64`) and breaks
    /// associativity on triples that never mention a float.
    let inline private p6Allows (candidate: DTypeCode) (signedBits: int) =
        candidate <> DTypeCode.Int128 || signedBits = 128

    let private floatLadder = [| DTypeCode.Float16; DTypeCode.Float32; DTypeCode.Float64 |]

    let inline private bitsOf (c: DTypeCode) = 8 * DTypeTables.itemSize.[int c]

    let inline private kindOf (c: DTypeCode) = DTypeTables.kind.[int c]

    let inline private isIntKind (k: DTypeKind) =
        k = DTypeKind.SignedInt || k = DTypeKind.UnsignedInt

    /// The float component type of a complex dtype.
    let private componentOf (c: DTypeCode) =
        match c with
        | DTypeCode.Complex64 -> DTypeCode.Float32
        | _ -> DTypeCode.Float64

    /// The complex dtype whose components are at least the given float dtype.
    let private liftToComplex (f: DTypeCode) =
        match f with
        | DTypeCode.Float64 -> DTypeCode.Complex128
        | _ -> DTypeCode.Complex64

    /// Smallest float whose mantissa covers `bits` value bits, capped at Float64 (rule P7).
    let private floatForIntBits (bits: int) =
        let mutable r = DTypeCode.Float64
        let mutable i = floatLadder.Length - 1
        while i >= 0 do
            if DTypeTables.mantissaBits.[int floatLadder.[i]] >= bits then r <- floatLadder.[i]
            i <- i - 1
        r

    /// Core rule engine. Returns the resulting dtype code as an int, or -1 for "no promotion".
    let rec private compute (a: DTypeCode) (b: DTypeCode) (strict: bool) : int =
        // P1
        if a = b then int a
        // P2 — Char behaves as UInt16 against anything that is not Char
        elif a = DTypeCode.Char then compute DTypeCode.UInt16 b strict
        elif b = DTypeCode.Char then compute a DTypeCode.UInt16 strict
        // P3 — Bool behaves as UInt8 against anything that is not Bool
        elif a = DTypeCode.Bool then compute DTypeCode.UInt8 b strict
        elif b = DTypeCode.Bool then compute a DTypeCode.UInt8 strict
        else
            let ka = kindOf a
            let kb = kindOf b
            // P19 — string against non-string. Same-family string pairs never reach the table.
            if ka = DTypeKind.Str || kb = DTypeKind.Str then -1
            // P14 / P15 — temporal dtypes promote only with themselves.
            elif ka = DTypeKind.Instant || ka = DTypeKind.Duration
                 || kb = DTypeKind.Instant || kb = DTypeKind.Duration then -1
            // P12 / P13 — decimal
            elif ka = DTypeKind.Decimal || kb = DTypeKind.Decimal then
                let other = if ka = DTypeKind.Decimal then b else a
                let ko = kindOf other
                if isIntKind ko && bitsOf other <= 64 then int DTypeCode.Decimal else -1
            // P4 / P5 / P6 — integers
            elif isIntKind ka && isIntKind kb then
                if ka = kb then
                    // both signed or both unsigned: the wider one
                    if bitsOf a >= bitsOf b then int a else int b
                else
                    let s, u = if ka = DTypeKind.SignedInt then a, b else b, a
                    let bs = bitsOf s
                    let bu = bitsOf u
                    let mutable pick = -1
                    let mutable i = 0
                    while pick < 0 && i < signedLadder.Length do
                        let c = signedLadder.[i]
                        let bc = bitsOf c
                        if bc > bu && bc >= bs && p6Allows c bs then pick <- int c
                        i <- i + 1
                    if pick >= 0 then pick
                    elif strict then -1
                    else int DTypeCode.Float64
            // P9 / P10 / P11 — anything with a complex
            elif ka = DTypeKind.Complex || kb = DTypeKind.Complex then
                if ka = kb then
                    // both complex: the wider one
                    if DTypeTables.itemSize.[int a] >= DTypeTables.itemSize.[int b] then int a else int b
                else
                    let cplx, other = if ka = DTypeKind.Complex then a, b else b, a
                    let comp = componentOf cplx
                    let r = compute other comp strict
                    if r < 0 then -1
                    else int (liftToComplex (LanguagePrimitives.EnumOfValue<byte, DTypeCode>(byte r)))
            // P8 — both float
            elif ka = DTypeKind.Float && kb = DTypeKind.Float then
                if DTypeTables.itemSize.[int a] >= DTypeTables.itemSize.[int b] then int a else int b
            // P7 — integer with float
            elif ka = DTypeKind.Float || kb = DTypeKind.Float then
                let f, i = if ka = DTypeKind.Float then a, b else b, a
                let need = floatForIntBits (bitsOf i)
                // the answer must also cover the float operand itself
                if DTypeTables.itemSize.[int need] >= DTypeTables.itemSize.[int f] then int need else int f
            else -1

    let private buildTable (strict: bool) =
        let n = DTypeCode.Count
        let t = Array.zeroCreate<int> (n * n)
        for i in 0 .. n - 1 do
            for j in 0 .. n - 1 do
                let a = LanguagePrimitives.EnumOfValue<byte, DTypeCode>(byte i)
                let b = LanguagePrimitives.EnumOfValue<byte, DTypeCode>(byte j)
                t.[i * n + j] <- compute a b strict
        t

    let private looseTable = buildTable false
    let private strictTable = buildTable true

    let inline private lookup (a: DTypeCode) (b: DTypeCode) =
        let t = if Config.StrictIntegerPromotion then strictTable else looseTable
        t.[int a * DTypeCode.Count + int b]

    /// <summary>
    /// Attempts the promotion, returning <c>ValueNone</c> where the lattice has no answer.
    /// </summary>
    /// <param name="a">First dtype.</param>
    /// <param name="b">Second dtype.</param>
    /// <returns>The common dtype, or <c>ValueNone</c> when the pair cannot be promoted.</returns>
    let tryPromote (a: DType) (b: DType) : DType voption =
        if a.Code = b.Code && a.Width = b.Width then ValueSome a
        elif a.IsString || b.IsString then
            match a.Code, b.Code with
            | DTypeCode.Utf16, DTypeCode.Utf16 -> ValueSome(DType.Utf16(max a.Width b.Width))
            | DTypeCode.Ascii, DTypeCode.Ascii -> ValueSome(DType.Ascii(max a.Width b.Width))
            | DTypeCode.Ascii, DTypeCode.Utf16
            | DTypeCode.Utf16, DTypeCode.Ascii -> ValueSome(DType.Utf16(max a.Width b.Width))
            | _ -> ValueNone
        else
            let r = lookup a.Code b.Code
            if r < 0 then ValueNone
            else ValueSome(DType.OfCode(LanguagePrimitives.EnumOfValue<byte, DTypeCode>(byte r)))

    /// <summary>
    /// The common dtype of two dtypes.
    /// </summary>
    /// <param name="a">First dtype.</param>
    /// <param name="b">Second dtype.</param>
    /// <returns>The dtype both operands promote to.</returns>
    /// <exception cref="T:YANum.Core.YANumTypeError">
    /// The pair has no common dtype — a string against a non-string, a temporal dtype against
    /// anything else, <c>Decimal</c> against a float, complex or 128-bit integer, or, under
    /// <see cref="P:YANum.Core.Config.StrictIntegerPromotion"/>, a signed/unsigned pair with no
    /// exact signed result.
    /// </exception>
    let promote (a: DType) (b: DType) : DType =
        match tryPromote a b with
        | ValueSome t -> t
        | ValueNone ->
            if not a.IsString && not b.IsString
               && Config.StrictIntegerPromotion
               && looseTable.[int a.Code * DTypeCode.Count + int b.Code] >= 0 then
                Errors.typeError
                    (sprintf "cannot promote %s and %s exactly: no signed integer dtype holds every value of \
                              both, and Config.StrictIntegerPromotion forbids the float64 fallback"
                        a.Name b.Name)
            else
                Errors.typeError (sprintf "cannot promote %s and %s: there is no common dtype" a.Name b.Name)

    /// <summary>Whether two dtypes have a common dtype under the current configuration.</summary>
    /// <param name="a">First dtype.</param>
    /// <param name="b">Second dtype.</param>
    /// <returns><c>true</c> when <see cref="M:YANum.Core.Promotion.promote"/> would succeed.</returns>
    let canPromote (a: DType) (b: DType) : bool =
        match tryPromote a b with
        | ValueSome _ -> true
        | ValueNone -> false
