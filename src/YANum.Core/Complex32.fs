namespace YANum.Core

open System
open System.Numerics
open System.Runtime.CompilerServices
open System.Runtime.InteropServices

/// <summary>
/// A single-precision complex number: two <see cref="T:System.Single"/> components,
/// exactly 8 bytes, sequentially laid out with no padding.
/// </summary>
/// <remarks>
/// <para>
/// .NET ships only the double-precision <see cref="T:System.Numerics.Complex"/>, so YANum
/// supplies this type to back the <c>Complex64</c> dtype. The byte image is the
/// little-endian concatenation of <see cref="P:YANum.Core.Complex32.Real"/> then
/// <see cref="P:YANum.Core.Complex32.Imaginary"/>, which is what NumPy's
/// <c>complex64</c> stores.
/// </para>
/// <para>
/// Equality is IEEE on both components: a value with a <c>NaN</c> component is not equal
/// to itself, matching <see cref="T:System.Numerics.Complex"/>. There is no ordering.
/// </para>
/// <para>
/// The static members <c>Sqrt</c>, <c>Exp</c>, <c>Log</c> and <c>Pow</c> follow the
/// FSharp.Core naming convention, so F#'s built-in <c>sqrt</c>, <c>exp</c>, <c>log</c> and
/// <c>( ** )</c> work directly on this type. <c>Abs</c> returns a
/// <see cref="T:System.Single"/> magnitude (as <c>Complex.Abs</c> does), so F#'s <c>abs</c>,
/// whose signature is <c>^T -&gt; ^T</c>, deliberately does not resolve against it.
/// </para>
/// </remarks>
[<Struct; CustomEquality; NoComparison; StructLayout(LayoutKind.Sequential)>]
type Complex32 =

    /// <summary>The real component.</summary>
    val Real : float32

    /// <summary>The imaginary component.</summary>
    val Imaginary : float32

    /// <summary>Creates a complex number from its two components.</summary>
    /// <param name="real">Real component.</param>
    /// <param name="imaginary">Imaginary component.</param>
    new (real: float32, imaginary: float32) = { Real = real; Imaginary = imaginary }

    /// <summary>The additive identity, <c>(0, 0)</c>.</summary>
    static member Zero = Complex32(0.0f, 0.0f)

    /// <summary>The multiplicative identity, <c>(1, 0)</c>.</summary>
    static member One = Complex32(1.0f, 0.0f)

    /// <summary>The imaginary unit, <c>(0, 1)</c>.</summary>
    static member ImaginaryOne = Complex32(0.0f, 1.0f)

    /// <summary>A value with both components <c>NaN</c>.</summary>
    static member NaN = Complex32(Single.NaN, Single.NaN)

    /// <summary>Widens this value to the double-precision <see cref="T:System.Numerics.Complex"/>.</summary>
    /// <returns>The equivalent double-precision value.</returns>
    member this.ToComplex() : Complex = Complex(float this.Real, float this.Imaginary)

    /// <summary>The modulus, computed in double precision to avoid intermediate overflow.</summary>
    member this.Magnitude : float32 =
        let r = float this.Real
        let i = float this.Imaginary
        float32 (Math.Sqrt(r * r + i * i))

    /// <summary>The argument in radians, in the range (-pi, pi].</summary>
    member this.Phase : float32 = float32 (Math.Atan2(float this.Imaginary, float this.Real))

    /// <summary>Narrows a double-precision complex number to single precision.</summary>
    /// <param name="c">The value to narrow.</param>
    /// <returns>The single-precision value.</returns>
    static member OfComplex (c: Complex) : Complex32 = Complex32(float32 c.Real, float32 c.Imaginary)

    /// <summary>Builds a value from polar coordinates.</summary>
    /// <param name="magnitude">The modulus.</param>
    /// <param name="phase">The argument in radians.</param>
    /// <returns>The corresponding rectangular value.</returns>
    static member FromPolarCoordinates (magnitude: float32, phase: float32) : Complex32 =
        Complex32.OfComplex(Complex.FromPolarCoordinates(float magnitude, float phase))

    /// <summary>The complex conjugate.</summary>
    /// <param name="c">The value.</param>
    /// <returns>A value with the imaginary component negated.</returns>
    static member Conjugate (c: Complex32) : Complex32 = Complex32(c.Real, -c.Imaginary)

    /// <summary>The modulus. Named to match <see cref="M:System.Numerics.Complex.Abs(System.Numerics.Complex)"/>.</summary>
    /// <param name="c">The value.</param>
    /// <returns>The modulus as a single-precision real.</returns>
    static member Abs (c: Complex32) : float32 = c.Magnitude

    /// <summary>The principal square root. Backs F#'s <c>sqrt</c>.</summary>
    /// <param name="c">The value.</param>
    /// <returns>The principal square root.</returns>
    static member Sqrt (c: Complex32) : Complex32 = Complex32.OfComplex(Complex.Sqrt(c.ToComplex()))

    /// <summary>The exponential. Backs F#'s <c>exp</c>.</summary>
    /// <param name="c">The value.</param>
    /// <returns><c>e</c> raised to the value.</returns>
    static member Exp (c: Complex32) : Complex32 = Complex32.OfComplex(Complex.Exp(c.ToComplex()))

    /// <summary>The principal natural logarithm. Backs F#'s <c>log</c>.</summary>
    /// <param name="c">The value.</param>
    /// <returns>The principal natural logarithm.</returns>
    static member Log (c: Complex32) : Complex32 = Complex32.OfComplex(Complex.Log(c.ToComplex()))

    /// <summary>The principal base-10 logarithm. Backs F#'s <c>log10</c>.</summary>
    /// <param name="c">The value.</param>
    /// <returns>The principal base-10 logarithm.</returns>
    static member Log10 (c: Complex32) : Complex32 = Complex32.OfComplex(Complex.Log10(c.ToComplex()))

    /// <summary>Raises a complex number to a complex power. Backs F#'s <c>( ** )</c>.</summary>
    /// <param name="a">The base.</param>
    /// <param name="b">The exponent.</param>
    /// <returns>The principal value of <c>a</c> raised to <c>b</c>.</returns>
    static member Pow (a: Complex32, b: Complex32) : Complex32 =
        Complex32.OfComplex(Complex.Pow(a.ToComplex(), b.ToComplex()))

    /// <summary>Adds two complex numbers.</summary>
    /// <param name="a">Left operand.</param>
    /// <param name="b">Right operand.</param>
    /// <returns>The sum.</returns>
    static member (+) (a: Complex32, b: Complex32) =
        Complex32(a.Real + b.Real, a.Imaginary + b.Imaginary)

    /// <summary>Subtracts two complex numbers.</summary>
    /// <param name="a">Left operand.</param>
    /// <param name="b">Right operand.</param>
    /// <returns>The difference.</returns>
    static member (-) (a: Complex32, b: Complex32) =
        Complex32(a.Real - b.Real, a.Imaginary - b.Imaginary)

    /// <summary>Multiplies two complex numbers.</summary>
    /// <param name="a">Left operand.</param>
    /// <param name="b">Right operand.</param>
    /// <returns>The product.</returns>
    static member (*) (a: Complex32, b: Complex32) =
        Complex32(a.Real * b.Real - a.Imaginary * b.Imaginary,
                  a.Real * b.Imaginary + a.Imaginary * b.Real)

    /// <summary>Divides two complex numbers, using the double-precision Smith algorithm.</summary>
    /// <param name="a">Numerator.</param>
    /// <param name="b">Denominator.</param>
    /// <returns>The quotient.</returns>
    static member (/) (a: Complex32, b: Complex32) =
        Complex32.OfComplex(Complex.Divide(a.ToComplex(), b.ToComplex()))

    /// <summary>Negates a complex number.</summary>
    /// <param name="a">The value.</param>
    /// <returns>The additive inverse.</returns>
    static member (~-) (a: Complex32) = Complex32(-a.Real, -a.Imaginary)

    /// <summary>Unary plus; returns the value unchanged.</summary>
    /// <param name="a">The value.</param>
    /// <returns>The value.</returns>
    static member (~+) (a: Complex32) = a

    /// <summary>Widening conversion to <see cref="T:System.Numerics.Complex"/>.</summary>
    /// <param name="c">The value to widen.</param>
    /// <returns>The double-precision value.</returns>
    static member op_Implicit (c: Complex32) : Complex = c.ToComplex()

    /// <summary>Narrowing conversion from <see cref="T:System.Numerics.Complex"/>.</summary>
    /// <param name="c">The value to narrow.</param>
    /// <returns>The single-precision value.</returns>
    static member op_Explicit (c: Complex) : Complex32 = Complex32.OfComplex c

    /// <summary>IEEE component-wise equality; <c>NaN</c> components compare unequal.</summary>
    /// <param name="other">The value to compare against.</param>
    /// <returns><c>true</c> when both components compare equal under IEEE rules.</returns>
    member this.Equals (other: Complex32) : bool =
        this.Real = other.Real && this.Imaginary = other.Imaginary

    /// <summary>Renders as <c>(real, imaginary)</c>, matching <see cref="T:System.Numerics.Complex"/>.</summary>
    /// <returns>The rendered value.</returns>
    override this.ToString() : string =
        "(" + this.Real.ToString(Globalization.CultureInfo.InvariantCulture)
            + ", " + this.Imaginary.ToString(Globalization.CultureInfo.InvariantCulture) + ")"

    /// <summary>IEEE component-wise equality against a boxed value.</summary>
    /// <param name="o">The boxed value to compare against.</param>
    /// <returns><c>true</c> when <paramref name="o"/> is a <see cref="T:YANum.Core.Complex32"/> that compares equal.</returns>
    override this.Equals (o: obj) : bool =
        match o with
        | :? Complex32 as c -> this.Equals c
        | _ -> false

    /// <summary>A hash consistent with equality except at <c>NaN</c>, as for any IEEE type.</summary>
    /// <returns>The hash code.</returns>
    override this.GetHashCode() : int =
        HashCode.Combine(this.Real, this.Imaginary)

    interface IEquatable<Complex32> with
        member this.Equals (other: Complex32) = this.Equals other

/// <summary>Layout assertions for <see cref="T:YANum.Core.Complex32"/>.</summary>
/// <remarks>
/// The authoritative fail-fast check for the whole library lives in <c>ArrayDesc.fs</c>;
/// this one exists so that a build of <c>Complex32.fs</c> alone still self-checks.
/// </remarks>
module Complex32Layout =

    /// <summary>The required size of <see cref="T:YANum.Core.Complex32"/> in bytes.</summary>
    [<Literal>]
    let ExpectedSize = 8

    /// <summary>Throws if the runtime layout of <see cref="T:YANum.Core.Complex32"/> is not 8 bytes.</summary>
    let verify () =
        let actual = Unsafe.SizeOf<Complex32>()
        if actual <> ExpectedSize then
            failwithf
                "YANum layout assertion failed: Unsafe.SizeOf<Complex32>() = %d, expected %d. \
                 Complex32 must be two float32 with no padding; check [<Struct>] and StructLayout(Sequential)."
                actual ExpectedSize

    do verify ()
