namespace YANum.Core

open System
open System.Runtime.CompilerServices

/// <summary>
/// Base of every exception YANum raises. Catch this to catch anything the library
/// throws deliberately.
/// </summary>
/// <remarks>
/// These are ordinary .NET exception classes rather than F# <c>exception</c>
/// declarations, because F# exception declarations cannot nominate a base class and
/// the specification requires a common base.
/// </remarks>
type YANumException =
    inherit Exception
    /// <summary>Creates the exception with a message.</summary>
    /// <param name="message">Human-readable description naming the operation involved.</param>
    new (message: string) = { inherit Exception(message) }
    /// <summary>Creates the exception with a message and an inner exception.</summary>
    /// <param name="message">Human-readable description naming the operation involved.</param>
    /// <param name="inner">The underlying cause.</param>
    new (message: string, inner: exn) = { inherit Exception(message, inner) }

/// <summary>An illegal dtype or dtype combination for the requested operation.</summary>
type YANumTypeError =
    inherit YANumException
    /// <summary>Creates the exception with a message.</summary>
    /// <param name="message">Description naming the operation and the dtypes involved.</param>
    new (message: string) = { inherit YANumException(message) }

/// <summary>A broadcast, reshape, rank or axis failure.</summary>
type YANumShapeError =
    inherit YANumException
    /// <summary>Creates the exception with a message.</summary>
    /// <param name="message">Description naming the operation and the shapes involved.</param>
    new (message: string) = { inherit YANumException(message) }

/// <summary>A value does not fit its destination (scalar out of range, string too long).</summary>
type YANumOverflowError =
    inherit YANumException
    /// <summary>Creates the exception with a message.</summary>
    /// <param name="message">Description naming the value and the destination.</param>
    new (message: string) = { inherit YANumException(message) }

/// <summary>A bad argument value that is neither a type nor a shape problem.</summary>
type YANumValueError =
    inherit YANumException
    /// <summary>Creates the exception with a message.</summary>
    /// <param name="message">Description naming the argument and the constraint it broke.</param>
    new (message: string) = { inherit YANumException(message) }

/// <summary>
/// Throw helpers. Every helper is <c>NoInlining</c> so that a call site keeps only a
/// call instruction on the happy path and the exception construction stays out of line.
/// </summary>
/// <remarks>
/// Error messages must name the operation and the dtypes or shapes involved, e.g.
/// <c>"shapes (3,4) and (5,4) are not broadcast-compatible on axis 0"</c>.
/// </remarks>
[<RequireQualifiedAccess>]
module Errors =

    /// <summary>Raises a general <see cref="T:YANum.Core.YANumException"/>.</summary>
    /// <param name="msg">Message naming the operation.</param>
    /// <returns>Never returns.</returns>
    [<MethodImpl(MethodImplOptions.NoInlining)>]
    let general (msg: string) : 'a = raise (YANumException msg)

    /// <summary>Raises a <see cref="T:YANum.Core.YANumTypeError"/>.</summary>
    /// <param name="msg">Message naming the operation and the dtypes involved.</param>
    /// <returns>Never returns.</returns>
    [<MethodImpl(MethodImplOptions.NoInlining)>]
    let typeError (msg: string) : 'a = raise (YANumTypeError msg)

    /// <summary>Raises a <see cref="T:YANum.Core.YANumShapeError"/>.</summary>
    /// <param name="msg">Message naming the operation and the shapes involved.</param>
    /// <returns>Never returns.</returns>
    [<MethodImpl(MethodImplOptions.NoInlining)>]
    let shapeError (msg: string) : 'a = raise (YANumShapeError msg)

    /// <summary>Raises a <see cref="T:YANum.Core.YANumOverflowError"/>.</summary>
    /// <param name="msg">Message naming the value and its destination.</param>
    /// <returns>Never returns.</returns>
    [<MethodImpl(MethodImplOptions.NoInlining)>]
    let overflowError (msg: string) : 'a = raise (YANumOverflowError msg)

    /// <summary>Raises a <see cref="T:YANum.Core.YANumValueError"/>.</summary>
    /// <param name="msg">Message naming the argument and the constraint it broke.</param>
    /// <returns>Never returns.</returns>
    [<MethodImpl(MethodImplOptions.NoInlining)>]
    let valueError (msg: string) : 'a = raise (YANumValueError msg)

    /// <summary>Raises an index-out-of-range failure describing the offending axis.</summary>
    /// <param name="op">Operation name.</param>
    /// <param name="axis">Axis whose index was out of range.</param>
    /// <param name="index">The offending index, after negative-index normalisation.</param>
    /// <param name="extent">The extent of that axis.</param>
    /// <returns>Never returns.</returns>
    [<MethodImpl(MethodImplOptions.NoInlining)>]
    let indexOutOfRange (op: string) (axis: int) (index: int) (extent: int) : 'a =
        raise (YANumValueError(
            sprintf "%s: index %d is out of bounds for axis %d with extent %d" op index axis extent))

    /// <summary>Formats a shape as NumPy does, e.g. <c>(3,4)</c> or <c>()</c> for rank 0.</summary>
    /// <param name="shape">The extents.</param>
    /// <returns>A parenthesised, comma-separated rendering.</returns>
    let formatShape (shape: int[]) : string =
        if isNull (box shape) then "(null)"
        elif shape.Length = 0 then "()"
        else "(" + String.Join(",", shape) + ")"
