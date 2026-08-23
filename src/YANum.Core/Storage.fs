namespace YANum.Core

#nowarn "9"
#nowarn "51"

open System
open System.Runtime.CompilerServices
open System.Runtime.InteropServices
open System.Threading
open Microsoft.FSharp.NativeInterop

/// How a <see cref="T:YANum.Core.Storage"/> block was obtained, and therefore how it is released.
type internal StorageKind =
    /// A pinned-object-heap byte array. Nothing to free; the GC owns the lifetime.
    | PinnedArray = 0
    /// NativeMemory.AlignedAlloc. Must be freed exactly once.
    | NativeAligned = 1
    /// A caller's array pinned with a GCHandle for zero-copy wrapping. The handle must be freed.
    | PinnedHandle = 2

/// <summary>
/// A flat, byte-addressed block of memory with a stable base address.
/// </summary>
/// <remarks>
/// <para>
/// <b>Managed backend (default).</b> A pinned array from the pinned object heap. Phase 0
/// confirmed the address survives repeated forced compacting gen2 collections, so the pointer
/// is taken once at construction and cached. <c>Dispose</c> is a no-op; the GC owns the
/// lifetime. Capacity is capped at <see cref="F:System.Array.MaxLength"/>.
/// </para>
/// <para>
/// <b>Native backend.</b> 64-byte-aligned native memory. It is reference counted: every array
/// referencing the block holds one count, views <c>AddRef</c> on construction and
/// <c>Release</c> on disposal or finalization, and a finalizer on the block frees it as a
/// backstop.
/// </para>
/// <para>
/// <b>Disposing a native-backed array that still has live views is a use-after-free.</b> The
/// reference count exists to make that survivable, but an explicit <c>Dispose</c> on the last
/// owner while a view is alive will free memory the view still points at. Set
/// <see cref="P:YANum.Core.Config.CheckAliasing"/> to have the block assert its count is 1
/// before freeing.
/// </para>
/// </remarks>
[<Sealed>]
type Storage =

    val mutable private RefCount : int
    val mutable private Freed : int
    val private KindTag : StorageKind
    val private Handle : GCHandle
    val private Buffer : byte[]
    val private BasePtr : nativeint
    val private LengthBytes : unativeint
    val private Writable : bool

    private new (kind: StorageKind, handle: GCHandle, buffer: byte[], ptr: nativeint,
                 lengthBytes: unativeint, writable: bool) =
        { RefCount = 1
          Freed = 0
          KindTag = kind
          Handle = handle
          Buffer = buffer
          BasePtr = ptr
          LengthBytes = lengthBytes
          Writable = writable }

    /// <summary>The size of the block in <b>bytes</b>.</summary>
    member this.Length : unativeint = this.LengthBytes

    /// <summary>The stable base address of the block.</summary>
    member this.Pointer : nativeint = this.BasePtr

    /// <summary>Which allocator backs this block.</summary>
    member this.Backend : StorageBackend =
        match this.KindTag with
        | StorageKind.NativeAligned -> Native
        | _ -> Managed

    /// <summary>Whether elements may be written through this block.</summary>
    member this.IsWritable : bool = this.Writable

    /// <summary>Whether the block has already been released.</summary>
    member this.IsFreed : bool = Volatile.Read(&Unsafe.AsRef(&this).Freed) <> 0

    /// <summary>The current reference count. Exposed for diagnostics and tests.</summary>
    member this.ReferenceCount : int = Volatile.Read(&Unsafe.AsRef(&this).RefCount)

    /// <summary>Adds one to the reference count. Every view of this block must call it once.</summary>
    member this.AddRef () : unit =
        Interlocked.Increment(&this.RefCount) |> ignore

    /// <summary>
    /// Drops one reference, releasing the block when the last one goes.
    /// </summary>
    /// <remarks>Releasing a managed block is a no-op beyond the bookkeeping.</remarks>
    member this.Release () : unit =
        if Interlocked.Decrement(&this.RefCount) <= 0 then
            this.Free()

    member private this.Free () : unit =
        if Interlocked.Exchange(&this.Freed, 1) = 0 then
            match this.KindTag with
            | StorageKind.NativeAligned ->
                if this.BasePtr <> 0n then
                    NativeMemory.AlignedFree(this.BasePtr.ToPointer())
            | StorageKind.PinnedHandle ->
                if this.Handle.IsAllocated then this.Handle.Free()
            | _ -> ()

    /// <summary>
    /// Drops this owner's reference. For the managed backend this is a no-op beyond bookkeeping.
    /// </summary>
    member this.Dispose () : unit =
        if Config.CheckAliasing && this.KindTag = StorageKind.NativeAligned then
            let rc = this.ReferenceCount
            if rc > 1 then
                Errors.general
                    (sprintf "Storage.Dispose: native block at 0x%x still has %d live references; \
                              disposing it now would leave dangling views" (int64 this.BasePtr) rc)
        this.Release()
        GC.SuppressFinalize this

    override this.Finalize () = this.Free()

    interface IDisposable with
        member this.Dispose () = this.Dispose()

    // Phase 5: ArrayPool — pooled rent/return goes here, keyed on rounded byte size.

    /// <summary>A byte address inside the block.</summary>
    /// <param name="offsetBytes">Offset from the base address, in bytes.</param>
    /// <returns>The absolute address.</returns>
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member this.ByteAddress (offsetBytes: nativeint) : nativeint = this.BasePtr + offsetBytes

    /// <summary>A typed pointer into the block.</summary>
    /// <param name="offsetElems">Offset from the base address, in <c>'T</c>-sized elements.</param>
    /// <returns>A pointer to that element.</returns>
    /// <remarks>The offset is <c>nativeint</c> so that blocks above 2^31 elements address correctly.</remarks>
    [<MethodImpl(MethodImplOptions.AggressiveInlining)>]
    member this.PointerAs<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType>
        (offsetElems: nativeint) : nativeptr<'T> =
        NativePtr.ofNativeInt<'T> (this.BasePtr + offsetElems * nativeint (Unsafe.SizeOf<'T>()))

    /// <summary>A typed span over a segment of the block.</summary>
    /// <param name="offsetElems">Offset from the base address, in <c>'T</c>-sized elements.</param>
    /// <param name="countElems">Number of elements in the span.</param>
    /// <returns>A span over that segment.</returns>
    /// <remarks>
    /// <see cref="T:System.Span`1"/> carries an <c>int</c> length, so this is limited to
    /// segments below 2^31 elements. The general path is
    /// <see cref="M:YANum.Core.Storage.PointerAs``1(System.IntPtr)"/> with <c>nativeint</c> counters.
    /// </remarks>
    /// <exception cref="T:YANum.Core.YANumValueError">The segment is negative or does not fit an <c>int</c> length.</exception>
    member this.AsSpan<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType>
        (offsetElems: nativeint, countElems: nativeint) : Span<'T> =
        // The throw helper is instantiated at `unit`, not at Span<'T>: CIL forbids a generic
        // instantiation at a byref-like type.
        if countElems < 0n || countElems > nativeint Int32.MaxValue then
            (Errors.valueError
                (sprintf "Storage.AsSpan: %d elements cannot be expressed as a Span length; use PointerAs instead"
                    (int64 countElems)) : unit)
        Span<'T>(this.PointerAs<'T>(offsetElems) |> NativePtr.toVoidPtr, int countElems)

    /// <summary>A byte span over the whole block.</summary>
    /// <returns>A span covering every byte.</returns>
    /// <exception cref="T:YANum.Core.YANumValueError">The block is larger than <c>Int32.MaxValue</c> bytes.</exception>
    member this.AsByteSpan () : Span<byte> =
        this.AsSpan<byte>(0n, nativeint this.LengthBytes)

    /// <summary>Allocates a zero-filled block.</summary>
    /// <param name="lengthBytes">Size in bytes.</param>
    /// <param name="backend">Which allocator to use.</param>
    /// <returns>The new block, with a reference count of 1.</returns>
    /// <exception cref="T:YANum.Core.YANumValueError">The request exceeds the backend's cap.</exception>
    static member AllocateZeroed (lengthBytes: unativeint, backend: StorageBackend) : Storage =
        Storage.Allocate(lengthBytes, backend, true)

    /// <summary>Allocates a block whose contents are undefined.</summary>
    /// <param name="lengthBytes">Size in bytes.</param>
    /// <param name="backend">Which allocator to use.</param>
    /// <returns>The new block, with a reference count of 1.</returns>
    /// <remarks>Only use this for a buffer that is about to be fully overwritten.</remarks>
    /// <exception cref="T:YANum.Core.YANumValueError">The request exceeds the backend's cap.</exception>
    static member AllocateUninitialized (lengthBytes: unativeint, backend: StorageBackend) : Storage =
        Storage.Allocate(lengthBytes, backend, false)

    static member private Allocate (lengthBytes: unativeint, backend: StorageBackend, zeroed: bool) : Storage =
        match backend with
        | Managed ->
            if lengthBytes > unativeint Array.MaxLength then
                Errors.valueError
                    (sprintf "Storage: %d bytes exceeds the managed backend cap of %d; use the Native backend"
                        (uint64 lengthBytes) Array.MaxLength)
            else
                let n = int lengthBytes
                let buf =
                    if zeroed then GC.AllocateArray<byte>(n, pinned = true)
                    else GC.AllocateUninitializedArray<byte>(n, pinned = true)
                let ptr =
                    NativePtr.toNativeInt (NativePtr.ofVoidPtr<byte> (Unsafe.AsPointer(&MemoryMarshal.GetArrayDataReference buf)))
                new Storage(StorageKind.PinnedArray, Unchecked.defaultof<GCHandle>, buf, ptr, lengthBytes, true)
        | Native ->
            if lengthBytes = 0un then
                // AlignedAlloc(0) is legal but returns a block nothing may be read from; a
                // managed empty block is simpler and needs no free.
                Storage.Allocate(0un, Managed, zeroed)
            else
                let p = NativeMemory.AlignedAlloc(lengthBytes, 64un)
                if zeroed then NativeMemory.Clear(p, lengthBytes)
                new Storage(StorageKind.NativeAligned, Unchecked.defaultof<GCHandle>, null,
                            NativePtr.toNativeInt (NativePtr.ofVoidPtr<byte> p), lengthBytes, true)

    /// <summary>
    /// Pins a caller's array and wraps it without copying.
    /// </summary>
    /// <param name="array">The array to wrap. It stays pinned until the block is released.</param>
    /// <returns>A block aliasing the array's elements.</returns>
    /// <remarks>
    /// The array is pinned with a <see cref="T:System.Runtime.InteropServices.GCHandle"/> for
    /// the lifetime of the block, which the GC cannot move or compact around. Prefer copying
    /// for long-lived arrays.
    /// </remarks>
    static member PinArray<'T when 'T : unmanaged and 'T : struct and 'T : (new : unit -> 'T) and 'T :> System.ValueType>
        (array: 'T[]) : Storage =
        if isNull (box array) then Errors.valueError "Storage.PinArray: array is null"
        elif array.Length = 0 then Storage.Allocate(0un, Managed, true)
        else
            let handle = GCHandle.Alloc(array, GCHandleType.Pinned)
            let ptr = handle.AddrOfPinnedObject()
            let bytes = unativeint array.Length * unativeint (Unsafe.SizeOf<'T>())
            new Storage(StorageKind.PinnedHandle, handle, null, ptr, bytes, true)
