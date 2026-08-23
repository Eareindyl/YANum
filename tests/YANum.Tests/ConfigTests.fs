namespace YANum.Tests

open System
open Xunit
open YANum.Core

module ConfigTests =

    [<Fact>]
    let ``defaults match the specification`` () =
        Config.reset ()
        Assert.Equal(C, Config.DefaultOrder)
        Assert.Equal(DType.Float64, Config.DefaultFloat)
        Assert.Equal(DType.Int64, Config.DefaultInt)
        Assert.Equal(Managed, Config.DefaultStorage)
        Assert.Equal(Fast, Config.ErrorMode)
        Assert.Equal(DebugOnly, Config.BoundsChecking)
        Assert.False Config.CheckAliasing
        Assert.False Config.StrictIntegerPromotion
        Assert.Equal(Throw, Config.StringOverflow)
        Assert.True Config.EnableSimd
        Assert.Equal(65536, Config.ParallelThreshold)
        Assert.Equal(Environment.ProcessorCount, Config.MaxDegreeOfParallelism)
        Assert.False Config.DeterministicReductions
        Assert.Equal(1000, Config.PrintThreshold)
        Assert.Equal(3, Config.PrintEdgeItems)
        Assert.Equal(8, Config.PrintPrecision)

    [<Fact>]
    let ``push applies and dispose restores`` () =
        Config.reset ()
        Assert.Equal(Fast, Config.ErrorMode)
        (use _ = Config.push (fun c -> c.ErrorMode <- Strict)
         Assert.Equal(Strict, Config.ErrorMode))
        Assert.Equal(Fast, Config.ErrorMode)

    [<Fact>]
    let ``nested pushes restore in order`` () =
        Config.reset ()
        (use _ = Config.push (fun c -> c.PrintPrecision <- 3)
         Assert.Equal(3, Config.PrintPrecision)
         (use _ = Config.push (fun c -> c.PrintPrecision <- 5)
          Assert.Equal(5, Config.PrintPrecision))
         Assert.Equal(3, Config.PrintPrecision))
        Assert.Equal(8, Config.PrintPrecision)

    [<Fact>]
    let ``push does not mutate the snapshot it replaced`` () =
        Config.reset ()
        let before = Config.Current
        (use _ = Config.push (fun c -> c.PrintPrecision <- 17)
         Assert.NotSame(before, Config.Current))
        Assert.Equal(8, before.PrintPrecision)
