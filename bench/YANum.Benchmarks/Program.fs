module YANum.Benchmarks.Program

open BenchmarkDotNet.Running

[<EntryPoint>]
let main argv =
    BenchmarkSwitcher.FromAssembly(typeof<DescriptorBenchmarks>.Assembly).Run(argv) |> ignore
    0
