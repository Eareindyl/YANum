```

BenchmarkDotNet v0.15.8, macOS Sequoia 15.7.9 (24G830) [Darwin 24.6.0]
Intel Xeon W-2140B CPU 3.20GHz, 1 CPU, 16 logical and 8 physical cores
.NET SDK 10.0.202
  [Host]   : .NET 10.0.6 (10.0.6, 10.0.626.17701), X64 RyuJIT x86-64-v3 DEBUG
  ShortRun : .NET 10.0.6 (10.0.6, 10.0.626.17701), X64 RyuJIT x86-64-v3

Job=ShortRun  IterationCount=3  LaunchCount=1  
WarmupCount=3  

```
| Method              | Mean      | Error      | StdDev    | Gen0   | Allocated |
|-------------------- |----------:|-----------:|----------:|-------:|----------:|
| DescriptorInit      |  6.713 ns |  0.7648 ns | 0.0419 ns |      - |         - |
| DescriptorTranspose |  6.587 ns |  1.0607 ns | 0.0581 ns |      - |         - |
| DescriptorReshape   |  7.622 ns |  2.2231 ns | 0.1219 ns |      - |         - |
| DescriptorPermute   |  7.156 ns |  0.7666 ns | 0.0420 ns |      - |         - |
| ReadThroughInRef    |  4.835 ns |  0.9089 ns | 0.0498 ns |      - |         - |
| ArrayTranspose      | 51.050 ns | 65.5300 ns | 3.5919 ns | 0.0459 |     264 B |
| ArrayReshape        | 63.114 ns | 79.9789 ns | 4.3839 ns | 0.0459 |     264 B |
