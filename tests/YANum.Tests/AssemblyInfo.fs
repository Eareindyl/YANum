namespace YANum.Tests

open Xunit

// YANum's configuration is process-global by design (specification section 3), so any test
// that uses Config.push mutates state every other test can observe. Running the suite
// serially is the only correct choice until configuration becomes scoped.
[<assembly: CollectionBehavior(DisableTestParallelization = true)>]
do ()
