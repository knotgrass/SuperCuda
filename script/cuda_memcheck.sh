<<comment
 # cuda-memcheck –-tool <tool> <application>
• memcheck : memory errors (access violations, leaks, API errors)
• synccheck: misuse of synchronization (invalid masks, conditions)
• racecheck: data races (read-after-write, write-after-read hazards)
• initcheck: evaluation of uninitialized values (global memory only)
comment

cuda-memcheck --tool memcheck bin/08_Reductions.out
cuda-memcheck --tool synccheck bin/08_Reductions.out
cuda-memcheck --tool synccheck: bin/08_Reductions.out
cuda-memcheck --tool synccheck bin/08_Reductions.out
cuda-memcheck --tool synccheck: bin/08_Reductions.out
cuda-memcheck --tool synccheck bin/08_Reductions.ou
