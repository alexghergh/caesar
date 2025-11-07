from enum import Enum


# this is supposed to be used as a set{}, e.g.
# strategy = { SHOW_INLINE_SYNTAX, PROFILER_FEEDBACK }
# if SHOW_INLINE_SYNTAX in strategy:
#   # do something
class Strategy(Enum):
    # whether to include CUDA syntax example in the prompt
    SHOW_INLINE_SYNTAX = "show_inline_cuda_syntax"

    # whether to enable compiler feedback; this also needs the compiler
    # transition in the state machine
    COMPILER_FEEDBACK = "use_compiler_feedback"

    # whether to enable correctness feedback (i.e. running inputs through the
    # model and checking for shape + value parity with the reference code); this
    # also needs the correctness transition in the state machine
    CORRECTNESS_FEEDBACK = "use_correctness_feedback"

    # whether to enable profiler feedback (i.e. torch profiler + nsys); this
    # also needs the profiler feedback transition in the state machine
    PROFILER_FEEDBACK = "use_profiler_feedback"


    def __str__(self) -> str:
        return self.name
