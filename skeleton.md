Caesar stores:
* Context/: dict from int -> str 
* Generation (model_output, kernel): dict from int -> str
* Feedback: dict from int -> (____)
* curr_state
* curr_iteration

### Actual States
* start
* generate kernel / code (query llm)
* compile: check can compile
* check corre
* perf: check profiler performance
* finish

# state configs

class CompileConfig:
    # define 

# table / dict / function
class TransitionConfig:
    start : ->
    generate : -> 
    compile : -> (T/F)
    correct : -> (T/F)
    profile: 

   start --> generate
   generate --> start
   # succeed
   compile[1] --> correct
   compile[2] ->


    

#### Transiitions:
* Start --> Gen (Prompt)


def 



Skeleton State Machine (launched as a process)


Input: 
- Problem 

Prompt(Problem, __) -> Inference -> Parse Result to extract Code -> Build the Cache -> Eval (GPU)
-> depends on Eval -> Prompt(Porblem, Eval)

Global Struct:
* Round #
* Current overall prompt (as formatted string). 
    - On init, this starts as initial problem statement.


Main Loop:
1. Read the current state, figure out what to do:
    * if Round # > max, end.
2. A bunch of state cases
    * Inference
    * 
3. Finish case, should return a new state.
3. 

Storage: 
- generations 
- context / feedback
store in a local file system
"""

# Start State --> enter the iteration, meta info,  problem, cache [Construct Initial Prompt]

# Model generation / Inference  () ->  Functional 
# input: new prompt (w feedback w env variable)
# side effect : update what are context at round i, what we generated
# output: kernel code (GPU code)

# Build / Eval phase 
# input: kernel code
# skip -> try again 
# logic: build the cache on CPU, when you get GPU, you eval the code
#    1. fail cpu build (nvcc compile ) -> EvalResult(compile=F, ) 
#     - update the context, feedback
#    2. success cpu build ->  
#       - 2a go on GPU EvalResult(compile=T, correct=F, )
#        - 2b  go on GPU EvalResult(compile=T, correct=T, performance) -> performance feedback.

#     - update the context, feedback   

# output: 

# Finish State --> exit the iteration