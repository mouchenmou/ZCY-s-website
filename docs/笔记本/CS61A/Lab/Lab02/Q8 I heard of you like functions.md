Define a function `cycle` that takes in three functions `f1`, `f2`, and `f3`, as arguments. `cycle` will return another function `g` that should take in an integer argument `n` and return another function `h`. That final function `h` should take in an argument `x` and cycle through applying `f1`, `f2`, and `f3` to `x`, depending on what `n` was. Here's what the final function `h` should do to `x` for a few values of `n`:

- `n = 0`, return `x`
- `n = 1`, apply `f1` to `x`, or return `f1(x)`
- `n = 2`, apply `f1` to `x` and then `f2` to the result of that, or return `f2(f1(x))`
- `n = 3`, apply `f1` to `x`, `f2` to the result of applying `f1`, and then `f3` to the result of applying `f2`, or `f3(f2(f1(x)))`
- `n = 4`, start the cycle again applying `f1`, then `f2`, then `f3`, then `f1` again, or `f1(f3(f2(f1(x))))`
- And so forth.

_Hint_: most of the work goes inside the most nested function.

```Python
    def cycle(f1, f2, f3): 
    """Returns a function that is itself a higher-order function. 
    
    >>> def add1(x):
            return x + 1 
    >>> def times2(x): 
            return x * 2 
    >>> def add3(x):
            return x + 3 
    >>> my_cycle = cycle(add1, times2, add3) 
    >>> identity = my_cycle(0) 
    >>> identity(5) 
    5 
    >>> add_one_then_double = my_cycle(2) 
    >>> add_one_then_double(1) 
    4 
    >>> do_all_functions = my_cycle(3) 
    >>> do_all_functions(2) 
    9 
    >>> do_more_than_a_cycle = my_cycle(4) 
    >>> do_more_than_a_cycle(2) 
    10 
    >>> do_two_cycles = my_cycle(6) 
    >>> do_two_cycles(1)
    19 
    """
     *** YOUR CODE HERE ***
     
     
    def g(n):
        def h(x):
            i=1
            total=x
            while(i<=n):
                if(i%3==0):
                    total = f3(total)
                elif(i%3==1):
                    total = f1(total)
                elif(i%3==2):
                    total = f2(total)
                i=i+1
            return total
        return h
    return g
```