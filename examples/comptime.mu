// everything in this block is evaluated at compile time
comptime {
    // recursively generates a tuple of numbers ranging [i..n)
    fn _range(n: num, i: num) {
        if i >= n {
            ()
        } else {
            (i,) ++ _range(n, i + 1)
        }
    }

    fn range(n: num) {
        // number ranges can be unpacked into tuples
        // generates a tuple of numbers ranging [0..n)
        (0..n...,)
    }

    // calls the provided function on all elements of a tuple
    fn map(f: fn(?) -> ?, t) {
        if t.len == 0 {
            ()
        } else {
            // split a tuple into the first value and the rest of the values
            let (val, rest) = t.split;
            (f(val),) ++ map(f, rest)
        }
    }

    fn reduce(f: fn(?, ?) -> ?, s, t) {
        if t.len == 0 {
            s
        } else {
            let (val, rest) = t.split;
            reduce(f, f(s, val), rest)
        }
    }

    fn factorial(n: num) {
        reduce(
            |x, y| x * y,
            1,
            map(
                |x| x + 1,
                range(n)
            )
        )
    }

    fn with_n(x, n) {
        // when called with a value that is not known at compile time
        // x has the type "Opaque" - it can only be passed around
        (x, n + 2)
    }
}

// comptime functions can be called from a runtime context
// if the provided parameters are known at compile time
print(factorial(5));

// the following is only allowed because the value
// is not processed in any way inside the "pass_through" function
let x = rand(5);
// values that are known at compile time can be mixed in
print(with_n(x, 7).1);
