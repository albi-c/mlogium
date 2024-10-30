comptime {
    fn _range(n: num, i: num) {
        if i >= n {
            ()
        } else {
            (i,) ++ _range(n, i + 1)
        }
    }

    fn range(n: num) {
        _range(n, 0)
    }

    fn map(f: fn(?) -> ?, t) {
        if t.len == 0 {
            ()
        } else {
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
}

print(factorial(5));
