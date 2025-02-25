fn __new_stream(&next, &has) {
    struct Stream_ {
        let next: typeof(next);
        let has: typeof(has);

        static fn new(&next, &has) {
            Stream::new(next, has)
        }

        fn [nodiscard] @iter() {
            (self.next, self.has)
        }

        fn [nodiscard] map(&f) {
            self::new(||[&self, &f] f(self.next()), self.has)
        }

        fn [nodiscard] unpack_map(&f) {
            self::new(||[&self, &f] f(self.next()...), self.has)
        }

        fn [nodiscard] zip(&g) {
            self::new(||[&self, &g] (self.next(), g.next()),
                      ||[&self, &g] self.has() && g.has())
        }

        fn [nodiscard] enumerate() {
            self::new(||[&self, i = -1] { i += 1; (i, self.next) }, self.has)
        }

        fn [nodiscard] peek(&f) {
            self::new(||[&self, &f] { let v = self.next(); f(v); v }, self.has)
        }

        fn [nodiscard] foreach(&f) {
            self.peek(f).consume()
        }

        fn [nodiscard] reduce(&f, s) {
            for v in self {
                s = f(s, v);
            }
            s
        }

        fn [nodiscard] sum() {
            self.reduce(|a, b| a + b, 0)
        }

        fn [nodiscard] prod() {
            self.reduce(|a, b| a * b, 1)
        }

        fn consume() {
            for _ in self {}
        }

        fn [nodiscard] count() {
            let n = 0;
            for _ in self {
                n += 1;
            }
            n
        }

        fn [nodiscard] all() {
            for v in self {
                if !v {
                    return false;
                }
            }
            return true;
        }

        fn [nodiscard] any() {
            for v in self {
                if v {
                    return true;
                }
            }
            return false;
        }

        fn [nodiscard] limit(&n: num) {
            let i = 0;
            self::new(||[&self, &i] { i += 1; self.next() },
                      ||[&self, &i, &n] { i < n && self.has() })
        }

        fn skip() {
            if self.has() {
                self.next();
            }
            self
        }

        fn skip_n(&n: num) {
            for _ in 0..n {
                if !self.has() {
                    break;
                }
                self.next();
            }
            self
        }
    }(next, has)
}

namespace Stream {
    fn new(&next, &has) {
        __new_stream(next, has)
    }

    fn @call(&iterable) {
        if const typeof(iterable).equals(Range) {
            Stream::range(iterable.start, iterable.end)
        } else if const typeof(iterable).equals(RangeWithStep) {
            Stream::range(iterable.start, iterable.end, iterable.step)
        } else {
            Stream::new(iterable.@iter()...)
        }
    }

    fn @from(&iterable) {
        Stream(iterable)
    }

    fn range(n...: num) {
        if const #has_attr(n, "3") {
            #static_assert(false, "Stream::range expects 1 to 3 parameters, got >3")
        } else if const #has_attr(n, "2") {
            Stream::_range_create(n.0, n.1, n.2)
        } else if const #has_attr(n, "1") {
            Stream::_range_create(n.0, n.1, 1)
        } else if const #has_attr(n, "0") {
            Stream::_range_create(0, n.0, 1)
        } else {
            #static_assert(false, "Stream::range expects 1 to 3 parameters, got 0")
        }
    }

    fn _range_create(&start: num, &end: num, &step: num) {
        let i = start;
        Stream::new(||[&i, &step] {
            let x = i;
            i += step;
            x
        }, ||[&i, &end] i < end)
    }

    fn generate(&f) {
        Stream::new(||[&f, i = -1] { i += 1; f(i) },
                    || true)
    }

    fn count() {
        Stream::new(||[i = -1] { i += 1; i },
                    || true)
    }

    fn buildings() {
        Stream::range(@links).map(getlink)
    }
}
