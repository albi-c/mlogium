fn __new_gen(&next, &has) {
    struct Generator {
        let next: typeof(next);
        let has: typeof(has);

        static fn new(&next, &has) {
            Gen::new(next, has)
        }

        fn @iter() {
            (self.next, self.has)
        }

        fn map(&f) {
            self::new(||[&self, &f] f(self.next()), self.has)
        }

        fn unpack_map(&f) {
            self::new(||[&self, &f] f(self.next()...), self.has)
        }

        fn zip(&g) {
            self::new(||[&self, &g] (self.next(), g.next()),
                      ||[&self, &g] self.has() && g.has())
        }

        fn enumerate() {
            self::new(||[&self, i = -1] { i += 1; (i, self.next) }, self.has)
        }

        fn peek(&f) {
            self::new(||[&self, &f] { let v = self.next(); f(v); v }, self.has)
        }

        fn foreach(&f) {
            self.peek(f).consume()
        }

        fn reduce(&f, s) {
            for v in self {
                s = f(s, v);
            }
            s
        }

        fn sum() {
            self.reduce(|a, b| a + b, 0)
        }

        fn prod() {
            self.reduce(|a, b| a * b, 1)
        }

        fn consume() {
            for _ in self {}
        }

        fn count() {
            let n = 0;
            for _ in self {
                n += 1;
            }
            n
        }

        fn all() {
            for v in self {
                if !v {
                    return false;
                }
            }
            return true;
        }

        fn any() {
            for v in self {
                if v {
                    return true;
                }
            }
            return false;
        }

        fn limit(&n: num) {
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

namespace Gen {
    fn new(&next, &has) {
        __new_gen(next, has)
    }

    fn range(&n: num) {
        let i = 0;
        Gen::new(||[&i] {
            let x = i;
            i += 1;
            x
        }, ||[&i, &n] i < n)
    }

    fn generate(&f) {
        Gen::new(||[&f, i = -1] { i += 1; f(i) },
                 || true)
    }

    fn count() {
        Gen::new(||[i = -1] { i += 1; i },
                 || true)
    }

    fn buildings() {
        Gen::range(@links).map(getlink)
    }
}
