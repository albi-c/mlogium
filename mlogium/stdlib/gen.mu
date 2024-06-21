struct Gen of ? {
    static fn new(&next, &has) {
        Gen((next, has))
    }

    fn next() {
        self.0()
    }

    fn has() {
        self.1()
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

    static fn range(&n: num) {
        let i = 0;
        Gen::new(||[&i] {
            let x = i;
            i += 1;
            x
        }, ||[&i, &n] i < n)
    }

    static fn generate(&f) {
        Gen::new(||[&f, i = -1] { i += 1; f(i) },
                 || true)
    }

    static fn count() {
        Gen::new(||[i = -1] { i += 1; i },
                 || true)
    }

    static fn buildings() {
        let i = 0;
        Gen::new(||[&i] {
            let b = getlink(i);
            i += 1;
            b
        }, ||[&i] i < @links)
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
        self.reduce(#op(+), 0)
    }

    fn prod() {
        self.reduce(#op(*), 1)
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
        Gen::new(||[&self, &i] { i += 1; self.next() },
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
}
