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
        self::new(||[&self, &f] f(self.next()),
                  ||[&self] self.has())
    }

    fn unpack_map(&f) {
        self::new(||[&self, &f] f(self.next()...),
                  ||[&self] self.has())
    }

    fn zip(&g) {
        self::new(||[&self, &g] (self.next(), g.next()),
                  ||[&self, &g] self.has() && g.has())
    }

    static fn range(&n: num) {
        let i = 0;
        Gen::new(||[&i] {
            let x = i;
            i += 1;
            x
        }, ||[&i, &n] i < n)
    }

    static fn buildings() {
        Gen::range(@links).map(getlink)
    }

    fn foreach(&f) {
        self::new(||[&self, &f] { let v = self.next(); f(v); v },
                  ||[&self] self.has())
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
