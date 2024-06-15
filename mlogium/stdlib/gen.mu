fn g_make(&next, &has) {
    (next, has)
}

fn g_map(&f, &g) {
    g_make(||[&g, &f] f(g.0()),
           ||[&g] g.1())
}

fn g_unpack_map(&f, &g) {
    g_make(||[&g, &f] f(g.0()...),
           ||[&g] g.1())
}

fn g_zip(&a, &b) {
    g_make(||[&a, &b] (a.0(), b.0()),
           ||[&a, &b] a.1() && b.1())
}

fn g_range(&n: num) {
    let i = 0;
    g_make(||[&i] {
        let x = i;
        i += 1;
        x
    }, ||[&i, &n] i < n)
}

fn g_buildings() {
    g_map(getlink, g_range(@links))
}

fn g_foreach(&f, &g) {
    g_make(||[&g, &f] { let v = g.0(); f(v); v },
           ||[&g] g.1())
}

fn g_reduce(&f, &g, s) {
    for v in g {
        s = f(s, v);
    }
    s
}

fn g_sum(&g) {
    g_reduce(#op(+), g, 0)
}

fn g_prod(&g) {
    g_reduce(#op(*), g, 1)
}

fn g_consume(&g) {
    for _ in g {}
}

fn g_foreach_consume(&f, &g) {
    for v in g {
        f(v);
    }
}

fn g_skip(&g) {
    if g.1() {
        g.0();
    }
    g
}

fn g_skip_n(&g, &n: num) {
    for _ in 0..n {
        if !g.1() {
            break;
        }
        g.0();
    }
    g
}
