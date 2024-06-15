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
