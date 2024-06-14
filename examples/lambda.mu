let app = |f| { f() };

fn square(x: num) -> num {
    app(|| x * x )
}

let lam = |x| x * x;

print(lam(3));
print(square(3));
