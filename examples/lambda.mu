fn square(x) {
    x * x
}

let lam = |x| x * x;

let double_apply = |f| |x|[f] f(f(x));

let doubled_1 = double_apply(lam);
let doubled_2 = double_apply(square);

print(lam(3));
print(square(3));
print(doubled_1(3));
print(doubled_2(3));

fn transform(&f, &s) {
    typeof(s).struct_base(f(...s)...)
}

struct A {
    let x: num;

    fn mod(n) {
        |&a|[n] { a.x += n }
    }

    fn mod2() {
        |n|[&self] { self.x += n }
    }
}

let a = A(1);
a.mod(3)(a);
a.mod2()(4);
print(a.x);
print(transform(|&t| t.map(|&x| x + 1), a).x);
