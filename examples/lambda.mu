fn square(x: num) -> num {
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

struct A {
    let x: num

    fn mod(n: num) -> ? {
        |&a|[n] { a.x += n }
    }

    fn mod2() -> ? {
        |n|[&self] { self.x += n }
    }

    fn add(n: num) {
        self.x += n;
    }
}

let a = A(1);
a.mod(3)(a);
a.mod2()(4);
a.add(2);
print(a.x);
