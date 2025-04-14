fn square(x: num) -> num {
    x * x
}

// anonymous function
// takes one parameter, multiplies it with itself
let lam = |x| x * x;

// takes a function as a parameter
// returns a closure that captures the function
// when called, applies the captured function twice
let double_apply = |f| |x|[f] f(f(x));

// doubled_1(x) == lam(lam(x))
let doubled_1 = double_apply(lam);
// doubled_2(x) == square(square(x));
let doubled_2 = double_apply(square);

print(lam(3));
print(square(3));
print(doubled_1(3));
print(doubled_2(3));

struct A {
    let x: num;

    static fn mod(n) {
        // closures can be returned from functions
        |&a|[n] { a.x += n }
    }

    fn mod2() {
        // the implicit "self" parameter can be captured
        |n|[&self] { self.x += n }
    }
}

let a = A(1);
A::mod(3)(a);
a.mod2()(4);
print(a.x);

// passes all fields of a structure converted to a tuple through the provided function
fn transform(&f, &s) {
    // the "..." prefix operators converts a structure into a tuple
    // A { x } -> (x,)
    // the provided function is then called on this tuple
    // the resulting tuple is unpacked into the structure's constructor
    typeof(s).base(f(...s)...)
}

// add 1 to all fields of structure "a"
print(transform(|&t| t.map(|&x| x + 1), a));
