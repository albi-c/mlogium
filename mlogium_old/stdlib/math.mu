struct Vec2 of [2]num {
    static fn new(&x: num, &y: num) -> Vec2 {
        Vec2((x, y))
    }

    static fn of(&x: num) -> Vec2 {
        Vec2::new(x, x)
    }

    static fn zero() -> Vec2 {
        Vec2::new(0, 0)
    }

    fn dot(&other: Vec2) -> num {
        (self * other).sum
    }

    fn length() -> num {
        sqrt(self.0 * self.0 + self.1 * self.1)
    }

    fn x() -> num {
        self.0
    }
    fn y() -> num {
        self.1
    }

    fn mul(&m: Mat2) -> Vec2 {
        m.mul(self)
    }
}

struct Vec3 of [3]num {
    static fn new(&x: num, &y: num, &z: num) -> Vec3 {
        Vec3((x, y, z))
    }

    static fn of(&x: num) -> Vec3 {
        Vec3::new(x, x, x)
    }

    static fn zero() -> Vec3 {
        Vec3::of(0)
    }

    fn dot(&other: Vec3) -> num {
        (self * other).sum
    }

    fn length() -> num {
        sqrt(self.0 * self.0 + self.1 * self.1 + self.2 * self.2)
    }

    fn x() -> num {
        self.0
    }
    fn y() -> num {
        self.1
    }
    fn z() -> num {
        self.2
    }

    fn xy() -> Vec2 {
        Vec2::new(self.0, self.1)
    }

    fn mul(&m: Mat3) -> Vec3 {
        m.mul(self)
    }
}

struct Vec4 of [4]num {
    static fn new(&x: num, &y: num, &z: num, &w: num) -> Vec4 {
        Vec4((x, y, z, w))
    }

    static fn of(&x: num) -> Vec4 {
        Vec4::new(x, x, x, x)
    }

    static fn zero() -> Vec4 {
        Vec4::of(0)
    }

    fn dot(&other: Vec4) -> num {
        (self * other).sum
    }

    fn length() -> num {
        sqrt(self.0 * self.0 + self.1 * self.1 + self.2 * self.2 + self.3 * self.3)
    }

    fn x() -> num {
        self.0
    }
    fn y() -> num {
        self.1
    }
    fn z() -> num {
        self.2
    }
    fn w() -> num {
        self.3
    }

    fn xy() -> Vec2 {
        Vec2::new(self.0, self.1)
    }
    fn xyz() -> Vec3 {
        Vec3::new(self.0, self.1, self.2)
    }

    fn mul(&m: Mat4) -> Vec4 {
        m.mul(self)
    }
}

struct Mat2 of [2]Vec2 {
    static fn new(&a: Vec2, &b: Vec2) -> Mat2 {
        Mat2((a, b))
    }

    static fn zero() -> Mat2 {
        Mat2::new(Vec2::zero(), Vec2::zero())
    }

    static fn scale(&s: Vec2) -> Mat2 {
        Mat2::new(Vec2::new(s.0, 0), Vec2::new(0, s.1))
    }

    static fn identity() -> Mat2 {
        Mat2::scale(Vec2::of(1))
    }

    fn mul(&v: Vec2) -> Vec2 {
        Vec2::new(self.0.dot(v), self.1.dot(v))
    }
}

struct Mat3 of [3]Vec3 {
    static fn new(&a: Vec3, &b: Vec3, &c: Vec3) -> Mat3 {
        Mat2((a, b, c))
    }

    static fn zero() -> Mat3 {
        Mat3::new(Vec3::zero(), Vec3::zero(), Vec3::zero())
    }

    static fn scale(&s: Vec3) -> Mat3 {
        Mat3::new(Vec3::new(s.0, 0, 0), Vec3::new(0, s.1, 0), Vec3::new(0, 0, s.2))
    }

    static fn identity() -> Mat3 {
        Mat3::scale(Vec3::of(1))
    }

    fn mul(&v: Vec3) -> Vec3 {
        Vec3::new(self.0.dot(v), self.1.dot(v), self.2.dot(v))
    }
}

struct Mat4 of [4]Vec4 {
    static fn new(&a: Vec4, &b: Vec4, &c: Vec4, &d: Vec4) -> Mat4 {
        Mat2((a, b, c, d))
    }

    static fn zero() -> Mat4 {
        Mat4::new(Vec4::zero(), Vec4::zero(), Vec4::zero(), Vec4::zero())
    }

    static fn scale(&s: Vec4) -> Mat4 {
        Mat4::new(Vec4::new(s.0, 0, 0, 0), Vec4::new(0, s.1, 0, 0), Vec4::new(0, 0, s.2, 0), Vec4::new(0, 0, 0, s.3))
    }

    static fn identity() -> Mat4 {
        Mat4::scale(Vec4::of(1))
    }

    fn mul(&v: Vec4) -> Vec4 {
        Vec4::new(self.0.dot(v), self.1.dot(v), self.2.dot(v), self.3.dot(v))
    }
}
