fn Vec(&T, &N: num) {
    struct Vec_ {
        let data: Tuple(T, N);
        let _index: num;

        static const capacity = N;
        static const T = T;
        static const _data_tuple_type = Tuple(T, N);

        static fn new() -> Self {
            self(self::_data_tuple_type.default(), 0)
        }

        const fn capacity() -> num {
            self::capacity
        }

        const fn size() -> num {
            self._index
        }

        const fn copy() -> Self {
            typeof(self).base(self.data, self._index)
        }

        fn set(i: num, &value: T) {
            self.data[i] = value;
        }

        const fn get(i: num) -> T {
            self.data[i] as self::T
        }

        fn push_back(&value: T) {
            self.data[self._index] = value;
            self._index += 1;
        }

        fn pop_back() -> T {
            self._index -= 1;
            self.data[self._index] as self::T
        }

        fn clear() {
            self._index = 0;
        }

        const fn @iter() {
            let i = 0;
            (||[&self, &i] {
                let v = self.data[i] as self::T;
                i += 1;
                v
            }, ||[&self, &i] {
                i < self._index
            })
        }

        fn @index(i: num) {
            self.data[i]
        }

        fn append(&other) {
            if const typeof(other).is_tuple {
                other.map(|x|[&self] self.push_back(x));
            } else {
                for x in other {
                    self.push_back(x);
                }
            }
        }
    }
}
