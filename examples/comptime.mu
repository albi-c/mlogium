comptime {
    fn get_nums() {
        (1, 2)
    }

    fn get_num() {
        let (x, y) = get_nums();
        y
    }
}

print(get_num());
