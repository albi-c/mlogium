// generate integer in range [0-7]
let idx = floor(rand(7.9));

// generates a lot of code, but works for most types
let x = *(1, 2, 8, -3, 34, 13, 5000, -1294)[idx];
// generates small programs, but only works for integers
let y = @lookup_table(1, 2, 8, -3, 34, 13, 5000, -1294)[idx];

// always prints "1" - both lookup tables have the same values and use the same index
print(x == y);

printflush(Block::message1);
