let cell = Block::cell1;

// writes can be direct
cell[0] = 12;
cell[12] = 64;

// the "*" operator dereferences values when indexing
let idx = *cell[0];
// an alternative is to use a cast
print(cell[idx] as num);

// the read and write methods can be used for complex values
cell.write(0, (1, 2));
// the type to be read has to be specified
print(cell.read(0, Tuple[num, num]));

printflush(Block::message1);
