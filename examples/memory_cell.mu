let cell = Block::cell1;

// writes can be direct
cell[0] = 12;
cell[12] = 64;

// the "*" operator dereferences values when indexing
// an alternative is to use a cast
let idx = *cell[0];
print(cell[idx] as num);

printflush(Block::message1);
