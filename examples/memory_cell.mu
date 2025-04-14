let cell = Block::cell1;

cell[0] = 12;
cell[12] = 64;
print(*cell[*cell[0]]);
printflush(Block::message1);
