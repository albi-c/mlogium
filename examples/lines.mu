fn draw_line(x1: num, y1: num, x2: num, y2: num) {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let D = 2 * dy - dx;
    let y = y1;

    for x in x1..x2 {
        draw.rect(x, y, 1, 1);

        if (D > 0) {
            y += 1;
            D = D - 2 * dx;
        }
        D = D + 2 * dy;
    }
}

draw.clear(0, 0, 0);
for i in 10..100 {
    let r = abs(noise(i, 0));
    let g = abs(noise(i, 1));
    let b = abs(noise(i, 2));

    draw.color(r * 255, g * 255, b * 255, 255);
    draw_line(i, 10, 100, 100);
    drawflush(Block::display1);
}
