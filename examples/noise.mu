// number of segments the display is split into
let PM = 11;
// display size
let DS = 176;
// display segment size
let dsize = DS / PM;

// foreground color, variable is shadowed later
let fg_color = (0, 1, 0.2);
// background color
let bg_color = (0, 0, 0);
let fade_speed = 48;

let seed = 0;
while true {
    // random foreground color
    // shadows previous variable with the same name
    // this one is used
    let fg_color = (rand(1), rand(1), rand(1));

    for y in 0..dsize {
        let y_offset = y * PM;
        let y_seed = y + seed;
        for x in 0..dsize {
            let val = abs(noise(x + seed, y_seed)) * 255;
            // operation is performed with all elements of tuple
            // equivalent to (val * fg_color.0, val * fg_color.1, val * fg_color.2)
            // use the "..." operator to unpack a tuple into function parameters
            draw.color((val * fg_color)..., 255);
            draw.rect(x * PM, y_offset, PM, PM);
        }
        drawflush(Block::display1);
        // tuple concatenation is performed using the "++" operator
        draw.color((bg_color ++ (fade_speed,))...);
        draw.rect(0, 0, DS, DS);
    }

    seed += 1;
}
