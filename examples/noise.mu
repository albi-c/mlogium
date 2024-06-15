let PM = 11;
let DS = 176;
let dsize = DS / PM;

let fg_color = (0, 1, 0.2);
let bg_color = (0, 0, 0);
let fade_speed = 48;

let seed = 0;
while true {
    let fg_color = (rand(1), rand(1), rand(1));

    for y in 0..dsize {
        let y_offset = y * PM;
        let y_seed = y + seed;
        for x in 0..dsize {
            let val = abs(noise(x + seed, y_seed)) * 255;
            draw.color((#repeat(3, val) * fg_color)..., 255);
            draw.rect(x * PM, y_offset, PM, PM);
        }
        drawflush(ExternBlock::display1);
        draw.color((bg_color ++ (fade_speed,))...);
        draw.rect(0, 0, DS, DS);
    }

    seed += 1;
}
