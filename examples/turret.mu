for i in 0..@links {
    let turret = getlink(i);

    // enum names can be omitted in function calls
    let enemy = radar(RadarFilter::enemy, RadarFilter::any, ::any, ::distance, turret, 1);

    // null comparison should be performed using the "===" operator
    if enemy === null {
        control.shoot(turret, 0, 0, false);
    } else {
        control.shootp(turret, enemy, true);
    }
}
