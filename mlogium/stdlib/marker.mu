struct Marker {
    let id: num;

    static fn new(&type: MarkerType, id: num, x: num, y: num, replace: num) {
        makemarker(type, id, x, y, replace);
        Marker(id)
    }
    static fn shapeText(id: num, x: num, y: num, replace: num) {
        Marker::new(::shapeText, id, x, y, replace)
    }
    static fn point(id: num, x: num, y: num, replace: num) {
        Marker::new(::point, id, x, y, replace)
    }
    static fn shape(id: num, x: num, y: num, replace: num) {
        Marker::new(::shape, id, x, y, replace)
    }
    static fn text(id: num, x: num, y: num, replace: num) {
        Marker::new(::text, id, x, y, replace)
    }
    static fn line(id: num, x: num, y: num, replace: num) {
        Marker::new(::line, id, x, y, replace)
    }
    static fn texture(id: num, x: num, y: num, replace: num) {
        Marker::new(::texture, id, x, y, replace)
    }
    static fn quad(id: num, x: num, y: num, replace: num) {
        Marker::new(::quad, id, x, y, replace)
    }

    fn remove() {
        setmarker.remove(self.id);
    }
    fn world(enabled: num) {
        setmarker.world(self.id, enabled);
        self
    }
    fn minimap(enabled: num) {
        setmarker.minimap(self.id, enabled);
        self
    }
    fn autoscale(enabled: num) {
        setmarker.autoscale(self.id, enabled);
        self
    }
    fn pos(x: num, y: num) {
        setmarker.pos(self.id, x, y);
        self
    }
    fn endPos(x: num, y: num) {
        setmarker.endPos(self.id, x, y);
        self
    }
    fn drawLayer(layer: num) {
        setmarker.drawLayer(self.id, layer);
        self
    }
    fn color(color: num) {
        setmarker.color(self.id, color);
        self
    }
    fn radius(radius: num) {
        setmarker.radius(self.id, radius);
        self
    }
    fn stroke(width: num) {
        setmarker.stroke(self.id, width);
        self
    }
    fn rotation(angle: num) {
        setmarker.rotation(self.id, angle);
        self
    }
    fn shape(sides: num, fill: num, outline: num) {
        setmarker.shape(self.id, sides, fill, outline);
        self
    }
    fn arc(start: num, end: num) {
        setmarker.arc(self.id, end);
        self
    }
    fn flushText(fetch: num) {
        setmarker.flushText(self.id, fetch);
        self
    }
    fn fontSize(size: num) {
        setmarker.fontSize(self.id, size);
        self
    }
    fn textHeight(height: num) {
        setmarker.textHeight(self.id, height);
        self
    }
    fn labelFlags(background: num, outline: num) {
        setmarker.labelFlags(self.id, background, outline);
        self
    }
    fn texture(name: str) {
        setmarker.texture(self.id, false, name);
        self
    }
    fn texturePrintFlush() {
        setmarker.texture(self.id, true, "");
        self
    }
    fn textureSize(width: num, height: num) {
        setmarker.textureSize(self.id, width, height);
        self
    }
    fn posi(index: num, x: num, y: num) {
        setmarker.posi(self.id, x, y);
        self
    }
    fn uvi(index: num, x: num, y: num) {
        setmarker.uvi(self.id, x, y);
        self
    }
    fn colori(index: num, color: num) {
        setmarker.colori(self.id, color);
        self
    }
}
