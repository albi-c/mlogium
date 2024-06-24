struct Marker {
    let id: num;

    static fn _new(&type, id: num, x: num, y: num, replace: num) -> Marker {
        makemarker(type, id, x, y, replace);
        Marker(id)
    }
    static fn shapeText(id: num, x: num, y: num, replace: num) -> Marker {
        Marker::_new(MarkerType::shapeText, id, x, y, replace)
    }
    static fn point(id: num, x: num, y: num, replace: num) -> Marker {
        Marker::_new(MarkerType::point, id, x, y, replace)
    }
    static fn shape(id: num, x: num, y: num, replace: num) -> Marker {
        Marker::_new(MarkerType::shape, id, x, y, replace)
    }
    static fn text(id: num, x: num, y: num, replace: num) -> Marker {
        Marker::_new(MarkerType::text, id, x, y, replace)
    }
    static fn line(id: num, x: num, y: num, replace: num) -> Marker {
        Marker::_new(MarkerType::line, id, x, y, replace)
    }
    static fn texture(id: num, x: num, y: num, replace: num) -> Marker {
        Marker::_new(MarkerType::texture, id, x, y, replace)
    }
    static fn quad(id: num, x: num, y: num, replace: num) -> Marker {
        Marker::new(::quad, id, x, y, replace)
    }

    fn remove() {
        setmarker.remove(self.id);
    }
    fn world(enabled: num) -> Marker {
        setmarker.world(self.id, enabled);
        self
    }
    fn minimap(enabled: num) -> Marker {
        setmarker.minimap(self.id, enabled);
        self
    }
    fn autoscale(enabled: num) -> Marker {
        setmarker.autoscale(self.id, enabled);
        self
    }
    fn pos(x: num, y: num) -> Marker {
        setmarker.pos(self.id, x, y);
        self
    }
    fn endPos(x: num, y: num) -> Marker {
        setmarker.endPos(self.id, x, y);
        self
    }
    fn drawLayer(layer: num) -> Marker {
        setmarker.drawLayer(self.id, layer);
        self
    }
    fn color(color: num) -> Marker {
        setmarker.color(self.id, color);
        self
    }
    fn radius(radius: num) -> Marker {
        setmarker.radius(self.id, radius);
        self
    }
    fn stroke(width: num) -> Marker {
        setmarker.stroke(self.id, width);
        self
    }
    fn rotation(angle: num) -> Marker {
        setmarker.rotation(self.id, angle);
        self
    }
    fn shape(sides: num, fill: num, outline: num) -> Marker {
        setmarker.shape(self.id, sides, fill, outline);
        self
    }
    fn arc(start: num, end: num) -> Marker {
        setmarker.arc(self.id, end);
        self
    }
    fn flushText(fetch: num) -> Marker {
        setmarker.flushText(self.id, fetch);
        self
    }
    fn fontSize(size: num) -> Marker {
        setmarker.fontSize(self.id, size);
        self
    }
    fn textHeight(height: num) -> Marker {
        setmarker.textHeight(self.id, height);
        self
    }
    fn labelFlags(background: num, outline: num) -> Marker {
        setmarker.labelFlags(self.id, background, outline);
        self
    }
    fn texture(name: str) -> Marker {
        setmarker.texture(self.id, false, name);
        self
    }
    fn texturePrintFlush() -> Marker {
        setmarker.texture(self.id, true, "");
        self
    }
    fn textureSize(width: num, height: num) -> Marker {
        setmarker.textureSize(self.id, width, height);
        self
    }
    fn posi(index: num, x: num, y: num) -> Marker {
        setmarker.posi(self.id, x, y);
        self
    }
    fn uvi(index: num, x: num, y: num) -> Marker {
        setmarker.uvi(self.id, x, y);
        self
    }
    fn colori(index: num, color: num) -> Marker {
        setmarker.colori(self.id, color);
        self
    }
}
