from .value_types import Types, UnionTypeRef


ENUM_BLOCKS: set[str] = {
    "graphite-press", "multi-press", "silicon-smelter", "silicon-crucible", "kiln", "plastanium-compressor",
    "phase-weaver", "cryofluid-mixer", "pyratite-mixer", "blast-mixer", "melter", "separator", "disassembler",
    "spore-press", "pulverizer", "coal-centrifuge", "incinerator", "copper-wall", "copper-wall-large",
    "titanium-wall", "titanium-wall-large", "plastanium-wall", "plastanium-wall-large", "thorium-wall",
    "thorium-wall-large", "phase-wall", "phase-wall-large", "surge-wall", "surge-wall-large", "door", "door-large",
    "scrap-wall", "scrap-wall-large", "scrap-wall-huge", "scrap-wall-gigantic", "mender", "mend-projector",
    "overdrive-projector", "overdrive-dome", "force-projector", "shock-mine", "conveyor", "titanium-conveyor",
    "plastanium-conveyor", "armored-conveyor", "junction", "bridge-conveyor", "phase-conveyor", "sorter",
    "inverted-sorter", "router", "distributor", "overflow-gate", "underflow-gate", "mass-driver", "duct",
    "duct-router", "duct-bridge", "mechanical-pump", "rotary-pump", "conduit", "pulse-conduit", "plated-conduit",
    "liquid-router", "liquid-tank", "liquid-junction", "bridge-conduit", "phase-conduit", "power-node",
    "power-node-large", "surge-tower", "diode", "battery", "battery-large", "combustion-generator",
    "thermal-generator", "steam-generator", "differential-generator", "rtg-generator", "solar-panel",
    "solar-panel-large", "thorium-reactor", "impact-reactor", "mechanical-drill", "pneumatic-drill", "laser-drill",
    "blast-drill", "water-extractor", "cultivator", "oil-extractor", "core-shard", "core-foundation",
    "core-nucleus", "vault", "container", "unloader", "duo", "scatter", "scorch", "hail", "wave", "lancer", "arc",
    "parallax", "swarmer", "salvo", "segment", "tsunami", "fuse", "ripple", "cyclone", "foreshadow", "spectre",
    "meltdown", "command-center", "ground-factory", "air-factory", "naval-factory", "additive-reconstructor",
    "multiplicative-reconstructor", "exponential-reconstructor", "tetrative-reconstructor", "repair-point",
    "repair-turret", "payload-conveyor", "payload-router", "payload-propulsion-tower", "power-source", "power-void",
    "item-source", "item-void", "liquid-source", "liquid-void", "payload-void", "payload-source", "illuminator",
    "launch-pad", "interplanetary-accelerator", "message", "switch", "micro-processor", "logic-processor",
    "hyper-processor", "memory-cell", "memory-bank", "logic-display", "large-logic-display", "liquid-container",
    "deconstructor", "constructor", "thruster", "large-constructor", "payload-loader", "payload-unloader",
    "silicon-arc-furnace", "cliff-crusher", "plasma-bore", "reinforced-liquid-junction", "breach", "core-bastion",
    "turbine-condenser", "beam-node", "beam-tower", "build-tower", "impact-drill", "carbide-crucible",
    "surge-conveyor", "duct-unloader", "surge-router", "reinforced-conduit", "reinforced-liquid-router",
    "reinforced-liquid-container", "reinforced-liquid-tank", "reinforced-bridge-conduit", "core-citadel",
    "core-acropolis", "heat-reactor", "impulse-pump", "reinforced-pump", "electrolyzer", "oxidation-chamber",
    "surge-smelter", "surge-crucible", "overflow-duct", "large-plasma-bore", "cyanogen-synthesizer",
    "slag-centrifuge", "electric-heater", "slag-incinerator", "phase-synthesizer", "sublimate",
    "reinforced-container", "reinforced-vault", "atmospheric-concentrator", "unit-cargo-loader",
    "unit-cargo-unload-point", "chemical-combustion-chamber", "pyrolysis-generator", "regen-projector", "titan",
    "small-deconstructor", "vent-condenser", "phase-heater", "heat-redirector", "tungsten-wall",
    "tungsten-wall-large", "tank-assembler", "beryllium-wall", "beryllium-wall-large", "eruption-drill",
    "ship-assembler", "mech-assembler", "shield-projector", "beam-link", "world-processor",
    "reinforced-payload-conveyor", "reinforced-payload-router", "disperse", "large-shield-projector",
    "payload-mass-driver", "world-cell", "carbide-wall", "carbide-wall-large", "tank-fabricator", "mech-fabricator",
    "ship-fabricator", "reinforced-surge-wall", "radar", "blast-door", "canvas", "armored-duct", "shield-breaker",
    "unit-repair-tower", "diffuse", "prime-refabricator", "basic-assembler-module", "reinforced-surge-wall-large",
    "tank-refabricator", "mech-refabricator", "ship-refabricator", "slag-heater", "afflict", "shielded-wall",
    "lustre", "scathe", "smite", "underflow-duct", "malign", "shockwave-tower", "heat-source", "flux-reactor",
    "neoplasia-reactor"
}

ENUM_ITEMS: set[str] = {
    "copper", "lead", "metaglass", "graphite", "sand", "coal", "titanium", "thorium", "scrap", "silicon",
    "plastanium", "phase-fabric", "surge-alloy", "spore-pod", "blast-compound", "pyratite", "beryllium",
    "tungsten", "oxide", "carbide"
}

ENUM_LIQUIDS: set[str] = {
    "water", "slag", "oil", "cryofluid", "neoplasm", "arkycite",
    "ozone", "hydrogen", "nitrogen", "cyanogen"
}

ENUM_UNITS: set[str] = {
    "dagger", "mace", "fortress", "scepter", "reign",
    "nova", "pulsar", "quasar", "vela", "corvus",
    "crawler", "atrax", "spiroct", "arkyid", "toxopid",
    "flare", "horizon", "zenith", "antumbra", "eclipse",
    "mono", "poly", "mega", "quad", "oct",
    "risso", "minke", "bryde", "sei", "omura",
    "retusa", "oxynoe", "cyerce", "aegires", "navanax",
    "alpha", "beta", "gamma",

    "stell", "locus", "precept", "vanquish", "conquer",
    "merui", "cleroi", "anthicus", "tecta", "collaris",
    "elude", "avert", "obviate", "quell", "disrupt",
    "evoke", "incite", "emanate"
}

ENUM_TEAMS: set[str] = {
    "derelict", "sharded", "crux", "malis", "green", "blue"
}

ENUM_SENSABLE: dict[str, Types] = {
    "totalItems": Types.NUM,
    "firstItem": Types.ITEM_TYPE,
    "totalLiquids": Types.NUM,
    "totalPower": Types.NUM,
    "itemCapacity": Types.NUM,
    "liquidCapacity": Types.NUM,
    "powerCapacity": Types.NUM,
    "powerNetStored": Types.NUM,
    "powerNetCapacity": Types.NUM,
    "powerNetIn": Types.NUM,
    "powerNetOut": Types.NUM,
    "ammo": Types.NUM,
    "ammoCapacity": Types.NUM,
    "health": Types.NUM,
    "maxHealth": Types.NUM,
    "heat": Types.NUM,
    "efficiency": Types.NUM,
    "progress": Types.NUM,
    "timescale": Types.NUM,
    "rotation": Types.NUM,
    "x": Types.NUM,
    "y": Types.NUM,
    "shootX": Types.NUM,
    "shootY": Types.NUM,
    "size": Types.NUM,
    "dead": Types.NUM,
    "range": Types.NUM,
    "shooting": Types.NUM,
    "boosting": Types.NUM,
    "mineX": Types.NUM,
    "mineY": Types.NUM,
    "mining": Types.NUM,
    "speed": Types.NUM,
    "team": Types.TEAM,
    "type": Types.NUM,
    "flag": Types.NUM,
    "controlled": Types.CONTROLLER,
    "controller": UnionTypeRef([Types.BLOCK, Types.UNIT]),
    "name": Types.NUM,
    "payloadCount": Types.NUM,
    "payloadType": UnionTypeRef([Types.BLOCK_TYPE, Types.UNIT_TYPE]),
    "totalPayload": Types.NUM,
    "payloadCapacity": Types.NUM,
    "enabled": Types.NUM,
    "config": Types.CONTENT,
    "color": Types.NUM,
    "solid": Types.NUM,
    "memoryCapacity": Types.NUM,
    "bufferUsage": Types.NUM,
    "displayWidth": Types.NUM,
    "displayHeight": Types.NUM,
} | {
    name: Types.NUM for name in ENUM_ITEMS | ENUM_LIQUIDS
}

ENUM_RADAR_FILTER = {
    "any", "enemy", "ally", "player", "attacker", "flying", "boss", "ground"
}
ENUM_RADAR_SORT = {
    "distance", "health", "shield", "armor", "maxHealth"
}

ENUM_LOCATE_TYPE = {
    "core", "storage", "generator", "turret", "factory", "repair", "battery", "reactor"
}

ENUM_STATUS = {
    "none", "burning", "freezing", "unmoving", "slow", "fase", "wet", "muddy", "melting", "sapped", "electrified",
    "spore-slowed", "tarred", "overdrive", "overclock", "shielded", "boss", "shocked", "blasted", "corroded",
    "disarmed", "invincible"
}

ENUM_RULES = {
    rule: False for rule in (
        "currentWaveTimer", "waveTimer", "waves", "wave", "waveSpacing", "waveSending", "attackMode",
        "enemyCoreBuildRadius", "dropZoneRadius", "unitCap", "lighting", "ambientLight", "solarMultiplier",
        "canGameOver"
    )
} | {
    rule: True for rule in (
        "buildSpeed", "unitHealth", "unitBuildSpeed", "unitCost", "unitDamage", "blockHealth", "blockDamage",
        "rtsMinWeight", "rtsMinSquad", "unitMineSpeed"
    )
}

ENUM_PROPERTY = {
    "x", "y", "rotation", "flag", "health", "totalPower", "shield"
}

ENUM_EFFECT = {
    effect: [Types.NUM, Types.NUM] for effect in (
        "warn", "cross", "spawn", "bubble"
    )
} | {
    effect: [Types.NUM, Types.NUM, Types.NUM] for effect in (
        "placeBlock", "plackBlockSpark", "breakBlock", "smokeSmall", "smokeBig", "explosion"
    )
} | {
    effect: ([Types.NUM, Types.NUM, Types.ANY, Types.NUM], True, [], {2: "_"}) for effect in (
        "smokeCloud", "vapor", "hit", "hitSquare", "spark", "sparkBig", "drill", "drillBig", "smokePuff",
        "sparkExplosion"
    )
} | {
    effect: [Types.NUM, Types.NUM, Types.NUM, Types.NUM] for effect in (
        "trail", "breakProp", "shootSmall", "shootBig", "smokeColor", "smokeSquare", "smokeSquareBig", "sparkShoot",
        "sparkShootBig", "lightBlock", "crossExplosion", "wave"
    )
} | {
    "blockFall": ([Types.NUM, Types.NUM, Types.ANY, Types.ANY, Types.BLOCK], True, [], {2: "_", 3: "_"})
}

ENUM_WEATHER = {
    "snowing", "rain", "sandstorm", "sporestorm", "fog", "suspend-particles"
}

ENUM_MARKER_TYPE = {
    "shapeText", "point", "shape", "text", "line", "texture", "quad"
}

ENUM_ALIGN = {
    "center", "top", "bottom", "left", "right", "topLeft", "topRight", "bottomLeft", "bottomRight"
}

ALL_ENUMS: dict[str, tuple[set[str], bool, bool]] = {
    "BlockType": (ENUM_BLOCKS, True, False),
    "ItemType": (ENUM_ITEMS, True, False),
    "LiquidType": (ENUM_LIQUIDS, True, False),
    "UnitType": (ENUM_UNITS, True, False),
    "Team": (ENUM_TEAMS, True, False),
    "RadarFilter": (ENUM_RADAR_FILTER, False, True),
    "RadarSort": (ENUM_RADAR_SORT, False, True),
    "LocateType": (ENUM_LOCATE_TYPE, False, True),
    "Status": (ENUM_STATUS, False, True),
    "Property": (ENUM_PROPERTY, True, False),
    "Weather": (ENUM_WEATHER, True, False),
    "MarkerType": (ENUM_MARKER_TYPE, False, True),
    "Align": (ENUM_ALIGN, False, True)
}
