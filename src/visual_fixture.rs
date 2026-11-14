//! A synthetic city used by the rendering tests: a style with a background,
//! land, water, parks, a road grid, points of interest and place names, and
//! the vector tiles that fill any camera's view of it.
//!
//! The geometry is derived from the tile id alone, so neighbouring tiles join
//! seamlessly and a pan reveals continuous content.

use std::{collections::BTreeMap, collections::HashMap, sync::Arc};

use geo_types::{Geometry, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon};
use maplibre_expr::{Color, Expr, Value};

use crate::{
    projection::{TILE_SIZE, TileId},
    style::{DemEncoding, LayerKind, MapStyle, SourceKind, StyleLayer, TileSource},
    tile::{TileFeature, TileLayer, VectorTile},
};

const EXTENT: f32 = 4_096.0;

fn literal(value: Value) -> Expr {
    Expr::Literal(value)
}

fn rgb(r: f64, g: f64, b: f64) -> Expr {
    literal(Value::Color(Color::new(r, g, b, 1.0)))
}

fn layer(id: &str, kind: LayerKind, source_layer: &str, properties: &[(&str, Expr)]) -> StyleLayer {
    StyleLayer {
        id: String::from(id),
        kind,
        source: Some(String::from("city")),
        source_layer: Some(String::from(source_layer)),
        min_zoom: f64::NEG_INFINITY,
        max_zoom: f64::INFINITY,
        filter: None,
        properties: properties
            .iter()
            .map(|(name, expression)| (String::from(*name), expression.clone()))
            .collect::<BTreeMap<_, _>>(),
        visible: true,
    }
}

/// The city's style.
pub fn style() -> MapStyle {
    let mut sources = BTreeMap::new();
    sources.insert(
        String::from("city"),
        TileSource {
            kind: SourceKind::Vector,
            templates: vec![String::from("https://tiles.example/{z}/{x}/{y}")],
            min_zoom: 0,
            max_zoom: 14,
            tile_size: TILE_SIZE,
            encoding: DemEncoding::Mapbox,
        },
    );
    let mut background = layer("background", LayerKind::Background, "", &[]);
    background.source = None;
    background.source_layer = None;
    background
        .properties
        .insert(String::from("background-color"), rgb(0.95, 0.93, 0.89));
    MapStyle {
        sources,
        layers: vec![
            background,
            layer(
                "water",
                LayerKind::Fill,
                "water",
                &[("fill-color", rgb(0.58, 0.78, 0.90))],
            ),
            layer(
                "parks",
                LayerKind::Fill,
                "park",
                &[("fill-color", rgb(0.70, 0.86, 0.66))],
            ),
            layer(
                "road-casing",
                LayerKind::Line,
                "road",
                &[
                    ("line-color", rgb(0.62, 0.60, 0.56)),
                    ("line-width", literal(Value::Number(9.0))),
                ],
            ),
            layer(
                "road",
                LayerKind::Line,
                "road",
                &[
                    ("line-color", rgb(1.0, 1.0, 1.0)),
                    ("line-width", literal(Value::Number(6.0))),
                ],
            ),
            layer(
                "poi",
                LayerKind::Circle,
                "poi",
                &[
                    ("circle-color", rgb(0.84, 0.30, 0.25)),
                    ("circle-radius", literal(Value::Number(5.0))),
                ],
            ),
            layer(
                "names",
                LayerKind::Symbol,
                "place",
                &[
                    (
                        "text-field",
                        literal(Value::String(String::from("Midtown"))),
                    ),
                    ("text-size", literal(Value::Number(15.0))),
                    ("text-color", rgb(0.15, 0.15, 0.18)),
                    ("text-halo-color", rgb(1.0, 1.0, 1.0)),
                    ("text-halo-width", literal(Value::Number(1.5))),
                ],
            ),
        ],
        fingerprint: 0x_c17a,
    }
}

fn rectangle(left: f32, top: f32, right: f32, bottom: f32) -> Geometry<f32> {
    Geometry::MultiPolygon(MultiPolygon::new(vec![Polygon::new(
        LineString::from(vec![
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom),
            (left, top),
        ]),
        Vec::new(),
    )]))
}

fn feature(geometry: Geometry<f32>) -> TileFeature {
    TileFeature {
        geometry,
        style: maplibre_expr::Feature::default(),
    }
}

/// One tile of the city. Roads run every quarter of a tile, parks and water
/// alternate by tile column and row.
pub fn tile(id: TileId) -> Arc<VectorTile> {
    let mut layers = HashMap::new();
    let mut put = |name: &str, features: Vec<TileFeature>| {
        layers.insert(
            String::from(name),
            TileLayer {
                extent: 4_096,
                features,
            },
        );
    };

    let water = if (id.x + id.y).is_multiple_of(5) {
        vec![feature(rectangle(0.0, 2_900.0, EXTENT, EXTENT))]
    } else {
        Vec::new()
    };
    put("water", water);

    let park = match (id.x % 3, id.y % 2) {
        (0, 0) => vec![feature(rectangle(300.0, 300.0, 1_700.0, 1_500.0))],
        (1, 1) => vec![feature(rectangle(2_200.0, 500.0, 3_700.0, 1_700.0))],
        _ => Vec::new(),
    };
    put("park", park);

    let mut roads = Vec::new();
    for step in 0..4_u8 {
        let offset = f32::from(step) * EXTENT / 4.0 + 120.0;
        roads.push(LineString::from(vec![(0.0, offset), (EXTENT, offset)]));
        roads.push(LineString::from(vec![(offset, 0.0), (offset, EXTENT)]));
    }
    roads.push(LineString::from(vec![(0.0, 0.0), (EXTENT, EXTENT)]));
    put(
        "road",
        vec![feature(Geometry::MultiLineString(MultiLineString::new(
            roads,
        )))],
    );

    put(
        "poi",
        vec![feature(Geometry::MultiPoint(MultiPoint::new(vec![
            Point::new(800.0, 700.0),
            Point::new(2_500.0, 2_300.0),
            Point::new(3_300.0, 3_400.0),
        ])))],
    );

    put(
        "place",
        vec![feature(Geometry::Point(Point::new(
            EXTENT / 2.0,
            EXTENT / 2.0,
        )))],
    );

    Arc::new(VectorTile {
        id,
        layers,
        byte_len: 8_192,
    })
}
