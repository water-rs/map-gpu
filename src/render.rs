use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap, HashSet},
    rc::Rc,
    sync::Arc,
};

// `std::time::Instant` panics on `wasm32-unknown-unknown`; `web_time` routes
// to `performance.now()` there and is API-identical.
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use cherenkov::{
    Draw, Fixed, Font, FontId, FontSource, Glyph, GlyphRun, GlyphStyle, Group, Image, ImageData,
    ImageId, Paint, Picture, Recorder, Rgba8, Sampling, ShapeData, Srgb,
};
use executor_core::spawn_local;
use futures::{StreamExt as _, future::join_all, stream};
use geo::{BoundingRect as _, Simplify as _};
use geo_types::{Coord, Geometry, LineString, Polygon};
use kurbo::{Affine, BezPath, Circle, Point, Rect, Shape as _, Stroke};
use lru::LruCache;
use maplibre_expr::{EvaluationContext, Value, evaluate};
use nami::{Binding, Computed, Signal as _, binding, watcher::BoxWatcherGuard};
use parley::{FontContext, LayoutContext, PositionedLayoutItem, StyleProperty};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use waterui_core::animation::Animation;
use waterui_graphics::{
    RecordingResources, Registered, SceneContent, SceneInvalidator, SceneView, ScrollUnit,
    SurfaceInputEvent,
};
use waterui_map::{Annotation, Coordinate, Location, MapConfig, MapStatus, MapVisibility, Region};

use crate::{
    MapGestureController, MapGpuOptions, MapLoadError, network,
    projection::{Camera, TILE_OVERSCAN_PIXELS, TileId, Viewport},
    style::{LayerKind, MapStyle, SourceKind, StyleLayer, TileSource},
    tile::{DemTile, RasterTile, TileFeature, VectorTile},
    unblock,
};

/// Every decoded tile for one prepared viewport, grouped by source name.
///
/// A source serves exactly one payload kind (validated when the style loads),
/// so a source name appears in exactly one of these maps.
#[derive(Debug, Default)]
struct SourceTiles {
    vector: BTreeMap<String, Vec<Arc<VectorTile>>>,
    raster: BTreeMap<String, Vec<Arc<RasterTile>>>,
    dem: BTreeMap<String, Vec<Arc<DemTile>>>,
}

impl SourceTiles {
    fn insert(&mut self, source: String, tiles: LoadedTiles) {
        match tiles {
            LoadedTiles::Vector(tiles) => {
                self.vector.insert(source, tiles);
            }
            LoadedTiles::Raster(tiles) => {
                self.raster.insert(source, tiles);
            }
            LoadedTiles::Dem(tiles) => {
                self.dem.insert(source, tiles);
            }
        }
    }

    fn tile_count(&self) -> usize {
        self.vector.values().map(Vec::len).sum::<usize>()
            + self.raster.values().map(Vec::len).sum::<usize>()
            + self.dem.values().map(Vec::len).sum::<usize>()
    }
}

/// One source's decoded tiles, before they are filed into [`SourceTiles`].
enum LoadedTiles {
    Vector(Vec<Arc<VectorTile>>),
    Raster(Vec<Arc<RasterTile>>),
    Dem(Vec<Arc<DemTile>>),
}

const MAP_BACKGROUND: [f32; 4] = [0.973, 0.957, 0.941, 1.0];
/// A colour in sRGB: the space every `MapLibre` style property is authored in.
type Color = cherenkov::Color<Srgb>;
/// Inset of the map's own chrome (compass, scale bar) from the viewport edge.
const CHROME_INSET: f64 = 12.0;
const COMPASS_RADIUS: f64 = 17.0;
const SCALE_TICK: f64 = 5.0;
/// The widest a scale bar is allowed to grow before it is rounded down.
const SCALE_MAX_WIDTH: f64 = 120.0;
const CHROME_SURFACE: Color = Color::new([1.0, 1.0, 1.0, 0.88]);
const CHROME_BORDER: Color = Color::new([0.24, 0.26, 0.30, 0.85]);
const CHROME_LABEL: Color = Color::new([0.12, 0.13, 0.16, 1.0]);
const COMPASS_NORTH: Color = Color::new([0.86, 0.22, 0.20, 1.0]);
const COMPASS_SOUTH: Color = Color::new([0.62, 0.64, 0.68, 1.0]);

/// Which of the map's own chrome overlays the configuration asked for.
#[derive(Debug, Clone, Copy)]
struct MapChrome {
    compass: MapVisibility,
    scale: MapVisibility,
}

/// Rounded distances a scale bar is allowed to represent, in metres.
const SCALE_STEPS: &[f64] = &[
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    50.0,
    100.0,
    200.0,
    500.0,
    1_000.0,
    2_000.0,
    5_000.0,
    10_000.0,
    20_000.0,
    50_000.0,
    100_000.0,
    200_000.0,
    500_000.0,
    1_000_000.0,
    2_000_000.0,
    5_000_000.0,
];

/// Picks the largest round distance whose bar fits the allowed width.
fn scale_bar_span(camera: Camera, latitude: f64) -> Option<(f64, f64)> {
    SCALE_STEPS
        .iter()
        .rev()
        .map(|meters| (*meters, camera.meters_to_pixels(latitude, *meters)))
        .find(|(_, width)| *width <= SCALE_MAX_WIDTH && *width >= 24.0)
}

fn format_scale_distance(meters: f64) -> String {
    if meters >= 1_000.0 {
        let kilometers = meters / 1_000.0;
        if (kilometers.fract()).abs() < f64::EPSILON {
            format!("{kilometers:.0} km")
        } else {
            format!("{kilometers:.1} km")
        }
    } else {
        format!("{meters:.0} m")
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "viewport and tile extents are represented as f32"
)]
const fn gpu_scalar(value: u32) -> f32 {
    value as f32
}

#[derive(Debug, Clone, PartialEq)]
struct RequestKey {
    region: Region,
    viewport: Viewport,
}

/// An LRU map of decoded tiles that reports its own memory footprint, so the
/// three payload caches can be evicted against one shared byte budget.
#[derive(Debug)]
struct TileMap<T> {
    tiles: LruCache<(String, TileId), Arc<T>>,
    bytes: u64,
}

/// Implemented by every decoded tile payload so [`TileMap`] can size it.
trait CachedTile {
    fn id(&self) -> TileId;
    fn byte_len(&self) -> usize;
}

macro_rules! impl_cached_tile {
    ($($tile:ty),+ $(,)?) => {
        $(
            impl CachedTile for $tile {
                fn id(&self) -> TileId {
                    self.id
                }

                fn byte_len(&self) -> usize {
                    self.byte_len
                }
            }
        )+
    };
}

impl_cached_tile!(VectorTile, RasterTile, DemTile);

impl<T: CachedTile> TileMap<T> {
    fn new() -> Self {
        Self {
            tiles: LruCache::unbounded(),
            bytes: 0,
        }
    }

    fn get(&mut self, source: &str, id: TileId) -> Option<Arc<T>> {
        self.tiles.get(&(source.to_owned(), id)).cloned()
    }

    fn insert(&mut self, source: String, tile: &Arc<T>) {
        let key = (source, tile.id());
        if let Some(replaced) = self.tiles.put(key, Arc::clone(tile)) {
            self.bytes = self.bytes.saturating_sub(tile_bytes(replaced.byte_len()));
        }
        self.bytes = self.bytes.saturating_add(tile_bytes(tile.byte_len()));
    }

    /// Drops the least recently used tile, returning the bytes it freed.
    fn evict_lru(&mut self) -> Option<u64> {
        let (_, evicted) = self.tiles.pop_lru()?;
        let freed = tile_bytes(evicted.byte_len());
        self.bytes = self.bytes.saturating_sub(freed);
        Some(freed)
    }
}

fn tile_bytes(byte_len: usize) -> u64 {
    u64::try_from(byte_len).unwrap_or(u64::MAX)
}

#[derive(Debug)]
struct TileCache {
    vector: TileMap<VectorTile>,
    raster: TileMap<RasterTile>,
    dem: TileMap<DemTile>,
    maximum_bytes: u64,
}

impl TileCache {
    fn new(maximum_bytes: u64) -> Self {
        Self {
            vector: TileMap::new(),
            raster: TileMap::new(),
            dem: TileMap::new(),
            maximum_bytes,
        }
    }

    const fn bytes(&self) -> u64 {
        self.vector
            .bytes
            .saturating_add(self.raster.bytes)
            .saturating_add(self.dem.bytes)
    }

    /// Evicts across all three payload caches until the shared budget holds.
    ///
    /// Each round drops the least recently used tile from the currently
    /// largest cache, so one payload kind cannot starve the others.
    fn enforce_budget(&mut self) {
        while self.bytes() > self.maximum_bytes {
            let largest = self.vector.bytes.max(self.raster.bytes).max(self.dem.bytes);
            let freed = if largest == self.vector.bytes {
                self.vector.evict_lru()
            } else if largest == self.raster.bytes {
                self.raster.evict_lru()
            } else {
                self.dem.evict_lru()
            };
            if freed.is_none() {
                break;
            }
        }
    }
}

/// A `Draw` target that accepts [`Fixed`] values for every operand the map
/// painter emits.
///
/// [`StaticRecorder`][cherenkov::StaticRecorder] (the frozen base picture)
/// and [`Recorder`][cherenkov::Recorder] (live frames) both satisfy these
/// `From`s — `Fixed` is the `StaticRecorder`'s operand type itself and
/// converts into the `Recorder`'s `Live` signal — so one painter records
/// into either target.
trait ReplayDraw:
    Draw<
        Value<ShapeData>: From<Fixed<ShapeData>>,
        Value<Paint>: From<Fixed<Paint>>,
        Value<Stroke>: From<Fixed<Stroke>>,
        Value<GlyphRun>: From<Fixed<GlyphRun>>,
        Value<Rect>: From<Fixed<Rect>>,
        Value<Affine>: From<Fixed<Affine>>,
        Value<Group>: From<Fixed<Group>>,
    >
{
}

impl<D> ReplayDraw for D where
    D: Draw<
            Value<ShapeData>: From<Fixed<ShapeData>>,
            Value<Paint>: From<Fixed<Paint>>,
            Value<Stroke>: From<Fixed<Stroke>>,
            Value<GlyphRun>: From<Fixed<GlyphRun>>,
            Value<Rect>: From<Fixed<Rect>>,
            Value<Affine>: From<Fixed<Affine>>,
            Value<Group>: From<Fixed<Group>>,
        >
{
}

/// Dedup key for a font: the font collection's blob id plus the face index.
type FontKey = (u64, u32);

/// Dedup key for an image: the pixel buffer's allocation identity and extents.
type ImageKey = (usize, u32, u32);

/// Registered handles kept for as long as a recording names them.
///
/// Mirrors `SceneResources`'s contract: a source the recording first reaches
/// is registered in the frame that draws it, and a source the engine rejects
/// is remembered in `failed_*` so it is skipped rather than re-registered.
/// Callers decide the lifetime: the frozen base [`Picture`] keeps its
/// `Registrations` for as long as the picture replays, while the per-frame
/// overlay set calls [`Registrations::end_frame`] after each recording.
#[derive(Debug, Default)]
struct Registrations {
    fonts: HashMap<FontKey, Registered<Font>>,
    images: HashMap<ImageKey, Registered<Image<Rgba8>>>,
    used_fonts: HashSet<FontKey>,
    used_images: HashSet<ImageKey>,
    failed_fonts: HashSet<FontKey>,
    failed_images: HashSet<ImageKey>,
}

impl Registrations {
    /// The `FontId` for `font`, registering it with `table` on first use.
    fn font(
        &mut self,
        font: &parley::FontData,
        table: &mut RecordingResources<'_>,
    ) -> Option<FontId> {
        let key = (font.data.id(), font.index);
        if let Some(handle) = self.fonts.get(&key) {
            self.used_fonts.insert(key);
            return Some(table.name(handle));
        }
        if self.failed_fonts.contains(&key) {
            return None;
        }
        let source = FontSource::bytes(Arc::<[u8]>::from(font.data.data())).with_index(font.index);
        match table.font(source) {
            Ok(handle) => {
                let id = table.name(&handle);
                self.fonts.insert(key, handle);
                self.used_fonts.insert(key);
                Some(id)
            }
            Err(error) => {
                tracing::warn!("GPU map font registration failed: {error}");
                self.failed_fonts.insert(key);
                None
            }
        }
    }

    /// The `ImageId` for `pixels`, registering it with `table` on first use.
    fn image(
        &mut self,
        pixels: &Arc<[u8]>,
        width: u32,
        height: u32,
        table: &mut RecordingResources<'_>,
    ) -> Option<ImageId> {
        let key = (Arc::as_ptr(pixels).cast::<u8>() as usize, width, height);
        if let Some(handle) = self.images.get(&key) {
            self.used_images.insert(key);
            return Some(table.name(handle));
        }
        if self.failed_images.contains(&key) {
            return None;
        }
        let data = ImageData::<Rgba8>::new(width, height, Arc::clone(pixels))
            .and_then(|data| table.image(data));
        match data {
            Ok(handle) => {
                let id = table.name(&handle);
                self.images.insert(key, handle);
                self.used_images.insert(key);
                Some(id)
            }
            Err(error) => {
                tracing::warn!("GPU map image registration failed: {error}");
                self.failed_images.insert(key);
                None
            }
        }
    }

    /// Releases every registration the frame that just recorded did not name.
    fn end_frame(&mut self) {
        let used = std::mem::take(&mut self.used_fonts);
        self.fonts.retain(|key, _| used.contains(key));
        let used = std::mem::take(&mut self.used_images);
        self.images.retain(|key, _| used.contains(key));
    }
}

/// The frozen display list of a map's base layers plus the engine
/// registrations it names.
///
/// A map is recorded content: the base layers are painted once into a
/// `Picture` at the prepared camera and replayed each frame under the
/// transform from that camera to the live one, so pans and pinches animate
/// at display-list cost instead of re-shaping the world. The registrations
/// must live as long as the picture can be drawn: dropping one releases the
/// font or image the recording still names.
struct BaseScene {
    picture: Picture,
    /// Registrations the picture names: fonts for symbol labels and the
    /// `raster`, `hillshade` and `heatmap` tile images. Held, never read —
    /// dropping it releases resources the picture still names.
    #[expect(
        dead_code,
        reason = "holding the handles keeps the picture's resources registered"
    )]
    registrations: Registrations,
}

impl std::fmt::Debug for BaseScene {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("BaseScene").finish_non_exhaustive()
    }
}

/// A fully loaded map scene ready for scene recording.
///
/// This is an internal loading product: it is produced by the tile loader and
/// consumed by the painter, and there is no way (nor reason) to build one from
/// outside the crate.
pub struct PreparedMap {
    style: MapStyle,
    camera: Camera,
    tiles: SourceTiles,
    annotations: Vec<Annotation>,
    location: Option<Location>,
    chrome: MapChrome,
    painter: MapPainter,
    /// Registrations the per-frame overlay painting names.
    registrations: Registrations,
    base: Option<BaseScene>,
}

impl std::fmt::Debug for PreparedMap {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedMap")
            .field("camera", &self.camera)
            .field("tile_count", &self.tiles.tile_count())
            .field("annotation_count", &self.annotations.len())
            .field("has_location", &self.location.is_some())
            .field("has_base", &self.base.is_some())
            .finish_non_exhaustive()
    }
}

impl PreparedMap {
    /// Loads the style metadata and visible vector tiles for a viewport,
    /// standing up a fresh tile cache. This is the test entry point; the live
    /// scene path drives [`Self::load_with_style`] with a shared cache instead.
    ///
    /// # Errors
    ///
    /// Returns provider, style-expression, or vector-tile decode failures.
    #[cfg(test)]
    #[allow(
        clippy::future_not_send,
        reason = "prepared maps load on WaterUI's main-thread local executor and retain main-thread caches"
    )]
    pub(crate) async fn load(
        options: &MapGpuOptions,
        region: Region,
        width: u32,
        height: u32,
    ) -> Result<Self, MapLoadError> {
        let style = MapStyle::load(options, waterui_map::MapStyle::Standard).await?;
        let cache = Rc::new(RefCell::new(TileCache::new(options.tile_cache_bytes.get())));
        Self::load_with_style(options, style, region, Viewport { width, height }, cache).await
    }

    #[allow(
        clippy::future_not_send,
        reason = "prepared maps load on WaterUI's main-thread local executor and retain main-thread caches"
    )]
    async fn load_with_style(
        options: &MapGpuOptions,
        style: MapStyle,
        region: Region,
        viewport: Viewport,
        cache: Rc<RefCell<TileCache>>,
    ) -> Result<Self, MapLoadError> {
        let (min_zoom, max_zoom) = MapStyle::camera_zoom_range();
        let camera = Camera::new(region, viewport, min_zoom, max_zoom);
        tracing::debug!(
            zoom = camera.zoom,
            preferred_tile_zoom = camera.tile_zoom,
            "preparing GPU map viewport"
        );
        let active_sources = style
            .layers
            .iter()
            .filter(|layer| layer.active(camera.zoom))
            .filter_map(|layer| layer.source.as_ref())
            .collect::<HashSet<_>>();
        let source_requests = active_sources.into_iter().map(|source_name| {
            let source = style.sources.get(source_name);
            let cache = Rc::clone(&cache);
            async move {
                let source = source.ok_or_else(|| {
                    MapLoadError::Unsupported(format!(
                        "active style layer references undefined source {source_name}"
                    ))
                })?;
                let tile_ids = camera.visible_tiles(
                    f64::from(TILE_OVERSCAN_PIXELS),
                    source.min_zoom,
                    source.max_zoom,
                    source.tile_size,
                );
                let source_tiles =
                    load_source_tiles(options, source_name, source, &tile_ids, cache).await?;
                tracing::debug!(
                    source = source_name.as_str(),
                    kind = ?source.kind,
                    tile_zoom = tile_ids.first().map_or(source.min_zoom, |tile| tile.z),
                    requested_tiles = tile_ids.len(),
                    "loaded GPU map source"
                );
                Ok::<_, MapLoadError>((source_name.clone(), source_tiles))
            }
        });
        let tiles = join_all(source_requests).await.into_iter().try_fold(
            SourceTiles::default(),
            |mut tiles, loaded| {
                let (source_name, loaded) = loaded?;
                tiles.insert(source_name, loaded);
                Ok::<_, MapLoadError>(tiles)
            },
        )?;
        Ok(Self {
            style,
            camera,
            tiles,
            annotations: Vec::new(),
            location: None,
            chrome: MapChrome {
                compass: MapVisibility::Hidden,
                scale: MapVisibility::Hidden,
            },
            painter: MapPainter::default(),
            registrations: Registrations::default(),
            base: None,
        })
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "validated positive offscreen dimensions are compared in the integer GPU viewport domain"
)]
impl SceneContent for PreparedMap {
    fn build_scene(
        &mut self,
        recorder: &mut Recorder,
        resources: &mut RecordingResources<'_>,
        width: f32,
        height: f32,
    ) -> bool {
        assert_eq!(
            (width as u32, height as u32),
            (self.camera.viewport.width, self.camera.viewport.height),
            "PreparedMap must be rendered at its prepared viewport size"
        );
        self.append_base_scene_for_camera(recorder, resources, self.camera);
        self.painter.paint_overlays(
            recorder,
            self.camera,
            &self.annotations,
            self.location.as_ref(),
            self.chrome,
            &mut self.registrations,
            resources,
        );
        self.registrations.end_frame();
        false
    }
}

impl PreparedMap {
    /// Replays the base picture under `camera`, recording it on first use.
    ///
    /// The picture names fonts and tile images through `base.registrations`,
    /// held for as long as the picture can be replayed.
    fn append_base_scene_for_camera(
        &mut self,
        scene: &mut impl ReplayDraw,
        resources: &mut RecordingResources<'_>,
        camera: Camera,
    ) {
        if self.base.is_none() {
            self.base = Some(self.build_base(resources));
        }
        let base = self
            .base
            .as_ref()
            .expect("the base scene was just recorded");
        scene.picture(
            &base.picture,
            Fixed(camera_transform(self.camera, camera).unwrap_or(Affine::IDENTITY)),
        );
    }

    /// Records the base layers into a frozen picture, registering every font
    /// and tile image the recording names.
    fn build_base(&self, resources: &mut RecordingResources<'_>) -> BaseScene {
        let mut registrations = Registrations::default();
        let picture = Picture::record(|recorded| {
            MapPainter::default().paint_base(
                recorded,
                &self.style,
                self.camera,
                &self.tiles,
                &mut registrations,
                resources,
            );
        });
        BaseScene {
            picture,
            registrations,
        }
    }
}

fn camera_transform(source: Camera, target: Camera) -> Option<Affine> {
    if source.region == target.region && source.viewport == target.viewport {
        return None;
    }
    let scale = (target.zoom - source.zoom).exp2();
    let source_center = source.coordinate_point(source.region.center);
    let target_source_center = target.coordinate_point(source.region.center);
    Some(Affine::new([
        scale,
        0.0,
        0.0,
        scale,
        scale.mul_add(-source_center.0, target_source_center.0),
        scale.mul_add(-source_center.1, target_source_center.1),
    ]))
}

#[allow(
    clippy::future_not_send,
    reason = "tile loading runs on WaterUI's main-thread local executor and uses an Rc tile cache"
)]
async fn load_source_tiles(
    options: &MapGpuOptions,
    source_name: &str,
    source: &TileSource,
    ids: &[TileId],
    cache: Rc<RefCell<TileCache>>,
) -> Result<LoadedTiles, MapLoadError> {
    let encoding = source.encoding;
    match source.kind {
        SourceKind::Vector => load_tiles(
            options,
            source_name,
            source,
            ids,
            cache,
            |cache| &mut cache.vector,
            move |id, bytes| VectorTile::decode(id, bytes.to_vec()),
        )
        .await
        .map(LoadedTiles::Vector),
        SourceKind::Raster => load_tiles(
            options,
            source_name,
            source,
            ids,
            cache,
            |cache| &mut cache.raster,
            move |id, bytes| RasterTile::decode(id, &bytes),
        )
        .await
        .map(LoadedTiles::Raster),
        SourceKind::RasterDem => load_tiles(
            options,
            source_name,
            source,
            ids,
            cache,
            |cache| &mut cache.dem,
            move |id, bytes| DemTile::decode(id, &bytes, encoding),
        )
        .await
        .map(LoadedTiles::Dem),
    }
}

#[allow(
    clippy::future_not_send,
    reason = "tile loading runs on WaterUI's main-thread local executor and uses an Rc tile cache"
)]
async fn load_tiles<T, Decode>(
    options: &MapGpuOptions,
    source_name: &str,
    source: &TileSource,
    ids: &[TileId],
    cache: Rc<RefCell<TileCache>>,
    select: fn(&mut TileCache) -> &mut TileMap<T>,
    decode: Decode,
) -> Result<Vec<Arc<T>>, MapLoadError>
where
    T: CachedTile + Send + 'static,
    Decode: Fn(TileId, zenwave::utils::Bytes) -> Result<T, MapLoadError> + Clone + Send + 'static,
{
    if source.templates.is_empty() {
        return Err(MapLoadError::Unsupported(format!(
            "source {source_name} has no tile templates"
        )));
    }
    let requests = ids.iter().copied().enumerate().map(|(index, id)| {
        let cached = select(&mut cache.borrow_mut()).get(source_name, id);
        let source_name = source_name.to_owned();
        let template_index = (usize::try_from(id.x).unwrap_or(0)
            + usize::try_from(id.y).unwrap_or(0))
            % source.templates.len();
        let url = tile_url(&source.templates[template_index], id);
        let cache = Rc::clone(&cache);
        let decode = decode.clone();
        async move {
            if let Some(tile) = cached {
                return Ok((index, tile));
            }
            let bytes = network::fetch(
                &url,
                options.maximum_tile_bytes.get(),
                options.network_request_timeout(),
            )
            .await?;
            let tile = Arc::new(unblock(move || decode(id, bytes)).await?);
            let mut cache = cache.borrow_mut();
            select(&mut cache).insert(source_name, &tile);
            cache.enforce_budget();
            Ok::<_, MapLoadError>((index, tile))
        }
    });
    let mut requests =
        stream::iter(requests).buffer_unordered(options.in_flight_tile_request_limit().get());
    let mut tiles = vec![None; ids.len()];
    while let Some(result) = requests.next().await {
        let (index, tile) = result?;
        tiles[index] = Some(tile);
        futures_lite::future::yield_now().await;
    }
    Ok(tiles
        .into_iter()
        .map(|tile| tile.expect("every requested map tile must complete"))
        .collect())
}

fn tile_url(template: &str, id: TileId) -> String {
    template
        .replace("{z}", &id.z.to_string())
        .replace("{x}", &id.x.to_string())
        .replace("{y}", &id.y.to_string())
}

#[derive(Debug)]
struct MapSceneState {
    style: Option<MapStyle>,
    prepared: Option<PreparedMap>,
    prepared_generation: Option<u64>,
    pending_prepared: Option<(u64, PreparedMap)>,
    request: Option<RequestKey>,
    generation: u64,
    last_error: Option<String>,
}

#[derive(Debug, Clone, Copy)]
struct NetworkRetryState {
    policy: crate::MapNetworkRetryPolicy,
    attempt: u32,
}

impl NetworkRetryState {
    const fn new(policy: crate::MapNetworkRetryPolicy) -> Self {
        Self { policy, attempt: 1 }
    }

    fn retry_after(&mut self, error: &MapLoadError) -> Option<core::time::Duration> {
        if !error.is_retryable() || self.attempt >= self.policy.maximum_attempts().get() {
            return None;
        }
        let delay = self.policy.delay_after_failure(self.attempt);
        self.attempt += 1;
        Some(delay)
    }

    const fn failed_attempt(self) -> u32 {
        self.attempt - 1
    }
}

struct MapRequestTask {
    generation: u64,
    map_style: waterui_map::MapStyle,
    status: Option<Binding<MapStatus>>,
    region: Region,
    viewport: Viewport,
    options: MapGpuOptions,
    state: Rc<RefCell<MapSceneState>>,
    cache: Rc<RefCell<TileCache>>,
    invalidator: Rc<RefCell<Option<SceneInvalidator>>>,
}

impl MapRequestTask {
    fn is_current(&self) -> bool {
        self.state.borrow().generation == self.generation
    }

    fn trace_cancellation(&self, reason: &'static str) {
        tracing::debug!(
            generation = self.generation,
            current_generation = self.state.borrow().generation,
            state = ?Rc::as_ptr(&self.state),
            reason,
            "stopped superseded GPU map request work"
        );
    }

    #[expect(
        clippy::future_not_send,
        reason = "map request attempts run on WaterUI's main-thread local executor and retain main-thread scene caches"
    )]
    async fn load_attempt(
        &self,
        cached_style: &mut Option<MapStyle>,
    ) -> Option<Result<(MapStyle, PreparedMap), MapLoadError>> {
        let style = match cached_style.clone() {
            Some(style) => Ok(style),
            None => MapStyle::load(&self.options, self.map_style).await,
        };
        if !self.is_current() {
            self.trace_cancellation("cancelled superseded GPU map style load");
            return None;
        }
        let style = match style {
            Ok(style) => style,
            Err(error) => return Some(Err(error)),
        };
        if cached_style.is_none() {
            self.state.borrow_mut().style = Some(style.clone());
            *cached_style = Some(style.clone());
        }
        let result = PreparedMap::load_with_style(
            &self.options,
            style.clone(),
            self.region,
            self.viewport,
            Rc::clone(&self.cache),
        )
        .await
        .map(|prepared| (style, prepared));
        Some(result)
    }

    #[expect(
        clippy::future_not_send,
        reason = "map request retries run on WaterUI's main-thread local executor and retain main-thread scene caches"
    )]
    async fn load_with_retry(&self) -> Option<Result<(MapStyle, PreparedMap), MapLoadError>> {
        let mut cached_style = self.state.borrow().style.clone();
        let mut retry = NetworkRetryState::new(self.options.retry_policy());
        loop {
            if !self.is_current() {
                self.trace_cancellation("cancelled superseded GPU map request");
                return None;
            }
            let attempt_result = self.load_attempt(&mut cached_style).await?;
            if !self.is_current() {
                self.trace_cancellation("cancelled superseded GPU map request");
                return None;
            }
            match attempt_result {
                Ok(result) => return Some(Ok(result)),
                Err(error) => {
                    let Some(delay) = retry.retry_after(&error) else {
                        return Some(Err(error));
                    };
                    tracing::warn!(
                        generation = self.generation,
                        failed_attempt = retry.failed_attempt(),
                        maximum_attempts = retry.policy.maximum_attempts().get(),
                        retry_after_ms = delay.as_secs_f64() * 1_000.0,
                        %error,
                        "retrying transient GPU map network failure"
                    );
                    native_executor::sleep(delay).await;
                }
            }
        }
    }

    fn finish(&self, result: Result<(MapStyle, PreparedMap), MapLoadError>) {
        let mut state = self.state.borrow_mut();
        if state.generation != self.generation {
            drop(state);
            self.trace_cancellation("discarded superseded GPU map request");
            return;
        }
        match result {
            Ok((style, prepared)) => {
                state.style = Some(style);
                state.pending_prepared = Some((self.generation, prepared));
                state.last_error = None;
                if let Some(status) = &self.status {
                    status.set(MapStatus::Ready);
                }
                tracing::debug!(
                    generation = self.generation,
                    state = ?Rc::as_ptr(&self.state),
                    "completed GPU map request"
                );
            }
            Err(error) => {
                tracing::warn!(
                    generation = self.generation,
                    %error,
                    "GPU map request failed without replacing the last prepared frame"
                );
                if let Some(status) = &self.status {
                    status.set(MapStatus::Failed(error.to_string().into()));
                }
                state.last_error = Some(error.to_string());
            }
        }
        drop(state);
        invalidate(&self.invalidator);
    }

    #[expect(
        clippy::future_not_send,
        reason = "map request tasks run on WaterUI's main-thread local executor and retain main-thread scene caches"
    )]
    async fn run(self) {
        if let Some(result) = self.load_with_retry().await {
            self.finish(result);
        }
    }
}

#[derive(Debug)]
struct ResolvedMapFrame {
    viewport: Viewport,
    camera: Option<Camera>,
    /// The generation a resolved frame was prepared at; read by the
    /// failure-recovery tests that pin which snapshot a frame replayed.
    #[allow(dead_code)]
    prepared_generation: Option<u64>,
    annotations: Vec<Annotation>,
    location: Option<Location>,
    animating: bool,
}

#[derive(Debug, Clone, Copy)]
struct CameraTransition {
    from: Region,
    to: Region,
    started_at: Instant,
}

#[derive(Debug)]
struct CameraMotion {
    displayed: Region,
    target: Region,
    transition: Option<CameraTransition>,
}

impl CameraMotion {
    const fn new(region: Region) -> Self {
        Self {
            displayed: region,
            target: region,
            transition: None,
        }
    }

    fn update(&mut self, target: Region, animate: bool, animation: &Animation) -> (Region, bool) {
        let now = Instant::now();
        let (current, _) = self.sample(animation, now);
        if target != self.target {
            self.target = target;
            if animate {
                self.transition = Some(CameraTransition {
                    from: current,
                    to: target,
                    started_at: now,
                });
            } else {
                self.displayed = target;
                self.transition = None;
            }
        }
        self.sample(animation, now)
    }

    fn sample(&mut self, animation: &Animation, now: Instant) -> (Region, bool) {
        let Some(transition) = self.transition else {
            return (self.displayed, false);
        };
        let elapsed = now.duration_since(transition.started_at);
        if animation.is_complete(elapsed) {
            self.displayed = transition.to;
            self.transition = None;
            return (self.displayed, false);
        }
        self.displayed = interpolate_region(
            transition.from,
            transition.to,
            f64::from(animation.progress(elapsed)),
        );
        (self.displayed, true)
    }
}

fn interpolate_region(from: Region, to: Region, progress: f64) -> Region {
    assert!(
        from.latitude_delta.is_finite()
            && from.latitude_delta > 0.0
            && from.longitude_delta.is_finite()
            && from.longitude_delta > 0.0
            && to.latitude_delta.is_finite()
            && to.latitude_delta > 0.0
            && to.longitude_delta.is_finite()
            && to.longitude_delta > 0.0,
        "Map camera animation requires finite positive spans"
    );
    let latitude = (to.center.latitude.get() - from.center.latitude.get())
        .mul_add(progress, from.center.latitude.get());
    let longitude_delta =
        (to.center.longitude.get() - from.center.longitude.get() + 180.0).rem_euclid(360.0) - 180.0;
    let longitude = longitude_delta.mul_add(progress, from.center.longitude.get());
    let latitude_delta = ((to.latitude_delta.ln() - from.latitude_delta.ln())
        .mul_add(progress, from.latitude_delta.ln()))
    .exp();
    let longitude_delta = ((to.longitude_delta.ln() - from.longitude_delta.ln())
        .mul_add(progress, from.longitude_delta.ln()))
    .exp();
    Region::new(
        crate::map_coordinate(latitude, longitude, latitude_delta),
        latitude_delta,
        longitude_delta,
    )
}

/// Live scene content used by the GPU map hook.
pub struct MapScene {
    options: MapGpuOptions,
    region: Computed<Region>,
    request_region: Computed<Region>,
    animate_camera_changes: Computed<bool>,
    annotations: Computed<Vec<Annotation>>,
    location: Computed<Option<Location>>,
    map_style: waterui_map::MapStyle,
    status: Option<Binding<MapStatus>>,
    chrome: MapChrome,
    invalidator: Rc<RefCell<Option<SceneInvalidator>>>,
    state: Rc<RefCell<MapSceneState>>,
    cache: Rc<RefCell<TileCache>>,
    viewport_size: Binding<Option<(f32, f32)>>,
    _watchers: Vec<BoxWatcherGuard>,
    painter: MapPainter,
    camera_motion: CameraMotion,
    /// Registrations the per-frame overlay painting names.
    registrations: Registrations,
    /// The gesture controller this surface drives, when the view is
    /// interactive — `None` on a static map.
    interaction: Option<Rc<MapGestureController>>,
    /// Tracks the camera's gesture origin across a continuous scroll.
    surface_camera: SurfaceCameraGesture,
    /// Pixel pan accumulated since the current scroll gesture started.
    scroll_offset: Point,
}

impl std::fmt::Debug for MapScene {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("MapScene").finish_non_exhaustive()
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "validated positive scene dimensions are quantized to the integer GPU viewport domain"
)]
impl MapScene {
    /// Creates live scene content from the semantic map configuration.
    #[must_use]
    pub fn new(config: MapConfig, options: MapGpuOptions) -> Self {
        let request_region = config.region.clone();
        Self::with_viewport_signal(
            config,
            options,
            binding(None),
            request_region,
            Computed::constant(false),
        )
    }

    pub(crate) fn with_viewport_signal(
        config: MapConfig,
        options: MapGpuOptions,
        viewport_size: Binding<Option<(f32, f32)>>,
        request_region: Computed<Region>,
        animate_camera_changes: Computed<bool>,
    ) -> Self {
        let location = config
            .user_location
            .filter(|_| config.user_location_visibility.is_visible())
            .unwrap_or_else(|| Computed::constant(None));
        let invalidator = Rc::new(RefCell::new(None::<SceneInvalidator>));
        let initial_region = config.region.snapshot();
        let mut watchers = Vec::with_capacity(4);
        for signal in [
            &config.region as &dyn ErasedRedrawSignal,
            &request_region,
            &config.annotations,
        ] {
            watchers.push(signal.redraw_guard(Rc::clone(&invalidator)));
        }
        watchers.push(location.watch({
            let invalidator = Rc::clone(&invalidator);
            move |_| invalidate(&invalidator)
        }));
        Self {
            cache: Rc::new(RefCell::new(TileCache::new(options.tile_cache_bytes.get()))),
            options,
            viewport_size,
            region: config.region,
            request_region,
            animate_camera_changes,
            annotations: config.annotations,
            location,
            map_style: config.style,
            status: config.status,
            chrome: MapChrome {
                compass: config.compass_visibility,
                scale: config.scale_visibility,
            },
            invalidator,
            state: Rc::new(RefCell::new(MapSceneState {
                style: None,
                prepared: None,
                prepared_generation: None,
                pending_prepared: None,
                request: None,
                generation: 0,
                last_error: None,
            })),
            _watchers: watchers,
            painter: MapPainter::default(),
            camera_motion: CameraMotion::new(initial_region),
            registrations: Registrations::default(),
            interaction: None,
            surface_camera: SurfaceCameraGesture::default(),
            scroll_offset: Point::ZERO,
        }
    }

    fn request(&self, viewport: Viewport) {
        let region = self.request_region.snapshot();
        let key = RequestKey { region, viewport };
        if self.state.borrow().request.as_ref() == Some(&key) {
            return;
        }
        let generation = {
            let mut state = self.state.borrow_mut();
            state.generation = state
                .generation
                .checked_add(1)
                .expect("Map load generation overflowed");
            state.request = Some(key);
            state.pending_prepared = None;
            state.last_error = None;
            state.generation
        };
        if let Some(status) = &self.status {
            status.set(MapStatus::Loading);
        }
        tracing::debug!(
            generation,
            width = viewport.width,
            height = viewport.height,
            latitude = region.center.latitude.get(),
            longitude = region.center.longitude.get(),
            latitude_delta = region.latitude_delta,
            longitude_delta = region.longitude_delta,
            state = ?Rc::as_ptr(&self.state),
            "started GPU map request"
        );
        spawn_local(
            MapRequestTask {
                generation,
                map_style: self.map_style,
                status: self.status.clone(),
                region,
                viewport,
                options: self.options.clone(),
                state: Rc::clone(&self.state),
                cache: Rc::clone(&self.cache),
                invalidator: Rc::clone(&self.invalidator),
            }
            .run(),
        )
        .detach();
    }

    fn resolve_frame(&mut self, width: f32, height: f32) -> ResolvedMapFrame {
        let size = (width.max(1.0), height.max(1.0));
        if self.viewport_size.snapshot() != Some(size) {
            self.viewport_size.set(Some(size));
        }
        let viewport = Viewport {
            width: size.0 as u32,
            height: size.1 as u32,
        };
        let (region, animating) = self.camera_motion.update(
            self.region.snapshot(),
            self.animate_camera_changes.snapshot(),
            &self.options.camera_animation,
        );
        if !animating {
            self.request(viewport);
        }

        let mut state = self.state.borrow_mut();
        if !animating && let Some((generation, prepared)) = state.pending_prepared.take() {
            state.prepared = Some(prepared);
            state.prepared_generation = Some(generation);
            tracing::debug!(
                generation,
                state = ?Rc::as_ptr(&self.state),
                "activated prepared GPU map"
            );
        }
        let camera = state
            .style
            .as_ref()
            .map(|_| MapStyle::camera_zoom_range())
            .and_then(|(min_zoom, max_zoom)| {
                state
                    .prepared
                    .as_ref()
                    .map(|_| Camera::new(region, viewport, min_zoom, max_zoom))
            });
        ResolvedMapFrame {
            viewport,
            camera,
            prepared_generation: state.prepared_generation,
            annotations: self.annotations.snapshot(),
            location: self.location.snapshot(),
            animating,
        }
    }
}

trait ErasedRedrawSignal {
    fn redraw_guard(&self, invalidator: Rc<RefCell<Option<SceneInvalidator>>>) -> BoxWatcherGuard;
}

impl<T: 'static> ErasedRedrawSignal for Computed<T> {
    fn redraw_guard(&self, invalidator: Rc<RefCell<Option<SceneInvalidator>>>) -> BoxWatcherGuard {
        self.watch(move |_| invalidate(&invalidator))
    }
}

fn invalidate(invalidator: &Rc<RefCell<Option<SceneInvalidator>>>) {
    if let Some(invalidator) = invalidator.borrow().as_ref() {
        invalidator();
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "validated positive scene dimensions are quantized to the integer GPU viewport domain"
)]
impl SceneContent for MapScene {
    fn build_scene(
        &mut self,
        recorder: &mut Recorder,
        resources: &mut RecordingResources<'_>,
        width: f32,
        height: f32,
    ) -> bool {
        let frame = self.resolve_frame(width, height);
        let mut state = self.state.borrow_mut();
        fill_viewport(recorder, frame.viewport, Color::new(MAP_BACKGROUND));
        if let (Some(prepared), Some(camera)) = (state.prepared.as_mut(), frame.camera) {
            prepared.append_base_scene_for_camera(recorder, resources, camera);
            self.painter.paint_overlays(
                recorder,
                camera,
                &frame.annotations,
                frame.location.as_ref(),
                self.chrome,
                &mut self.registrations,
                resources,
            );
        }
        self.registrations.end_frame();
        frame.animating
    }

    fn set_invalidator(&mut self, invalidator: Option<SceneInvalidator>) {
        *self.invalidator.borrow_mut() = invalidator;
        invalidate(&self.invalidator);
    }

    fn wants_input_events(&self) -> bool {
        self.interaction.is_some()
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the accumulated scroll offset is a display-space pixel distance"
    )]
    fn input(&mut self, event: &SurfaceInputEvent) {
        let SurfaceInputEvent::Scroll {
            delta_x,
            delta_y,
            unit,
            finished,
            ..
        } = event
        else {
            return;
        };
        // Line deltas are wheel notches; the map's own scrolling is the
        // trackpad's pixel stream.
        if *unit != ScrollUnit::Pixel {
            return;
        }
        let Some(interaction) = self.interaction.clone() else {
            return;
        };
        let Some((width, height)) = self.viewport_size.snapshot() else {
            return;
        };
        let viewport = Viewport {
            width: width.max(1.0) as u32,
            height: height.max(1.0) as u32,
        };
        // Scroll deltas are content-relative: `delta_x` is positive when the
        // content should move left, which is a negative pixel translation.
        self.scroll_offset.x -= delta_x;
        self.scroll_offset.y -= delta_y;
        let pan_offset = self.scroll_offset;
        self.surface_camera.apply(
            &interaction,
            SurfaceGesture::pan(pan_offset, true),
            viewport,
        );
        if *finished {
            // A `finished` scroll both ends a continuous gesture and carries
            // discrete notches; settle the accumulated pan on top of the
            // live application.
            self.surface_camera.apply(
                &interaction,
                SurfaceGesture::pan(pan_offset, false),
                viewport,
            );
            self.scroll_offset = Point::ZERO;
        }
        invalidate(&self.invalidator);
    }
}

/// One increment of a surface camera gesture, assembled from
/// [`SurfaceInputEvent::Scroll`] deltas (or by a test driving the gesture
/// math directly): a pixel pan, a pinch scale around an optional focal
/// point, and whether the gesture is still in flight.
#[derive(Clone, Copy)]
struct SurfaceGesture {
    pinch_scale: f32,
    pinch_center: Option<Point>,
    pan_offset: Point,
    active: bool,
}

impl SurfaceGesture {
    /// A pan-only gesture applying `offset` pixels.
    const fn pan(offset: Point, active: bool) -> Self {
        Self {
            pinch_scale: 1.0,
            pinch_center: None,
            pan_offset: offset,
            active,
        }
    }
}

/// Camera gesture driven by the surface's input events — trackpad scrolls on
/// desktop. Pinch-to-zoom reaches the map through the view's
/// `MagnificationGesture` metadata rather than a surface event.
#[derive(Default)]
struct SurfaceCameraGesture {
    origin: Option<Region>,
}

impl SurfaceCameraGesture {
    fn apply(
        &mut self,
        interaction: &MapGestureController,
        gesture: SurfaceGesture,
        viewport: Viewport,
    ) {
        if gesture.active && self.origin.is_none() {
            self.origin = Some(interaction.region.snapshot());
            interaction.animate_camera_changes.set(false);
        }
        let Some(origin) = self.origin else {
            return;
        };
        let width = gpu_scalar(viewport.width);
        let height = gpu_scalar(viewport.height);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "display-space gesture offsets fit the f32 translation domain"
        )]
        let pan_x = gesture.pan_offset.x as f32;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "display-space gesture offsets fit the f32 translation domain"
        )]
        let pan_y = gesture.pan_offset.y as f32;
        let mut region = crate::translated_region(
            origin,
            pan_x,
            pan_y,
            f64::from(viewport.width),
            f64::from(viewport.height),
        );
        // A pinch zooms around its focal point; without one (trackpad pinch
        // that never reported a center) the viewport center anchors the zoom.
        if (gesture.pinch_scale - 1.0).abs() > f32::EPSILON {
            let (center_x, center_y) = gesture.pinch_center.map_or_else(
                || (f64::from(width / 2.0), f64::from(height / 2.0)),
                |center| (center.x, center.y),
            );
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the focal point is a display-space position"
            )]
            {
                region = crate::magnified_region(
                    region,
                    f64::from(gesture.pinch_scale),
                    center_x as f32,
                    center_y as f32,
                    width,
                    height,
                );
            }
        }
        if gesture.active {
            interaction.set_live_region(region);
        } else {
            interaction.settle_region(region);
            self.origin = None;
        }
    }
}

/// Renders a custom [`MapScene`] as a view.
///
/// This is the public path from a hand-built scene to a renderable view: build
/// a [`MapScene`] from a [`MapConfig`](waterui_map::MapConfig), hand it here,
/// and place the returned [`SceneView`] like any other view. The built-in
/// `Map` view goes through the same surface, adding the gesture
/// controller that drives pan and zoom.
#[must_use]
pub fn map_surface(map: MapScene) -> SceneView {
    map_surface_with_interaction(map, None)
}

/// `map` with a gesture controller attached: scroll events reaching the
/// surface drive the map's camera through it.
pub fn map_surface_with_interaction(
    mut map: MapScene,
    interaction: Option<Rc<MapGestureController>>,
) -> SceneView {
    map.interaction = interaction;
    SceneView::new(map)
}

#[derive(Default)]
struct MapPainter {
    fonts: FontContext,
    layouts: LayoutContext<[u8; 4]>,
    occupied_labels: Vec<Rect>,
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::too_many_arguments,
    reason = "style evaluation converts bounded MapLibre numeric values and forwards explicit render-state parameters"
)]
impl MapPainter {
    fn paint_base(
        &mut self,
        scene: &mut impl ReplayDraw,
        style: &MapStyle,
        camera: Camera,
        tiles: &SourceTiles,
        registrations: &mut Registrations,
        resources: &mut RecordingResources<'_>,
    ) {
        self.paint_base_in_bounds(
            scene,
            style,
            camera,
            tiles,
            camera_render_bounds(camera),
            registrations,
            resources,
        );
    }

    #[expect(clippy::too_many_arguments, reason = "explicit render state")]
    fn paint_base_in_bounds(
        &mut self,
        scene: &mut impl ReplayDraw,
        style: &MapStyle,
        camera: Camera,
        tiles: &SourceTiles,
        render_bounds: Rect,
        registrations: &mut Registrations,
        resources: &mut RecordingResources<'_>,
    ) {
        self.occupied_labels.clear();
        for layer in &style.layers {
            if !layer.active(camera.zoom) {
                continue;
            }
            match layer.kind {
                LayerKind::Background => Self::paint_background(scene, layer, camera),
                LayerKind::Fill
                | LayerKind::FillExtrusion
                | LayerKind::Line
                | LayerKind::Symbol
                | LayerKind::Circle => {
                    self.paint_vector_layer(
                        scene,
                        layer,
                        camera,
                        tiles,
                        render_bounds,
                        registrations,
                        resources,
                    );
                }
                LayerKind::Heatmap => {
                    Self::paint_heatmap(
                        scene,
                        layer,
                        camera,
                        tiles,
                        render_bounds,
                        registrations,
                        resources,
                    );
                }
                LayerKind::Raster => {
                    Self::paint_raster(scene, layer, camera, tiles, registrations, resources);
                }
                LayerKind::Hillshade => {
                    Self::paint_hillshade(scene, layer, camera, tiles, registrations, resources);
                }
                // `sky` only has a visible extent once the camera can pitch away
                // from straight-down. This camera is always top-down, so the
                // layer is accepted, contributes nothing, and says so once.
                LayerKind::Sky => tracing::debug!(
                    layer = layer.id,
                    "skipped sky layer: the GPU map camera has no pitch"
                ),
            }
        }
    }

    /// Draws a `heatmap` layer.
    ///
    /// Density is accumulated on the CPU into a viewport-sized field, then run
    /// through the layer's `heatmap-color` ramp. That keeps the result on the
    /// same image path as `raster` and `hillshade` rather than adding a second
    /// GPU accumulation target, and the field is the only allocation.
    fn paint_heatmap(
        scene: &mut impl ReplayDraw,
        layer: &StyleLayer,
        camera: Camera,
        tiles: &SourceTiles,
        render_bounds: Rect,
        registrations: &mut Registrations,
        resources: &mut RecordingResources<'_>,
    ) {
        let zoom_context = EvaluationContext::new().with_zoom(camera.zoom);
        let opacity = property_number(layer, "heatmap-opacity", &zoom_context).unwrap_or(1.0);
        let radius = property_number(layer, "heatmap-radius", &zoom_context).unwrap_or(30.0);
        let intensity = property_number(layer, "heatmap-intensity", &zoom_context).unwrap_or(1.0);
        if opacity <= 0.0 || radius <= 0.0 || intensity <= 0.0 {
            return;
        }
        let source_name = layer
            .source
            .as_ref()
            .expect("heatmap layer source is validated when the style loads");
        let source_layer = layer
            .source_layer
            .as_ref()
            .unwrap_or_else(|| panic!("heatmap layer {} requires source-layer", layer.id));
        let source_tiles = tiles
            .vector
            .get(source_name)
            .unwrap_or_else(|| panic!("vector source {source_name} was not prepared"));

        let mut field = HeatmapField::new(camera.viewport, radius);
        for tile in source_tiles {
            let Some(tile_layer) = tile.layers.get(source_layer) else {
                continue;
            };
            for feature in &tile_layer.features {
                let context = EvaluationContext::new()
                    .with_zoom(camera.zoom)
                    .with_feature(feature.style.clone());
                if !passes_filter(layer, &context) {
                    continue;
                }
                let weight = property_number(layer, "heatmap-weight", &context).unwrap_or(1.0);
                if weight <= 0.0 {
                    continue;
                }
                for point in geometry_points(&feature.geometry) {
                    let (x, y) = camera.tile_point(tile.id, tile_layer.extent, point.x, point.y);
                    if render_bounds.contains(kurbo::Point::new(x, y)) {
                        field.accumulate(x, y, weight * intensity);
                    }
                }
            }
        }
        let Some(pixels) = field.colorize(layer, camera.zoom) else {
            return;
        };
        let Some(image) = registrations.image(
            &Arc::<[u8]>::from(pixels),
            camera.viewport.width,
            camera.viewport.height,
            resources,
        ) else {
            return;
        };
        let bounds = Rect::new(
            0.0,
            0.0,
            f64::from(camera.viewport.width),
            f64::from(camera.viewport.height),
        );
        let alpha = clamped_alpha(opacity);
        if alpha < 1.0 {
            scene.clip(Fixed(ShapeData::of(&render_bounds.to_path(0.1))), |scene| {
                scene.group(Fixed(Group::new().opacity(alpha)), |scene| {
                    scene.image(image, Fixed(bounds), Sampling::Linear);
                });
            });
        } else {
            scene.image(image, Fixed(bounds), Sampling::Linear);
        }
    }

    /// Draws a `raster` layer by placing each decoded image tile at the screen
    /// rect its tile id occupies under the current camera.
    fn paint_raster(
        scene: &mut impl ReplayDraw,
        layer: &StyleLayer,
        camera: Camera,
        tiles: &SourceTiles,
        registrations: &mut Registrations,
        resources: &mut RecordingResources<'_>,
    ) {
        let context = EvaluationContext::new().with_zoom(camera.zoom);
        let opacity = property_number(layer, "raster-opacity", &context).unwrap_or(1.0);
        if opacity <= 0.0 {
            return;
        }
        let source_name = layer
            .source
            .as_ref()
            .expect("raster layer source is validated when the style loads");
        let source_tiles = tiles
            .raster
            .get(source_name)
            .unwrap_or_else(|| panic!("raster source {source_name} was not prepared"));
        let alpha = clamped_alpha(opacity);
        let clip = ShapeData::of(
            &Rect::new(
                0.0,
                0.0,
                f64::from(camera.viewport.width),
                f64::from(camera.viewport.height),
            )
            .to_path(0.1),
        );
        if alpha < 1.0 {
            scene.clip(Fixed(clip), |scene| {
                scene.group(Fixed(Group::new().opacity(alpha)), |scene| {
                    for tile in source_tiles {
                        let Some(image) =
                            registrations.image(&tile.pixels, tile.width, tile.height, resources)
                        else {
                            continue;
                        };
                        scene.image(
                            image,
                            Fixed(tile_image_rect(camera, tile.id, tile.width)),
                            Sampling::Linear,
                        );
                    }
                });
            });
            return;
        }
        for tile in source_tiles {
            let Some(image) = registrations.image(&tile.pixels, tile.width, tile.height, resources)
            else {
                continue;
            };
            scene.image(
                image,
                Fixed(tile_image_rect(camera, tile.id, tile.width)),
                Sampling::Linear,
            );
        }
    }

    /// Draws a `hillshade` layer by shading each DEM tile on the CPU and
    /// blitting the result, which keeps terrain on the same image path as
    /// `raster` instead of introducing a second sampling mechanism.
    fn paint_hillshade(
        scene: &mut impl ReplayDraw,
        layer: &StyleLayer,
        camera: Camera,
        tiles: &SourceTiles,
        registrations: &mut Registrations,
        resources: &mut RecordingResources<'_>,
    ) {
        let context = EvaluationContext::new().with_zoom(camera.zoom);
        let exaggeration =
            property_number(layer, "hillshade-exaggeration", &context).unwrap_or(0.5);
        if exaggeration <= 0.0 {
            return;
        }
        let shadow = property_color(layer, "hillshade-shadow-color", &context)
            .unwrap_or(Color::new([0.0, 0.0, 0.0, 1.0]));
        let highlight = property_color(layer, "hillshade-highlight-color", &context)
            .unwrap_or(Color::new([1.0, 1.0, 1.0, 1.0]));
        let source_name = layer
            .source
            .as_ref()
            .expect("hillshade layer source is validated when the style loads");
        let source_tiles = tiles
            .dem
            .get(source_name)
            .unwrap_or_else(|| panic!("raster-dem source {source_name} was not prepared"));
        for tile in source_tiles {
            let shaded = Arc::<[u8]>::from(shade_dem_tile(
                tile,
                camera,
                exaggeration,
                shadow,
                highlight,
            ));
            let Some(image) = registrations.image(&shaded, tile.width, tile.height, resources)
            else {
                continue;
            };
            scene.image(
                image,
                Fixed(tile_image_rect(camera, tile.id, tile.width)),
                Sampling::Linear,
            );
        }
    }

    /// Draws one `circle` layer feature: a screen-space disc per geometry point.
    fn paint_circle(
        scene: &mut impl ReplayDraw,
        layer: &StyleLayer,
        context: &EvaluationContext,
        camera: Camera,
        tile: TileId,
        extent: u32,
        geometry: &Geometry<f32>,
    ) {
        let radius = property_number(layer, "circle-radius", context).unwrap_or(5.0);
        if radius <= 0.0 {
            return;
        }
        let fill = property_color(layer, "circle-color", context).map(|color| {
            color_with_opacity(
                color,
                property_number(layer, "circle-opacity", context).unwrap_or(1.0),
            )
        });
        let stroke_width = property_number(layer, "circle-stroke-width", context).unwrap_or(0.0);
        let stroke = (stroke_width > 0.0)
            .then(|| property_color(layer, "circle-stroke-color", context))
            .flatten()
            .map(|color| {
                color_with_opacity(
                    color,
                    property_number(layer, "circle-stroke-opacity", context).unwrap_or(1.0),
                )
            });
        if fill.is_none() && stroke.is_none() {
            return;
        }
        for point in geometry_points(geometry) {
            let (x, y) = camera.tile_point(tile, extent, point.x, point.y);
            let disc = ShapeData::of(&Circle::new((x, y), radius).to_path(0.1));
            if let Some(fill) = fill {
                scene.fill(Fixed(disc.clone()), Fixed(Paint::from(fill)));
            }
            if let Some(stroke) = stroke {
                scene.stroke(
                    Fixed(disc.clone()),
                    Fixed(Stroke::new(stroke_width)),
                    Fixed(Paint::from(stroke)),
                );
            }
        }
    }

    fn paint_overlays(
        &mut self,
        scene: &mut impl ReplayDraw,
        camera: Camera,
        annotations: &[Annotation],
        location: Option<&Location>,
        chrome: MapChrome,
        registrations: &mut Registrations,
        resources: &mut RecordingResources<'_>,
    ) {
        self.occupied_labels.clear();
        self.paint_annotations(scene, camera, annotations, registrations, resources);
        if let Some(location) = location {
            Self::paint_location(scene, camera, location);
        }
        if chrome.compass.is_visible() {
            Self::paint_compass(scene, camera);
        }
        if chrome.scale.is_visible() {
            self.paint_scale_bar(scene, camera, registrations, resources);
        }
    }

    /// Draws the north indicator in the top-trailing corner.
    ///
    /// This camera is always north-up, so the needle is fixed; it is drawn
    /// because the application asked for it, the same way a platform map keeps
    /// a compass affordance visible when configured to.
    fn paint_compass(scene: &mut impl ReplayDraw, camera: Camera) {
        let center = (
            f64::from(camera.viewport.width) - CHROME_INSET - COMPASS_RADIUS,
            CHROME_INSET + COMPASS_RADIUS,
        );
        let dial = ShapeData::of(&Circle::new(center, COMPASS_RADIUS).to_path(0.1));
        scene.fill(Fixed(dial.clone()), Fixed(Paint::from(CHROME_SURFACE)));
        scene.stroke(
            Fixed(dial),
            Fixed(Stroke::new(1.0)),
            Fixed(Paint::from(CHROME_BORDER)),
        );

        // The needle: a north half in the accent colour over a muted south half.
        let tip = COMPASS_RADIUS - 4.0;
        let waist = COMPASS_RADIUS * 0.34;
        let mut north = BezPath::new();
        north.move_to((center.0, center.1 - tip));
        north.line_to((center.0 - waist, center.1));
        north.line_to((center.0 + waist, center.1));
        north.close_path();
        scene.fill(
            Fixed(ShapeData::of(&north)),
            Fixed(Paint::from(COMPASS_NORTH)),
        );
        let mut south = BezPath::new();
        south.move_to((center.0, center.1 + tip));
        south.line_to((center.0 - waist, center.1));
        south.line_to((center.0 + waist, center.1));
        south.close_path();
        scene.fill(
            Fixed(ShapeData::of(&south)),
            Fixed(Paint::from(COMPASS_SOUTH)),
        );
    }

    /// Draws a scale bar whose length is a round distance at this latitude.
    fn paint_scale_bar(
        &mut self,
        scene: &mut impl ReplayDraw,
        camera: Camera,
        registrations: &mut Registrations,
        resources: &mut RecordingResources<'_>,
    ) {
        let latitude = camera.region.center.latitude.get();
        let Some((meters, width)) = scale_bar_span(camera, latitude) else {
            return;
        };
        let bottom = f64::from(camera.viewport.height) - CHROME_INSET;
        let left = CHROME_INSET;
        let mut bar = BezPath::new();
        bar.move_to((left, bottom - SCALE_TICK));
        bar.line_to((left, bottom));
        bar.line_to((left + width, bottom));
        bar.line_to((left + width, bottom - SCALE_TICK));
        scene.stroke(
            Fixed(ShapeData::of(&bar)),
            Fixed(Stroke::new(2.0)),
            Fixed(Paint::from(CHROME_BORDER)),
        );
        self.draw_label(
            scene,
            &format_scale_distance(meters),
            11.0,
            (left + width / 2.0, bottom - SCALE_TICK - 4.0),
            "center",
            (0.0, 0.0),
            0.0,
            2.0,
            CHROME_LABEL,
            CHROME_SURFACE,
            1.0,
            registrations,
            resources,
        );
    }

    fn paint_background(scene: &mut impl ReplayDraw, layer: &StyleLayer, camera: Camera) {
        let context = EvaluationContext::new().with_zoom(camera.zoom);
        let color = property_color(layer, "background-color", &context)
            .unwrap_or(Color::new([1.0, 1.0, 1.0, 1.0]));
        fill_viewport(scene, camera.viewport, color);
    }

    #[expect(clippy::too_many_arguments, reason = "explicit render state")]
    fn paint_vector_layer(
        &mut self,
        scene: &mut impl ReplayDraw,
        layer: &StyleLayer,
        camera: Camera,
        tiles: &SourceTiles,
        render_bounds: Rect,
        registrations: &mut Registrations,
        resources: &mut RecordingResources<'_>,
    ) {
        let source_name = layer
            .source
            .as_ref()
            .unwrap_or_else(|| panic!("layer {} requires a vector source", layer.id));
        let source_layer = layer
            .source_layer
            .as_ref()
            .unwrap_or_else(|| panic!("layer {} requires source-layer", layer.id));
        let source_tiles = tiles
            .vector
            .get(source_name)
            .unwrap_or_else(|| panic!("vector source {source_name} was not prepared"));
        let mut matched_features = 0_usize;
        let mut fill_batches = BTreeMap::<ColorKey, FillBatch>::new();
        let mut line_batches = BTreeMap::<LineKey, LineBatch>::new();
        for tile in source_tiles {
            let Some(tile_layer) = tile.layers.get(source_layer) else {
                continue;
            };
            for feature in &tile_layer.features {
                if !feature_visible(
                    camera,
                    tile.id,
                    tile_layer.extent,
                    &feature.geometry,
                    render_bounds,
                ) {
                    continue;
                }
                let context = EvaluationContext::new()
                    .with_zoom(camera.zoom)
                    .with_feature(feature.style.clone());
                if !passes_filter(layer, &context) {
                    continue;
                }
                matched_features += 1;
                match layer.kind {
                    LayerKind::Fill | LayerKind::FillExtrusion => {
                        collect_fill(
                            &mut fill_batches,
                            &mut line_batches,
                            layer,
                            &context,
                            camera,
                            tile.id,
                            tile_layer.extent,
                            &feature.geometry,
                        );
                    }
                    LayerKind::Line => collect_line(
                        &mut line_batches,
                        layer,
                        &context,
                        camera,
                        tile.id,
                        tile_layer.extent,
                        &feature.geometry,
                    ),
                    LayerKind::Symbol => self.paint_symbol(
                        scene,
                        layer,
                        &context,
                        camera,
                        tile.id,
                        tile_layer.extent,
                        feature,
                        render_bounds,
                        registrations,
                        resources,
                    ),
                    LayerKind::Circle => Self::paint_circle(
                        scene,
                        layer,
                        &context,
                        camera,
                        tile.id,
                        tile_layer.extent,
                        &feature.geometry,
                    ),
                    _ => unreachable!("vector layer kind was matched above"),
                }
            }
        }
        Self::paint_vector_batches(
            scene,
            layer,
            source_layer,
            matched_features,
            &fill_batches,
            &line_batches,
        );
    }

    fn paint_vector_batches(
        scene: &mut impl ReplayDraw,
        layer: &StyleLayer,
        source_layer: &str,
        matched_features: usize,
        fill_batches: &BTreeMap<ColorKey, FillBatch>,
        line_batches: &BTreeMap<LineKey, LineBatch>,
    ) {
        for batch in fill_batches.values() {
            emit_fill_batch(scene, batch);
        }
        for batch in line_batches.values() {
            emit_line_batch(scene, batch);
        }
        tracing::trace!(
            layer = layer.id,
            source_layer,
            matched_features,
            path_elements = fill_batches
                .values()
                .flat_map(|batch| batch.paths.iter())
                .map(|(_, path)| path.elements().len())
                .sum::<usize>()
                + line_batches
                    .values()
                    .flat_map(|batch| batch.paths.iter())
                    .map(|(_, path)| path.elements().len())
                    .sum::<usize>(),
            "encoded GPU map style layer"
        );
    }

    #[expect(clippy::too_many_arguments, reason = "explicit render state")]
    fn paint_symbol(
        &mut self,
        scene: &mut impl ReplayDraw,
        layer: &StyleLayer,
        context: &EvaluationContext,
        camera: Camera,
        tile: TileId,
        extent: u32,
        feature: &TileFeature,
        render_bounds: Rect,
        registrations: &mut Registrations,
        resources: &mut RecordingResources<'_>,
    ) {
        let Some(text) = property_string(layer, "text-field", context) else {
            return;
        };
        if text.trim().is_empty() {
            return;
        }
        let placement =
            property_string(layer, "symbol-placement", context).unwrap_or_else(|| "point".into());
        let Some((x, y, angle)) =
            feature_label_anchor(camera, tile, extent, &feature.geometry, &placement)
        else {
            return;
        };
        if !render_bounds.contains((x, y)) {
            return;
        }
        let size = property_number(layer, "text-size", context).unwrap_or(12.0) as f32;
        let text_color = property_color(layer, "text-color", context)
            .unwrap_or(Color::new([0.2, 0.2, 0.2, 1.0]));
        let halo_color = property_color(layer, "text-halo-color", context)
            .unwrap_or(Color::new([1.0, 1.0, 1.0, 0.85]));
        let halo_width = property_number(layer, "text-halo-width", context).unwrap_or(0.0);
        let padding = property_number(layer, "text-padding", context).unwrap_or(2.0);
        let anchor =
            property_string(layer, "text-anchor", context).unwrap_or_else(|| "center".into());
        let offset = property_number_pair(layer, "text-offset", context).unwrap_or((0.0, 0.0));
        self.draw_label(
            scene,
            &text,
            size,
            (x, y),
            &anchor,
            (offset.0 * f64::from(size), offset.1 * f64::from(size)),
            angle,
            padding,
            text_color,
            halo_color,
            halo_width,
            registrations,
            resources,
        );
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "label layout carries the full MapLibre text stack"
    )]
    fn draw_label(
        &mut self,
        scene: &mut impl ReplayDraw,
        text: &str,
        size: f32,
        anchor: (f64, f64),
        text_anchor: &str,
        offset: (f64, f64),
        angle: f64,
        padding: f64,
        color: Color,
        halo_color: Color,
        halo_width: f64,
        registrations: &mut Registrations,
        resources: &mut RecordingResources<'_>,
    ) {
        let mut builder = self
            .layouts
            .ranged_builder(&mut self.fonts, text, 1.0, true);
        builder.push_default(StyleProperty::FontSize(size));
        builder.push_default(StyleProperty::FontFamily(
            parley::style::GenericFamily::SansSerif.into(),
        ));
        let mut layout = builder.build(text);
        layout.break_all_lines(None);
        let width = f64::from(layout.width());
        let height = f64::from(layout.height());
        let origin = label_origin(anchor, offset, width, height, text_anchor);
        let rotated_width = f64::mul_add(height, angle.sin().abs(), width * angle.cos().abs());
        let rotated_height = f64::mul_add(height, angle.cos().abs(), width * angle.sin().abs());
        let center = (
            f64::mul_add(width, 0.5, origin.0),
            f64::mul_add(height, 0.5, origin.1),
        );
        let bounds = Rect::new(
            rotated_width.mul_add(-0.5, center.0) - padding,
            rotated_height.mul_add(-0.5, center.1) - padding,
            rotated_width.mul_add(0.5, center.0) + padding,
            rotated_height.mul_add(0.5, center.1) + padding,
        );
        if self
            .occupied_labels
            .iter()
            .any(|occupied| occupied.overlaps(bounds))
        {
            return;
        }
        self.occupied_labels.push(bounds);
        let transform = Affine::translate(center)
            * Affine::rotate(angle)
            * Affine::translate((-width * 0.5, -height * 0.5));
        let halo_stroke = Stroke::new(halo_width * 2.0);
        for line in layout.lines() {
            for item in line.items() {
                let PositionedLayoutItem::GlyphRun(glyph_run) = item else {
                    continue;
                };
                let run = glyph_run.run();
                let Some(font) = registrations.font(run.font(), resources) else {
                    continue;
                };
                let coords = Arc::<[i16]>::from(run.normalized_coords());
                let run_glyphs = glyphs(&glyph_run);
                let run_of = |style: GlyphStyle| GlyphRun {
                    font,
                    size: run.font_size(),
                    coords: Arc::clone(&coords),
                    glyphs: Arc::clone(&run_glyphs),
                    style,
                };
                // The halo is the same run painted underneath as a thick
                // stroke, so the fill that follows reads against any basemap.
                let fill_run = run_of(GlyphStyle::Fill);
                scene.transform(Fixed(transform), |scene| {
                    if halo_width > 0.0 {
                        scene.glyphs(
                            Fixed(run_of(GlyphStyle::Stroke(halo_stroke.clone()))),
                            Fixed(Paint::from(halo_color)),
                        );
                    }
                    scene.glyphs(Fixed(fill_run), Fixed(Paint::from(color)));
                });
            }
        }
    }

    fn paint_annotations(
        &mut self,
        scene: &mut impl ReplayDraw,
        camera: Camera,
        annotations: &[Annotation],
        registrations: &mut Registrations,
        resources: &mut RecordingResources<'_>,
    ) {
        for annotation in annotations {
            let (x, y) = camera.coordinate_point(annotation.coordinate);
            let marker = ShapeData::of(&Circle::new((x, y), 6.0).to_path(0.1));
            scene.fill(
                Fixed(marker),
                Fixed(Paint::from(Color::new([0.86, 0.12, 0.18, 1.0]))),
            );
            self.draw_label(
                scene,
                annotation.title.as_str(),
                13.0,
                (x, y - 15.0),
                "center",
                (0.0, 0.0),
                0.0,
                3.0,
                Color::new([0.12, 0.12, 0.14, 1.0]),
                Color::new([1.0, 1.0, 1.0, 0.95]),
                1.5,
                registrations,
                resources,
            );
        }
    }

    fn paint_location(scene: &mut impl ReplayDraw, camera: Camera, location: &Location) {
        let coordinate = Coordinate::from_location(location);
        let (x, y) = camera.coordinate_point(coordinate);
        if let Some(accuracy) = location.horizontal_accuracy() {
            let radius = camera
                .meters_to_pixels(location.latitude().get(), accuracy)
                .max(4.0);
            let accuracy_circle = ShapeData::of(&Circle::new((x, y), radius).to_path(0.2));
            scene.fill(
                Fixed(accuracy_circle.clone()),
                Fixed(Paint::from(Color::new([0.08, 0.45, 0.95, 0.14]))),
            );
            scene.stroke(
                Fixed(accuracy_circle),
                Fixed(Stroke::new(1.0)),
                Fixed(Paint::from(Color::new([0.08, 0.45, 0.95, 0.38]))),
            );
        }
        let outer = ShapeData::of(&Circle::new((x, y), 8.0).to_path(0.1));
        scene.fill(
            Fixed(outer),
            Fixed(Paint::from(Color::new([1.0, 1.0, 1.0, 1.0]))),
        );
        let inner = ShapeData::of(&Circle::new((x, y), 5.5).to_path(0.1));
        scene.fill(
            Fixed(inner),
            Fixed(Paint::from(Color::new([0.05, 0.42, 0.95, 1.0]))),
        );
    }
}

fn glyphs(glyph_run: &parley::GlyphRun<'_, [u8; 4]>) -> Arc<[Glyph]> {
    let mut run_x = glyph_run.offset();
    let run_y = glyph_run.baseline();
    glyph_run
        .glyphs()
        .map(|glyph| {
            let x = run_x + glyph.x;
            let y = run_y - glyph.y;
            run_x += glyph.advance;
            Glyph {
                id: glyph.id,
                x,
                y,
                transform: None,
            }
        })
        .collect()
}

fn fill_viewport(scene: &mut impl ReplayDraw, viewport: Viewport, color: Color) {
    let path = ShapeData::of(
        &Rect::new(
            0.0,
            0.0,
            f64::from(viewport.width),
            f64::from(viewport.height),
        )
        .to_path(0.1),
    );
    scene.fill(Fixed(path), Fixed(Paint::from(color)));
}

fn passes_filter(layer: &StyleLayer, context: &EvaluationContext) -> bool {
    layer.filter.as_ref().is_none_or(|filter| {
        evaluate(filter, context)
            .ok()
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ColorKey([u32; 4]);

impl ColorKey {
    fn new(color: Color) -> Self {
        Self(color.components().map(f32::to_bits))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct LineKey {
    color: ColorKey,
    width: u64,
}

/// A viewport-sized scalar density field used to rasterize `heatmap` layers.
struct HeatmapField {
    viewport: Viewport,
    radius: f64,
    density: Vec<f32>,
}

impl HeatmapField {
    fn new(viewport: Viewport, radius: f64) -> Self {
        let width = usize::try_from(viewport.width).expect("viewport width must fit usize");
        let height = usize::try_from(viewport.height).expect("viewport height must fit usize");
        Self {
            viewport,
            radius,
            density: vec![0.0; width * height],
        }
    }

    /// Splats one weighted sample with the quartic kernel `MapLibre` uses.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "screen-space splat bounds are clamped to the integer viewport"
    )]
    fn accumulate(&mut self, x: f64, y: f64, weight: f64) {
        let radius = self.radius;
        let width = i64::from(self.viewport.width);
        let height = i64::from(self.viewport.height);
        let min_x = ((x - radius).floor() as i64).max(0);
        let max_x = ((x + radius).ceil() as i64).min(width - 1);
        let min_y = ((y - radius).floor() as i64).max(0);
        let max_y = ((y + radius).ceil() as i64).min(height - 1);
        for row in min_y..=max_y {
            for column in min_x..=max_x {
                let dx = (f64::from(column as i32) - x) / radius;
                let dy = (f64::from(row as i32) - y) / radius;
                let squared = dx.mul_add(dx, dy * dy);
                if squared >= 1.0 {
                    continue;
                }
                // Quartic falloff: (1 - d^2)^2, peaking at the sample.
                let falloff = (1.0 - squared) * (1.0 - squared);
                let index = usize::try_from(row * width + column)
                    .expect("clamped heatmap index must fit usize");
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "density accumulates in f32 to halve the field's footprint"
                )]
                {
                    self.density[index] += (falloff * weight) as f32;
                }
            }
        }
    }

    /// Applies the layer's `heatmap-color` ramp to the normalized field.
    ///
    /// Returns `None` when nothing accumulated, so an empty heatmap costs no
    /// image upload at all.
    fn colorize(&self, layer: &StyleLayer, zoom: f64) -> Option<Vec<u8>> {
        let peak = self.density.iter().copied().fold(0.0_f32, f32::max);
        if peak <= 0.0 {
            return None;
        }
        let mut pixels = vec![0_u8; self.density.len() * 4];
        for (texel, density) in pixels.as_chunks_mut::<4>().0.iter_mut().zip(&self.density) {
            let normalized = f64::from(density / peak);
            // `heatmap-color` is a ramp over the synthetic `heatmap-density`
            // input rather than over feature properties.
            let mut context = EvaluationContext::new().with_zoom(zoom);
            context.heatmap_density = Some(normalized);
            let Some(color) = property_color(layer, "heatmap-color", &context) else {
                continue;
            };
            let components = color.components();
            for (channel, value) in texel.iter_mut().take(3).enumerate() {
                *value = channel_byte(components[channel]);
            }
            texel[3] = channel_byte(components[3]);
        }
        Some(pixels)
    }
}

/// The screen rect an image tile's pixel grid occupies under `camera`.
///
/// `tile_point` already places tile-local coordinates for an arbitrary extent,
/// so passing the image's pixel size as the extent maps texels straight to
/// device pixels.
fn tile_image_rect(camera: Camera, tile: TileId, size: u32) -> Rect {
    #[allow(
        clippy::cast_precision_loss,
        reason = "raster tile pixel sizes are small powers of two"
    )]
    let extent = size as f32;
    let (x0, y0) = camera.tile_point(tile, size, 0.0, 0.0);
    let (x1, y1) = camera.tile_point(tile, size, extent, extent);
    Rect::new(x0, y0, x1, y1)
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "style opacities are authored in f64 and encoded as f32 alpha"
)]
const fn clamped_alpha(opacity: f64) -> f32 {
    opacity.clamp(0.0, 1.0) as f32
}

/// Illumination `MapLibre` assumes when a style does not override it: light from
/// the top-left at 45 degrees.
const HILLSHADE_LIGHT_AZIMUTH: f32 = 315.0;
const HILLSHADE_LIGHT_ALTITUDE: f32 = 45.0;

/// Shades one DEM tile into RGBA8 with Horn's surface-normal hillshade.
///
/// Slope is metres of rise over metres of run, so the horizontal spacing comes
/// from the tile's ground resolution at this zoom rather than from texels.
fn shade_dem_tile(
    tile: &DemTile,
    camera: Camera,
    exaggeration: f64,
    shadow: Color,
    highlight: Color,
) -> Vec<u8> {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "shading runs in f32 over bounded camera and style inputs"
    )]
    let (spacing, exaggeration) = (
        camera.tile_ground_resolution(tile.id, tile.width) as f32,
        exaggeration as f32,
    );
    let azimuth = HILLSHADE_LIGHT_AZIMUTH.to_radians();
    let altitude = HILLSHADE_LIGHT_ALTITUDE.to_radians();
    let (light_x, light_y) = (azimuth.sin(), azimuth.cos());
    let (light_z, horizontal) = (altitude.sin(), altitude.cos());

    let width = usize::try_from(tile.width).expect("DEM tile width must fit usize");
    let mut pixels = vec![0_u8; width * usize::try_from(tile.height).unwrap_or(0) * 4];
    #[cfg(not(target_arch = "wasm32"))]
    pixels
        .par_chunks_exact_mut(width * 4)
        .enumerate()
        .for_each(|(row, line)| {
            shade_dem_row(
                tile,
                row,
                line,
                spacing,
                exaggeration,
                light_x,
                light_y,
                light_z,
                horizontal,
                shadow,
                highlight,
            );
        });
    #[cfg(target_arch = "wasm32")]
    pixels
        .chunks_exact_mut(width * 4)
        .enumerate()
        .for_each(|(row, line)| {
            shade_dem_row(
                tile,
                row,
                line,
                spacing,
                exaggeration,
                light_x,
                light_y,
                light_z,
                horizontal,
                shadow,
                highlight,
            );
        });
    pixels
}

#[allow(
    clippy::too_many_arguments,
    reason = "the row shader receives the precomputed light model"
)]
fn shade_dem_row(
    tile: &DemTile,
    row: usize,
    line: &mut [u8],
    spacing: f32,
    exaggeration: f32,
    light_x: f32,
    light_y: f32,
    light_z: f32,
    horizontal: f32,
    shadow: Color,
    highlight: Color,
) {
    let y = i64::try_from(row).expect("DEM row must fit i64");
    for (column, texel) in line.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let x = i64::try_from(column).expect("DEM column must fit i64");
        let slope_x =
            (tile.height_at(x - 1, y) - tile.height_at(x + 1, y)) * exaggeration / (2.0 * spacing);
        let slope_y =
            (tile.height_at(x, y - 1) - tile.height_at(x, y + 1)) * exaggeration / (2.0 * spacing);
        let normal = slope_x
            .mul_add(slope_x, slope_y.mul_add(slope_y, 1.0))
            .sqrt();
        let illumination = (slope_x
            .mul_add(light_x, slope_y * light_y)
            .mul_add(horizontal, light_z)
            / normal)
            .clamp(-1.0, 1.0);
        let (color, weight) = if illumination < 0.0 {
            (shadow, -illumination)
        } else {
            (highlight, illumination)
        };
        let components = color.components();
        for (channel, value) in texel.iter_mut().take(3).enumerate() {
            *value = channel_byte(components[channel]);
        }
        texel[3] = channel_byte(components[3] * weight);
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "normalized colour components are quantized to the 8-bit texture domain"
)]
fn channel_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Returns the points a point-oriented layer (`circle`, `heatmap`) draws from.
///
/// Non-point geometry is represented by its bounding-rect centre, matching how
/// `MapLibre` anchors these layers on non-point features.
fn geometry_points(geometry: &Geometry<f32>) -> Vec<Coord<f32>> {
    match geometry {
        Geometry::Point(point) => vec![point.0],
        Geometry::MultiPoint(points) => points.iter().map(|point| point.0).collect(),
        Geometry::Line(_)
        | Geometry::LineString(_)
        | Geometry::MultiLineString(_)
        | Geometry::Polygon(_)
        | Geometry::MultiPolygon(_)
        | Geometry::Rect(_)
        | Geometry::Triangle(_)
        | Geometry::GeometryCollection(_) => geometry
            .bounding_rect()
            .map(|bounds| vec![bounds.center()])
            .unwrap_or_default(),
    }
}

/// One colour's collected fill geometries for a style layer.
#[derive(Debug)]
struct FillBatch {
    color: Color,
    /// One merged path per tile, in collection order.
    paths: Vec<(TileId, BezPath)>,
}

/// One `(color, width)` pair's collected stroke geometries for a style layer.
#[derive(Debug)]
struct LineBatch {
    color: Color,
    width: f64,
    /// One merged path per tile, in collection order.
    paths: Vec<(TileId, BezPath)>,
}

/// Appends `path` to the batch's last tile entry, or opens a new one: features
/// arrive grouped by tile, so this merges all of a tile's same-colour
/// geometries into one path.
fn push_tile_path(paths: &mut Vec<(TileId, BezPath)>, tile: TileId, path: BezPath) {
    match paths.last_mut() {
        Some((last_tile, last_path)) if *last_tile == tile => {
            last_path.extend(path.elements().iter().copied());
        }
        _ => paths.push((tile, path)),
    }
}

/// Concatenates per-tile `paths` into one path: each tile's geometry was
/// already a set of self-contained subpaths, so the merge is purely
/// administrative.
fn merge_paths(paths: &[(TileId, BezPath)]) -> BezPath {
    let mut merged = BezPath::new();
    for (_, path) in paths {
        merged.extend(path.elements().iter().copied());
    }
    merged
}

/// Emits a fill batch. Opaque colours go out one path per tile: merged and
/// split draws cover the same winding region, and the engine rasterizes a
/// path once per recorded op, so a city-scale merged path would be one
/// quadratic rasterize while per-tile paths stay small. Translucent colours
/// stay fully merged — a single draw composites the alpha once over every
/// overlap, which split ops would darken.
fn emit_fill_batch(scene: &mut impl ReplayDraw, batch: &FillBatch) {
    if batch.color.components()[3] >= 1.0 {
        for (_, path) in &batch.paths {
            scene.fill(Fixed(ShapeData::of(path)), Fixed(Paint::from(batch.color)));
        }
    } else {
        scene.fill(
            Fixed(ShapeData::of(&merge_paths(&batch.paths))),
            Fixed(Paint::from(batch.color)),
        );
    }
}

/// Emits a stroke batch under the same opaque/translucent rule as
/// [`emit_fill_batch`].
fn emit_line_batch(scene: &mut impl ReplayDraw, batch: &LineBatch) {
    if batch.color.components()[3] >= 1.0 {
        for (_, path) in &batch.paths {
            scene.stroke(
                Fixed(ShapeData::of(path)),
                Fixed(Stroke::new(batch.width)),
                Fixed(Paint::from(batch.color)),
            );
        }
    } else {
        scene.stroke(
            Fixed(ShapeData::of(&merge_paths(&batch.paths))),
            Fixed(Stroke::new(batch.width)),
            Fixed(Paint::from(batch.color)),
        );
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "fill batching keeps style, projection, and MVT geometry inputs explicit"
)]
fn collect_fill(
    fills: &mut BTreeMap<ColorKey, FillBatch>,
    lines: &mut BTreeMap<LineKey, LineBatch>,
    layer: &StyleLayer,
    context: &EvaluationContext,
    camera: Camera,
    tile: TileId,
    extent: u32,
    geometry: &Geometry<f32>,
) {
    let color_property = if layer.kind == LayerKind::FillExtrusion {
        "fill-extrusion-color"
    } else {
        "fill-color"
    };
    let opacity_property = if layer.kind == LayerKind::FillExtrusion {
        "fill-extrusion-opacity"
    } else {
        "fill-opacity"
    };
    let Some(mut color) = property_color(layer, color_property, context) else {
        return;
    };
    color = color_with_opacity(
        color,
        property_number(layer, opacity_property, context).unwrap_or(1.0),
    );
    let Some(path) = geometry_path(camera, tile, extent, geometry, true) else {
        return;
    };
    let key = ColorKey::new(color);
    push_tile_path(
        &mut fills
            .entry(key)
            .or_insert_with(|| FillBatch {
                color,
                paths: Vec::new(),
            })
            .paths,
        tile,
        path,
    );
    if let Some(outline) = property_color(layer, "fill-outline-color", context)
        && let Some(path) = fill_outline_path(camera, tile, extent, geometry)
    {
        let key = LineKey {
            color: ColorKey::new(outline),
            width: 1.0_f64.to_bits(),
        };
        push_tile_path(
            &mut lines
                .entry(key)
                .or_insert_with(|| LineBatch {
                    color: outline,
                    width: 1.0,
                    paths: Vec::new(),
                })
                .paths,
            tile,
            path,
        );
    }
}

fn collect_line(
    lines: &mut BTreeMap<LineKey, LineBatch>,
    layer: &StyleLayer,
    context: &EvaluationContext,
    camera: Camera,
    tile: TileId,
    extent: u32,
    geometry: &Geometry<f32>,
) {
    let Some(mut color) = property_color(layer, "line-color", context) else {
        return;
    };
    color = color_with_opacity(
        color,
        property_number(layer, "line-opacity", context).unwrap_or(1.0),
    );
    let width = property_number(layer, "line-gap-width", context)
        .unwrap_or(0.0)
        .mul_add(
            2.0,
            property_number(layer, "line-width", context).unwrap_or(1.0),
        );
    let Some(path) = geometry_path(camera, tile, extent, geometry, false) else {
        return;
    };
    let width = width.max(0.0);
    let key = LineKey {
        color: ColorKey::new(color),
        width: width.to_bits(),
    };
    push_tile_path(
        &mut lines
            .entry(key)
            .or_insert_with(|| LineBatch {
                color,
                width,
                paths: Vec::new(),
            })
            .paths,
        tile,
        path,
    );
}

fn property_value(layer: &StyleLayer, name: &str, context: &EvaluationContext) -> Option<Value> {
    layer.property(name).map(|expression| {
        evaluate(expression, context).unwrap_or_else(|error| {
            panic!(
                "layer {} property {name} evaluation failed: {error}",
                layer.id
            )
        })
    })
}

fn property_number(layer: &StyleLayer, name: &str, context: &EvaluationContext) -> Option<f64> {
    property_value(layer, name, context).map(|value| {
        value
            .as_number()
            .expect("type-checked map property must evaluate to number")
    })
}

fn property_string(layer: &StyleLayer, name: &str, context: &EvaluationContext) -> Option<String> {
    let expression = layer.property(name)?;
    match evaluate(expression, context) {
        Ok(Value::Null) | Err(_) => None,
        Ok(value) => Some(value.to_string()),
    }
}

fn property_number_pair(
    layer: &StyleLayer,
    name: &str,
    context: &EvaluationContext,
) -> Option<(f64, f64)> {
    property_value(layer, name, context).map(|value| {
        let values = match value {
            Value::Array(values) => values
                .into_iter()
                .map(|value| {
                    value
                        .as_number()
                        .expect("type-checked map array property must contain numbers")
                })
                .collect::<Vec<_>>(),
            Value::NumberArray(values) => values,
            _ => panic!("type-checked map property {name} must evaluate to a number array"),
        };
        let values: [f64; 2] = values
            .try_into()
            .expect("type-checked map offset must contain exactly two numbers");
        values.into()
    })
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "MapLibre f64 color channels are intentionally converted to f32 channels"
)]
fn property_color(layer: &StyleLayer, name: &str, context: &EvaluationContext) -> Option<Color> {
    property_value(layer, name, context).map(|value| {
        let Value::Color(color) = value else {
            panic!("type-checked map property {name} must evaluate to color");
        };
        Color::new([
            color.r as f32,
            color.g as f32,
            color.b as f32,
            color.a as f32,
        ])
    })
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "MapLibre f64 opacity is clamped before conversion to f32"
)]
fn color_with_opacity(color: Color, opacity: f64) -> Color {
    let [red, green, blue, alpha] = color.components();
    Color::new([red, green, blue, alpha * opacity.clamp(0.0, 1.0) as f32])
}

fn label_origin(
    anchor: (f64, f64),
    offset: (f64, f64),
    width: f64,
    height: f64,
    text_anchor: &str,
) -> (f64, f64) {
    let x = match text_anchor {
        "left" | "top-left" | "bottom-left" => anchor.0 + offset.0,
        "right" | "top-right" | "bottom-right" => anchor.0 + offset.0 - width,
        "center" | "top" | "bottom" => width.mul_add(-0.5, anchor.0 + offset.0),
        other => panic!("unsupported MapLibre text-anchor {other}"),
    };
    let y = match text_anchor {
        "top" | "top-left" | "top-right" => anchor.1 + offset.1,
        "bottom" | "bottom-left" | "bottom-right" => anchor.1 + offset.1 - height,
        "center" | "left" | "right" => height.mul_add(-0.5, anchor.1 + offset.1),
        other => panic!("unsupported MapLibre text-anchor {other}"),
    };
    (x, y)
}

fn geometry_path(
    camera: Camera,
    tile: TileId,
    extent: u32,
    geometry: &Geometry<f32>,
    close: bool,
) -> Option<BezPath> {
    let mut path = BezPath::new();
    let tolerance = camera.tile_units_per_pixel(tile.z, extent) * 2.0;
    match geometry {
        Geometry::LineString(line) => {
            add_line(
                &mut path,
                camera,
                tile,
                extent,
                &line.simplify(tolerance),
                close,
            );
        }
        Geometry::MultiLineString(lines) => {
            for line in &lines.0 {
                add_line(
                    &mut path,
                    camera,
                    tile,
                    extent,
                    &line.simplify(tolerance),
                    close,
                );
            }
        }
        Geometry::Polygon(polygon) => {
            add_polygon(
                &mut path,
                camera,
                tile,
                extent,
                &polygon.simplify(tolerance),
            );
        }
        Geometry::MultiPolygon(polygons) => {
            for polygon in &polygons.0 {
                add_polygon(
                    &mut path,
                    camera,
                    tile,
                    extent,
                    &polygon.simplify(tolerance),
                );
            }
        }
        _ => return None,
    }
    (!path.is_empty()).then_some(path)
}

fn fill_outline_path(
    camera: Camera,
    tile: TileId,
    extent: u32,
    geometry: &Geometry<f32>,
) -> Option<BezPath> {
    let mut path = BezPath::new();
    let tolerance = camera.tile_units_per_pixel(tile.z, extent) * 2.0;
    match geometry {
        Geometry::Polygon(polygon) => {
            add_polygon_outline(
                &mut path,
                camera,
                tile,
                extent,
                &polygon.simplify(tolerance),
            );
        }
        Geometry::MultiPolygon(polygons) => {
            for polygon in &polygons.0 {
                add_polygon_outline(
                    &mut path,
                    camera,
                    tile,
                    extent,
                    &polygon.simplify(tolerance),
                );
            }
        }
        _ => return None,
    }
    (!path.is_empty()).then_some(path)
}

fn camera_render_bounds(camera: Camera) -> Rect {
    Rect::new(
        -f64::from(TILE_OVERSCAN_PIXELS),
        -f64::from(TILE_OVERSCAN_PIXELS),
        f64::from(camera.viewport.width) + f64::from(TILE_OVERSCAN_PIXELS),
        f64::from(camera.viewport.height) + f64::from(TILE_OVERSCAN_PIXELS),
    )
}

fn feature_visible(
    camera: Camera,
    tile: TileId,
    extent: u32,
    geometry: &Geometry<f32>,
    render_bounds: Rect,
) -> bool {
    let Some(bounds) = geometry.bounding_rect() else {
        return true;
    };
    let top_left = camera.tile_point(tile, extent, bounds.min().x, bounds.min().y);
    let bottom_right = camera.tile_point(tile, extent, bounds.max().x, bounds.max().y);
    let feature_bounds = Rect::new(
        top_left.0.min(bottom_right.0),
        top_left.1.min(bottom_right.1),
        top_left.0.max(bottom_right.0),
        top_left.1.max(bottom_right.1),
    );
    feature_bounds.overlaps(render_bounds)
}

fn add_polygon(
    path: &mut BezPath,
    camera: Camera,
    tile: TileId,
    extent: u32,
    polygon: &Polygon<f32>,
) {
    add_line(path, camera, tile, extent, polygon.exterior(), true);
    for interior in polygon.interiors() {
        add_line(path, camera, tile, extent, interior, true);
    }
}

fn add_polygon_outline(
    path: &mut BezPath,
    camera: Camera,
    tile: TileId,
    extent: u32,
    polygon: &Polygon<f32>,
) {
    add_ring_outline(path, camera, tile, extent, polygon.exterior());
    for interior in polygon.interiors() {
        add_ring_outline(path, camera, tile, extent, interior);
    }
}

fn add_ring_outline(
    path: &mut BezPath,
    camera: Camera,
    tile: TileId,
    extent: u32,
    ring: &LineString<f32>,
) {
    let tile_extent = gpu_scalar(extent);
    let mut previous_end = None;
    for segment in ring.0.windows(2) {
        let [start, end] = [segment[0], segment[1]];
        if tile_boundary_segment(start, end, tile_extent) {
            previous_end = None;
            continue;
        }
        if previous_end != Some(start) {
            path.move_to(camera.tile_point(tile, extent, start.x, start.y));
        }
        path.line_to(camera.tile_point(tile, extent, end.x, end.y));
        previous_end = Some(end);
    }
}

const fn tile_boundary_segment(start: Coord<f32>, end: Coord<f32>, extent: f32) -> bool {
    let zero = 0.0_f32.to_bits();
    let extent = extent.to_bits();
    (start.x.to_bits() == zero && end.x.to_bits() == zero)
        || (start.y.to_bits() == zero && end.y.to_bits() == zero)
        || (start.x.to_bits() == extent && end.x.to_bits() == extent)
        || (start.y.to_bits() == extent && end.y.to_bits() == extent)
}

fn add_line(
    path: &mut BezPath,
    camera: Camera,
    tile: TileId,
    extent: u32,
    line: &LineString<f32>,
    close: bool,
) {
    let mut points = line.points();
    let Some(first) = points.next() else {
        return;
    };
    path.move_to(camera.tile_point(tile, extent, first.x(), first.y()));
    for point in points {
        path.line_to(camera.tile_point(tile, extent, point.x(), point.y()));
    }
    if close {
        path.close_path();
    }
}

fn feature_label_anchor(
    camera: Camera,
    tile: TileId,
    extent: u32,
    geometry: &Geometry<f32>,
    placement: &str,
) -> Option<(f64, f64, f64)> {
    let line_placement = match placement {
        "point" => false,
        "line" | "line-center" => true,
        other => panic!("unsupported MapLibre symbol-placement {other}"),
    };
    match geometry {
        Geometry::Point(point) => {
            let (x, y) = camera.tile_point(tile, extent, point.x(), point.y());
            Some((x, y, 0.0))
        }
        Geometry::MultiPoint(points) => points
            .0
            .first()
            .map(|point| camera.tile_point(tile, extent, point.x(), point.y()))
            .map(|(x, y)| (x, y, 0.0)),
        Geometry::LineString(line) => line_label_anchor(camera, tile, extent, line, line_placement),
        Geometry::MultiLineString(lines) => longest_line(&lines.0)
            .and_then(|line| line_label_anchor(camera, tile, extent, line, line_placement)),
        Geometry::Polygon(polygon) => {
            polygon_anchor(camera, tile, extent, polygon).map(|(x, y)| (x, y, 0.0))
        }
        Geometry::MultiPolygon(polygons) => largest_polygon(&polygons.0)
            .and_then(|polygon| polygon_anchor(camera, tile, extent, polygon))
            .map(|(x, y)| (x, y, 0.0)),
        _ => None,
    }
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "the normalized segment fraction is bounded to zero through one before MVT f32 interpolation"
)]
fn line_label_anchor(
    camera: Camera,
    tile: TileId,
    extent: u32,
    line: &LineString<f32>,
    rotate: bool,
) -> Option<(f64, f64, f64)> {
    let points = &line.0;
    if points.len() < 2 {
        return points
            .first()
            .map(|point| camera.tile_point(tile, extent, point.x, point.y))
            .map(|(x, y)| (x, y, 0.0));
    }
    let total = points
        .windows(2)
        .map(|pair| segment_length(pair[0], pair[1]))
        .sum::<f64>();
    let target = total * 0.5;
    let mut traversed = 0.0;
    for pair in points.windows(2) {
        let length = segment_length(pair[0], pair[1]);
        if traversed + length >= target {
            let t = ((target - traversed) / length.max(f64::EPSILON)) as f32;
            let anchor = camera.tile_point(
                tile,
                extent,
                (pair[1].x - pair[0].x).mul_add(t, pair[0].x),
                (pair[1].y - pair[0].y).mul_add(t, pair[0].y),
            );
            let start = camera.tile_point(tile, extent, pair[0].x, pair[0].y);
            let end = camera.tile_point(tile, extent, pair[1].x, pair[1].y);
            let mut angle = if rotate {
                (end.1 - start.1).atan2(end.0 - start.0)
            } else {
                0.0
            };
            if angle > std::f64::consts::FRAC_PI_2 {
                angle -= std::f64::consts::PI;
            } else if angle < -std::f64::consts::FRAC_PI_2 {
                angle += std::f64::consts::PI;
            }
            return Some((anchor.0, anchor.1, angle));
        }
        traversed += length;
    }
    None
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "MVT polygon coordinates are f32 and the centroid average remains in the same bounded tile extent"
)]
fn polygon_anchor(
    camera: Camera,
    tile: TileId,
    extent: u32,
    polygon: &Polygon<f32>,
) -> Option<(f64, f64)> {
    let points = &polygon.exterior().0;
    if points.is_empty() {
        return None;
    }
    let (sum_x, sum_y) = points.iter().fold((0.0_f64, 0.0_f64), |(x, y), point| {
        (x + f64::from(point.x), y + f64::from(point.y))
    });
    let count = f64::from(
        u32::try_from(points.len()).expect("an MVT polygon point count must fit into u32"),
    );
    Some(camera.tile_point(tile, extent, (sum_x / count) as f32, (sum_y / count) as f32))
}

fn longest_line(lines: &[LineString<f32>]) -> Option<&LineString<f32>> {
    lines.iter().max_by(|left, right| {
        line_length(left)
            .partial_cmp(&line_length(right))
            .expect("finite vector-tile geometry length")
    })
}

fn largest_polygon(polygons: &[Polygon<f32>]) -> Option<&Polygon<f32>> {
    polygons
        .iter()
        .max_by_key(|polygon| polygon.exterior().0.len())
}

fn line_length(line: &LineString<f32>) -> f64 {
    line.0
        .windows(2)
        .map(|pair| segment_length(pair[0], pair[1]))
        .sum()
}

fn segment_length(left: Coord<f32>, right: Coord<f32>) -> f64 {
    f64::from((right.x - left.x).hypot(right.y - left.y))
}

#[cfg(test)]
mod chrome_tests {
    use super::*;

    fn camera_at(zoom_span: f64) -> Camera {
        Camera::new(
            Region::new(
                Coordinate::from_degrees(40.7580, -73.9855).expect("valid coordinate"),
                zoom_span,
                zoom_span,
            ),
            Viewport {
                width: 800,
                height: 600,
            },
            0,
            crate::projection::MAX_CAMERA_ZOOM,
        )
    }

    /// The bar must represent a round distance and stay inside its budget,
    /// otherwise it reads as a precise measurement it is not.
    #[test]
    fn the_scale_bar_picks_a_round_distance_that_fits() {
        for span in [0.001, 0.01, 0.1, 1.0, 10.0] {
            let camera = camera_at(span);
            let latitude = camera.region.center.latitude.get();
            let (meters, width) = scale_bar_span(camera, latitude)
                .expect("every reasonable zoom must yield a scale bar");

            assert!(
                SCALE_STEPS.contains(&meters),
                "scale bar must use a round distance, got {meters}"
            );
            assert!(
                width <= SCALE_MAX_WIDTH,
                "scale bar must fit its budget, got {width}"
            );
            let expected = camera.meters_to_pixels(latitude, meters);
            assert!(
                (width - expected).abs() < 1e-6,
                "bar width must match the distance it claims"
            );
        }
    }

    #[test]
    fn scale_distances_read_in_metres_or_kilometres() {
        assert_eq!(format_scale_distance(500.0), "500 m");
        assert_eq!(format_scale_distance(1_000.0), "1 km");
        assert_eq!(format_scale_distance(2_000.0), "2 km");
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use kurbo::{PathEl, Point};
    use num_traits::ToPrimitive as _;
    use waterui_graphics::cherenkov_cpu::Raster;
    use waterui_graphics::{OffscreenRenderer, OffscreenSize};
    use waterui_map::{MapInteractivity, MapVisibility};
    use waterui_url::Url;

    use super::*;

    fn manhattan_region(latitude_delta: f64, longitude_delta: f64) -> Region {
        Region::new(
            Coordinate::from_degrees(40.7580, -73.9855)
                .expect("Manhattan coordinate must be valid"),
            latitude_delta,
            longitude_delta,
        )
    }

    #[test]
    fn cached_scene_transform_matches_live_camera_projection() {
        let viewport = Viewport {
            width: 1_200,
            height: 800,
        };
        let source = Camera::new(manhattan_region(0.030, 0.050), viewport, 0, 22);
        let target_region = Region::new(
            Coordinate::from_degrees(40.7680, -73.9655)
                .expect("shifted Manhattan coordinate must be valid"),
            0.015,
            0.025,
        );
        let target = Camera::new(target_region, viewport, 0, 22);
        let coordinate =
            Coordinate::from_degrees(40.7484, -73.9857).expect("test coordinate must be valid");
        let source_point = source.coordinate_point(coordinate);
        let expected = target.coordinate_point(coordinate);
        let transformed = camera_transform(source, target)
            .expect("different cameras require a reprojection transform")
            * Point::new(source_point.0, source_point.1);

        assert!((transformed.x - expected.0).abs() < 1e-6);
        assert!((transformed.y - expected.1).abs() < 1e-6);
    }

    #[test]
    fn camera_animation_interpolates_zoom_in_log_space() {
        let from = manhattan_region(0.040, 0.090);
        let to = Region::new(from.center, 0.010, 0.010);

        let halfway = interpolate_region(from, to, 0.5);

        assert!((halfway.latitude_delta - 0.020).abs() < 1e-12);
        assert!((halfway.longitude_delta - 0.030).abs() < 1e-12);
    }

    #[test]
    fn camera_animation_takes_the_short_path_across_the_antimeridian() {
        let from = Region::new(
            Coordinate::from_degrees(0.0, 179.0).expect("test coordinate must be valid"),
            1.0,
            1.0,
        );
        let to = Region::new(
            Coordinate::from_degrees(0.0, -179.0).expect("test coordinate must be valid"),
            1.0,
            1.0,
        );

        let halfway = interpolate_region(from, to, 0.5);

        assert!((halfway.center.longitude.get() + 180.0).abs() < 1e-12);
    }

    #[test]
    fn fill_outlines_omit_vector_tile_clip_edges() {
        let camera = Camera::new(
            manhattan_region(0.030, 0.050),
            Viewport {
                width: 1_200,
                height: 800,
            },
            0,
            22,
        );
        let polygon = Polygon::new(
            LineString::from(vec![
                (0.0, 10.0),
                (40.0, 10.0),
                (40.0, 90.0),
                (0.0, 90.0),
                (0.0, 10.0),
            ]),
            Vec::new(),
        );
        let path = fill_outline_path(
            camera,
            TileId {
                z: 14,
                x: 4_824,
                y: 6_160,
            },
            100,
            &Geometry::Polygon(polygon),
        )
        .expect("non-clipped polygon edges must produce an outline");

        assert_eq!(path.elements().len(), 4);
        assert!(matches!(path.elements()[0], PathEl::MoveTo(_)));
        assert!(
            path.elements()[1..]
                .iter()
                .all(|element| matches!(element, PathEl::LineTo(_)))
        );
    }

    #[test]
    fn surface_pan_updates_and_settles_the_camera_without_rebound() {
        let controller =
            MapGestureController::new(&Computed::constant(manhattan_region(0.030, 0.050)));
        let viewport = Viewport {
            width: 1_000,
            height: 500,
        };
        let mut surface_gesture = SurfaceCameraGesture::default();
        let moving = SurfaceGesture::pan(Point::new(100.0, 50.0), true);

        surface_gesture.apply(&controller, moving, viewport);
        let visible = controller.region.snapshot();
        assert!((visible.center.latitude.get() - 40.761).abs() < 1e-9);
        assert!((visible.center.longitude.get() + 73.9905).abs() < 1e-9);
        assert_eq!(
            controller.settled_region.snapshot(),
            manhattan_region(0.030, 0.050)
        );

        surface_gesture.apply(
            &controller,
            SurfaceGesture::pan(Point::new(100.0, 50.0), false),
            viewport,
        );
        assert_eq!(controller.region.snapshot(), visible);
        assert_eq!(controller.settled_region.snapshot(), visible);

        surface_gesture.apply(
            &controller,
            SurfaceGesture::pan(Point::ZERO, false),
            viewport,
        );
        assert_eq!(controller.region.snapshot(), visible);
        assert_eq!(controller.settled_region.snapshot(), visible);
    }

    #[test]
    fn surface_pinch_zooms_around_the_focal_point_and_settles() {
        let controller =
            MapGestureController::new(&Computed::constant(manhattan_region(0.030, 0.050)));
        let viewport = Viewport {
            width: 1_000,
            height: 500,
        };
        let mut surface_gesture = SurfaceCameraGesture::default();
        let pinching = SurfaceGesture {
            pinch_scale: 2.0,
            pinch_center: Some(Point::new(500.0, 250.0)),
            pan_offset: Point::ZERO,
            active: true,
        };

        surface_gesture.apply(&controller, pinching, viewport);
        let visible = controller.region.snapshot();
        assert!((visible.latitude_delta - 0.015).abs() < 1e-9);
        assert!((visible.longitude_delta - 0.025).abs() < 1e-9);
        // The pinch focus sits at the viewport center, so the center holds.
        let origin = manhattan_region(0.030, 0.050);
        assert!((visible.center.latitude.get() - origin.center.latitude.get()).abs() < 1e-9);
        assert!((visible.center.longitude.get() - origin.center.longitude.get()).abs() < 1e-9);

        surface_gesture.apply(
            &controller,
            SurfaceGesture {
                active: false,
                ..pinching
            },
            viewport,
        );
        assert_eq!(controller.settled_region.snapshot(), visible);
    }

    #[test]
    fn network_failure_keeps_map_alive_without_prepared_tiles() {
        let region = manhattan_region(0.030, 0.050);
        let viewport = Viewport {
            width: 1_200,
            height: 800,
        };
        let config = MapConfig {
            region: Computed::constant(region),
            annotations: Computed::constant(Vec::new()),
            style: waterui_map::MapStyle::Standard,
            user_location_visibility: MapVisibility::Hidden,
            user_location: None,
            interactivity: MapInteractivity::ReadOnly,
            compass_visibility: MapVisibility::Hidden,
            scale_visibility: MapVisibility::Hidden,
            status: None,
        };
        let options = MapGpuOptions::new(Url::new("https://tiles.openfreemap.org/styles/positron"));
        let mut map = MapScene::new(config, options);
        {
            let mut state = map.state.borrow_mut();
            state.request = Some(RequestKey { region, viewport });
            state.last_error = Some(String::from("simulated connection timeout"));
        }

        let frame = map.resolve_frame(
            viewport
                .width
                .to_f32()
                .expect("test viewport width must fit into f32"),
            viewport
                .height
                .to_f32()
                .expect("test viewport height must fit into f32"),
        );

        assert_eq!(frame.viewport, viewport);
        assert!(frame.camera.is_none());
        assert!(map.state.borrow().prepared.is_none());
        assert_eq!(
            map.state.borrow().last_error.as_deref(),
            Some("simulated connection timeout")
        );
    }

    #[test]
    fn network_failure_preserves_the_last_prepared_camera() {
        let region = manhattan_region(0.030, 0.050);
        let viewport = Viewport {
            width: 1_200,
            height: 800,
        };
        let style = MapStyle {
            sources: BTreeMap::new(),
            layers: Vec::new(),
        };
        let prepared = PreparedMap {
            chrome: MapChrome {
                compass: MapVisibility::Hidden,
                scale: MapVisibility::Hidden,
            },
            style: style.clone(),
            camera: Camera::new(region, viewport, 0, 22),
            tiles: SourceTiles::default(),
            annotations: Vec::new(),
            location: None,
            painter: MapPainter::default(),
            registrations: Registrations::default(),
            base: None,
        };
        let config = MapConfig {
            region: Computed::constant(region),
            annotations: Computed::constant(Vec::new()),
            style: waterui_map::MapStyle::Standard,
            user_location_visibility: MapVisibility::Hidden,
            user_location: None,
            interactivity: MapInteractivity::ReadOnly,
            compass_visibility: MapVisibility::Hidden,
            scale_visibility: MapVisibility::Hidden,
            status: None,
        };
        let options = MapGpuOptions::new(Url::new("https://tiles.openfreemap.org/styles/positron"));
        let mut map = MapScene::new(config, options);
        {
            let mut state = map.state.borrow_mut();
            state.style = Some(style);
            state.prepared = Some(prepared);
            state.prepared_generation = Some(1);
            state.request = Some(RequestKey { region, viewport });
            state.last_error = Some(String::from("simulated connection timeout"));
        }

        let frame = map.resolve_frame(
            viewport
                .width
                .to_f32()
                .expect("test viewport width must fit into f32"),
            viewport
                .height
                .to_f32()
                .expect("test viewport height must fit into f32"),
        );

        assert!(frame.camera.is_some());
        assert_eq!(frame.prepared_generation, Some(1));
        assert!(map.state.borrow().prepared.is_some());
    }

    #[test]
    fn retry_state_uses_bounded_backoff_only_for_transient_errors() {
        let policy = crate::MapNetworkRetryPolicy::new(
            NonZeroU32::new(3).expect("test retry attempt count must be non-zero"),
            core::time::Duration::from_millis(10),
            core::time::Duration::from_millis(20),
        );
        let transient = MapLoadError::Request {
            url: String::from("https://tiles.openfreemap.org/styles/positron"),
            message: String::from("connection reset"),
        };
        let mut retry = NetworkRetryState::new(policy);

        assert_eq!(
            retry.retry_after(&transient),
            Some(core::time::Duration::from_millis(10))
        );
        assert_eq!(
            retry.retry_after(&transient),
            Some(core::time::Duration::from_millis(20))
        );
        assert_eq!(retry.retry_after(&transient), None);

        let permanent = MapLoadError::Http {
            url: String::from("https://tiles.openfreemap.org/styles/positron"),
            status: 404,
        };
        assert_eq!(NetworkRetryState::new(policy).retry_after(&permanent), None);

        for status in [408, 425, 429, 500, 503, 599] {
            let transient_http = MapLoadError::Http {
                url: String::from("https://tiles.openfreemap.org/styles/positron"),
                status,
            };
            assert_eq!(
                NetworkRetryState::new(policy).retry_after(&transient_http),
                Some(core::time::Duration::from_millis(10))
            );
        }
    }

    /// The frozen base picture replays under a camera transform — the same
    /// path a pan gesture animates — so rendering the prepared map offscreen
    /// exercises the whole port: tile decode, resource registration and the
    /// Cherenkov engine. The recorded content is backend-agnostic; the export
    /// uses the CPU backend because the GPU backend's coverage atlas (4096²
    /// texels, 11,933 cells ≈ 14M texels needed here) cannot hold a
    /// city-scale scene yet — an engine capacity issue, not a content one.
    /// The PNG goes to `/tmp/waterui_map_gpu/` for visual comparison against
    /// the pre-port export.
    #[test]
    #[ignore = "requires network access and a real tile provider"]
    fn cached_camera_pipeline_exports_manhattan() {
        let _ = executor_core::try_init_global_executor(native_executor::NativeExecutor::new());
        let width = 1_600;
        let height = 1_200;
        let region = manhattan_region(0.030, 0.050);
        let options = MapGpuOptions::new(Url::new("https://tiles.openfreemap.org/styles/positron"));
        let mut prepared = pollster::block_on(PreparedMap::load(&options, region, width, height))
            .expect("OpenFreeMap Manhattan scene must load");

        let renderer =
            OffscreenRenderer::<Raster>::cpu().expect("cached map visual requires an engine");
        let size =
            OffscreenSize::try_from_pixels(width, height).expect("visual size must be non-zero");
        let output = renderer
            .render(&mut prepared, size, 1.0)
            .expect("cached map camera replay must render");
        let output_path = std::path::Path::new("/tmp/waterui_map_gpu/cached_camera.png");
        std::fs::create_dir_all(
            output_path
                .parent()
                .expect("cached map output must have a parent"),
        )
        .expect("cached map output directory must be created");
        output
            .save_png(output_path)
            .expect("cached map output must be saved");
    }
}
