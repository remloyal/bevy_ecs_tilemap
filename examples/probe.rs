//! Deterministic probe: change exactly `CHANGED_PER_FRAME` tiles per frame, then
//! report the median/p95 frame time over a fixed number of frames.
//!
//! Self-reported and median-based on purpose. Sampling `LogDiagnosticsPlugin`
//! output gave run-to-run spreads of 18-35ms on identical binaries, which is
//! wide enough to swallow any optimisation being tested. Edit
//! `CHANGED_PER_FRAME` to vary the load; 0 is the static baseline.
use bevy::{app::AppExit, prelude::*, window::PresentMode};
use bevy_ecs_tilemap::prelude::*;
use rand::{RngExt, rng};

mod helpers;

/// Tiles changed per frame. 0 = static baseline.
const CHANGED_PER_FRAME: usize = 1000;
const MAP: u32 = 320;

/// Frames discarded before sampling, so shaders and buffers can settle.
const WARMUP_FRAMES: usize = 180;
/// Frames measured before the app exits.
const SAMPLE_FRAMES: usize = 900;

#[derive(Resource)]
struct Sampler {
    frame: usize,
    samples: Vec<f32>,
}

fn startup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn(Camera2d);
    let texture_handle: Handle<Image> = asset_server.load("tiles.png");
    let map_size = TilemapSize { x: MAP, y: MAP };
    let mut tile_storage = TileStorage::empty(map_size);
    let tilemap_entity = commands.spawn_empty().id();

    fill_tilemap(
        TileTextureIndex(0),
        map_size,
        TilemapId(tilemap_entity),
        &mut commands,
        &mut tile_storage,
    );

    let tile_size = TilemapTileSize { x: 16.0, y: 16.0 };
    commands.entity(tilemap_entity).insert(TilemapBundle {
        grid_size: tile_size.into(),
        map_type: TilemapType::default(),
        size: map_size,
        storage: tile_storage,
        texture: TilemapTexture::Single(texture_handle),
        tile_size,
        ..Default::default()
    });
}

/// Changes a fixed, rotating slice of tiles so the work per frame is constant.
/// Uses a single `iter_mut` pass (not `nth()`, which would be O(n) per lookup and
/// dominate the measurement).
fn churn(mut query: Query<&mut TileTextureIndex>) {
    let mut r = rng();
    let mut i = 0usize;
    for mut t in query.iter_mut() {
        if i < CHANGED_PER_FRAME {
            t.0 = r.random_range(0..6);
            i += 1;
        } else {
            break;
        }
    }
}

/// Records frame times after warmup, then prints robust statistics and exits.
///
/// Sampling `LogDiagnosticsPlugin` instead was the source of the earlier
/// misleading numbers: it reports a window average only about once a second, so
/// each run produced a dozen samples that swung by ~17ms.
fn sample(time: Res<Time>, mut sampler: ResMut<Sampler>, mut exit: MessageWriter<AppExit>) {
    if sampler.frame < WARMUP_FRAMES {
        sampler.frame += 1;
        return;
    }
    if sampler.samples.len() >= SAMPLE_FRAMES {
        let mut sorted = sampler.samples.clone();
        sorted.sort_by(f32::total_cmp);
        let n = sorted.len();
        println!(
            "RESULT changed={} median={:.2}ms p95={:.2}ms mean={:.2}ms n={}",
            CHANGED_PER_FRAME,
            sorted[n / 2],
            sorted[(n as f64 * 0.95) as usize],
            sorted.iter().sum::<f32>() / n as f32,
            n,
        );
        exit.write(AppExit::Success);
        return;
    }
    sampler.samples.push(time.delta_secs() * 1000.0);
}

fn main() {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: format!("probe: {CHANGED_PER_FRAME} tiles/frame"),
                        present_mode: PresentMode::AutoNoVsync,
                        ..Default::default()
                    }),
                    ..default()
                })
                .set(ImagePlugin::default_nearest()),
        )
        .add_plugins(TilemapPlugin)
        .insert_resource(Sampler {
            frame: 0,
            samples: Vec::with_capacity(SAMPLE_FRAMES),
        })
        .add_systems(Startup, startup)
        .add_systems(Update, (churn, sample))
        .run();
}
