#![allow(dead_code)]

//! Bevy ECS Tilemap plugin is a ECS driven tilemap rendering library. It's designed to be fast and highly customizable. Each tile is considered a unique entity and all tiles are stored in the game world.
//!
//!
//! ## Features
//! - A tile per entity.
//! - Fast rendering using a chunked approach.
//! - Layers and sparse tile maps.
//! - GPU powered animations.
//! - Isometric and Hexagonal tile maps.
//! - Initial support for Tiled file exports.
//! - Support for isometric and hexagon rendering.
//! - Built in animation support  – see [`animation` example](https://github.com/StarArawn/bevy_ecs_tilemap/blob/main/examples/animation.rs).
//! - Texture array support.
//! - Can `Anchor` tilemap like a sprite.

use bevy::{
    ecs::schedule::IntoScheduleConfigs,
    prelude::{
        Bundle, Changed, Component, Deref, First, GlobalTransform, InheritedVisibility, Plugin,
        Query, Reflect, ReflectComponent, SystemSet, Transform, ViewVisibility, Visibility,
    },
    render::sync_world::SyncToRenderWorld,
    scene::{Scene, bsn},
    time::TimeSystems,
};

#[cfg(feature = "render")]
use render::material::MaterialTilemapHandle;

use anchor::TilemapAnchor;
use map::{
    TilemapGridSize, TilemapSize, TilemapSpacing, TilemapTexture, TilemapTextureSize,
    TilemapTileSize, TilemapType,
};
use prelude::{TilemapId, TilemapRenderSettings};
#[cfg(feature = "render")]
use render::material::{MaterialTilemap, StandardTilemapMaterial};
use tiles::{
    AnimatedTile, TileColor, TileFlip, TilePos, TilePosOld, TileStorage, TileTextureIndex,
    TileVisible,
};

#[cfg(all(not(feature = "atlas"), feature = "render"))]
use bevy::render::{ExtractSchedule, RenderApp};

pub mod anchor;
/// A module that allows pre-loading of atlases into array textures.
#[cfg(all(not(feature = "atlas"), feature = "render"))]
mod array_texture_preload;
/// A module which provides helper functions.
pub mod helpers;
/// A module which contains tilemap components.
pub mod map;
#[cfg(feature = "render")]
pub(crate) mod render;
/// A module which contains tile components.
pub mod tiles;

/// A bevy tilemap plugin. This must be included in order for everything to be rendered.
/// But is not necessary if you are running without a renderer.
pub struct TilemapPlugin;

impl Plugin for TilemapPlugin {
    fn build(&self, app: &mut bevy::prelude::App) {
        #[cfg(feature = "render")]
        app.add_plugins(render::TilemapRenderingPlugin);

        app.add_systems(First, update_changed_tile_positions.in_set(TilemapFirstSet));

        #[cfg(all(not(feature = "atlas"), feature = "render"))]
        {
            app.insert_resource(array_texture_preload::ArrayTextureLoader::default());
            let render_app = app.sub_app_mut(RenderApp);
            render_app.add_systems(ExtractSchedule, array_texture_preload::extract);
        }

        app.register_type::<FrustumCulling>()
            .register_type::<TilemapId>()
            .register_type::<TilemapSize>()
            .register_type::<TilemapTexture>()
            .register_type::<TilemapTileSize>()
            .register_type::<TilemapGridSize>()
            .register_type::<TilemapSpacing>()
            .register_type::<TilemapTextureSize>()
            .register_type::<TilemapType>()
            .register_type::<TilemapAnchor>()
            .register_type::<TilePos>()
            .register_type::<TileTextureIndex>()
            .register_type::<TileColor>()
            .register_type::<TileVisible>()
            .register_type::<TileFlip>()
            .register_type::<TileStorage>()
            .register_type::<TilePosOld>()
            .register_type::<AnimatedTile>()
            .configure_sets(First, TilemapFirstSet.after(TimeSystems));
    }
}

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct TilemapFirstSet;

#[derive(Component, Reflect, Debug, Clone, Copy, Deref)]
#[reflect(Component)]
pub struct FrustumCulling(pub bool);

impl Default for FrustumCulling {
    /// By default, `FrustumCulling` is `true`.
    fn default() -> Self {
        FrustumCulling(true)
    }
}

#[cfg(feature = "render")]
pub type TilemapBundle = MaterialTilemapBundle<StandardTilemapMaterial>;

/// Props for [`tilemap_scene`]. Mirrors the fields of [`TilemapBundle`].
#[cfg(feature = "render")]
#[derive(Default, Debug, Clone)]
pub struct TilemapSceneProps<M: MaterialTilemap = StandardTilemapMaterial> {
    pub grid_size: TilemapGridSize,
    pub map_type: TilemapType,
    pub size: TilemapSize,
    pub spacing: TilemapSpacing,
    pub texture: TilemapTexture,
    pub tile_size: TilemapTileSize,
    pub storage: TileStorage,
    pub transform: Transform,
    pub anchor: TilemapAnchor,
    pub render_settings: TilemapRenderSettings,
    pub material: MaterialTilemapHandle<M>,
    pub visibility: Visibility,
    pub frustum_culling: FrustumCulling,
}

/// Builds a [BSN](https://docs.rs/bevy_scene/latest/bevy_scene/) scene that spawns a tilemap.
///
/// This is the scene-based counterpart to [`TilemapBundle`]. It is purely additive —
/// spawning the bundle directly keeps working unchanged.
///
/// The three enum components (`TilemapType`, `TilemapAnchor`, `TilemapTexture`) are not
/// included, because `bsn!` cannot take an arbitrary expression for an enum; pass them in
/// your own `bsn!` block instead:
///
/// ```no_run
/// # use bevy::prelude::*;
/// # use bevy_ecs_tilemap::prelude::*;
/// fn setup(mut commands: Commands) {
///     commands.spawn_scene((
///         tilemap_scene(TilemapSceneProps {
///             tile_size: TilemapTileSize { x: 16.0, y: 16.0 },
///             ..default()
///         }),
///         bsn! {
///             TilemapType::Hexagon(HexCoordSystem::Column)
///             TilemapAnchor::Center
///         },
///     ));
/// }
/// ```
#[cfg(feature = "render")]
pub fn tilemap_scene<M: MaterialTilemap>(props: TilemapSceneProps<M>) -> impl Scene {
    // Inside `bsn!` a bare identifier inserts that variable as a component.
    let storage = props.storage;
    let transform = props.transform;
    let visibility = props.visibility;
    let frustum_culling = props.frustum_culling;

    bsn! {
        TilemapGridSize { x: {props.grid_size.x}, y: {props.grid_size.y} }
        TilemapSize { x: {props.size.x}, y: {props.size.y} }
        TilemapTileSize { x: {props.tile_size.x}, y: {props.tile_size.y} }
        TilemapSpacing { x: {props.spacing.x}, y: {props.spacing.y} }
        TilemapRenderSettings {
            render_chunk_size: {props.render_settings.render_chunk_size},
            y_sort: {props.render_settings.y_sort},
        }
        storage
        transform
        visibility
        frustum_culling
    }
}

#[cfg(feature = "render")]
/// The default tilemap bundle. All of the components within are required.
#[derive(Bundle, Debug, Default, Clone)]
pub struct MaterialTilemapBundle<M: MaterialTilemap> {
    pub grid_size: TilemapGridSize,
    pub map_type: TilemapType,
    pub size: TilemapSize,
    pub spacing: TilemapSpacing,
    pub storage: TileStorage,
    pub texture: TilemapTexture,
    pub tile_size: TilemapTileSize,
    pub transform: Transform,
    pub global_transform: GlobalTransform,
    pub render_settings: TilemapRenderSettings,
    /// User indication of whether an entity is visible
    pub visibility: Visibility,
    /// Algorithmically-computed indication of whether an entity is visible and should be extracted
    /// for rendering
    pub inherited_visibility: InheritedVisibility,
    pub view_visibility: ViewVisibility,
    /// User indication of whether tilemap should be frustum culled.
    pub frustum_culling: FrustumCulling,
    pub material: MaterialTilemapHandle<M>,
    pub sync: SyncToRenderWorld,
    pub anchor: TilemapAnchor,
}

#[cfg(not(feature = "render"))]
/// The default tilemap bundle. All of the components within are required.
#[derive(Bundle, Debug, Default, Clone)]
pub struct StandardTilemapBundle {
    pub grid_size: TilemapGridSize,
    pub map_type: TilemapType,
    pub size: TilemapSize,
    pub spacing: TilemapSpacing,
    pub storage: TileStorage,
    pub texture: TilemapTexture,
    pub tile_size: TilemapTileSize,
    pub transform: Transform,
    pub global_transform: GlobalTransform,
    pub render_settings: TilemapRenderSettings,
    /// User indication of whether an entity is visible
    pub visibility: Visibility,
    /// Algorithmically-computed indication of whether an entity is visible and should be extracted
    /// for rendering
    pub inherited_visibility: InheritedVisibility,
    pub view_visibility: ViewVisibility,
    /// User indication of whether tilemap should be frustum culled.
    pub frustum_culling: FrustumCulling,
    pub sync: SyncToRenderWorld,
}

/// A module which exports commonly used dependencies.
pub mod prelude {
    #[cfg(feature = "render")]
    pub use crate::MaterialTilemapBundle;
    #[cfg(feature = "render")]
    pub use crate::TilemapBundle;
    pub use crate::TilemapPlugin;
    pub use crate::anchor::TilemapAnchor;
    #[cfg(all(not(feature = "atlas"), feature = "render"))]
    pub use crate::array_texture_preload::*;
    pub use crate::helpers;
    pub use crate::helpers::filling::*;
    pub use crate::helpers::geometry::*;
    pub use crate::helpers::transform::*;
    pub use crate::map::*;
    #[cfg(feature = "render")]
    pub use crate::render::material::MaterialTilemap;
    #[cfg(feature = "render")]
    pub use crate::render::material::MaterialTilemapHandle;
    #[cfg(feature = "render")]
    pub use crate::render::material::MaterialTilemapKey;
    #[cfg(feature = "render")]
    pub use crate::render::material::MaterialTilemapPlugin;
    #[cfg(feature = "render")]
    pub use crate::render::material::StandardTilemapMaterial;
    pub use crate::tiles::*;
    #[cfg(feature = "render")]
    pub use crate::{TilemapSceneProps, tilemap_scene};
}

/// Updates old tile positions with the new values from the last frame.
fn update_changed_tile_positions(mut query: Query<(&TilePos, &mut TilePosOld), Changed<TilePos>>) {
    for (tile_pos, mut tile_pos_old) in query.iter_mut() {
        tile_pos_old.0 = *tile_pos;
    }
}
