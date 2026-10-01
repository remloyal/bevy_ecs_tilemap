use bevy::{
    camera::primitives::{Aabb, Frustum},
    math::Affine3A,
    platform::collections::{HashMap, HashSet},
    prelude::*,
    render::{
        Extract,
        render_resource::{FilterMode, TextureFormat},
        sync_world::RenderEntity,
    },
};

use crate::anchor::TilemapAnchor;
use crate::prelude::TilemapGridSize;
use crate::prelude::TilemapRenderSettings;
use crate::render::DefaultSampler;
use crate::tiles::AnimatedTile;
use crate::tiles::TilePosOld;
use crate::{
    FrustumCulling,
    map::{
        TilemapId, TilemapSize, TilemapSpacing, TilemapTexture, TilemapTextureSize,
        TilemapTileSize, TilemapType,
    },
    tiles::{TileColor, TileFlip, TilePos, TileTextureIndex, TileVisible},
};

use super::ModifiedImageIds;
use super::chunk::{PackedTileData, TilemapTextureSizes};

#[derive(Component)]
pub struct ChangedInMainWorld;

#[derive(Component)]
pub struct ExtractedTile {
    pub entity: Entity,
    pub position: TilePos,
    pub old_position: TilePosOld,
    pub tile: PackedTileData,
    pub tilemap_id: TilemapId,
}

#[derive(Bundle)]
pub struct ExtractedTileBundle {
    tile: ExtractedTile,
    changed: ChangedInMainWorld,
}

#[derive(Bundle)]
pub struct ExtractedTilemapBundle {
    transform: GlobalTransform,
    tile_size: TilemapTileSize,
    grid_size: TilemapGridSize,
    spacing: TilemapSpacing,
    map_type: TilemapType,
    texture: TilemapTexture,
    map_size: TilemapSize,
    visibility: InheritedVisibility,
    frustum_culling: FrustumCulling,
    render_settings: TilemapRenderSettings,
    changed: ChangedInMainWorld,
    anchor: TilemapAnchor,
}

#[derive(Component)]
pub(crate) struct ExtractedTilemapTexture {
    pub tilemap_id: TilemapId,
    pub tile_size: TilemapTileSize,
    pub texture_size: TilemapTextureSize,
    pub tile_spacing: TilemapSpacing,
    pub tile_count: u32,
    pub texture: TilemapTexture,
    pub filtering: FilterMode,
    pub format: TextureFormat,
}

impl ExtractedTilemapTexture {
    pub fn new(
        tilemap_entity: Entity,
        texture: TilemapTexture,
        tile_size: TilemapTileSize,
        tile_spacing: TilemapSpacing,
        filtering: FilterMode,
        image_assets: &Res<Assets<Image>>,
    ) -> ExtractedTilemapTexture {
        let (tile_count, texture_size, format) = match &texture {
            TilemapTexture::Single(handle) => {
                let image = image_assets.get(handle).expect(
                    "Expected image to have finished loading if \
                    it is being extracted as a texture!",
                );
                let texture_size: TilemapTextureSize = image.size_f32().into();
                let tile_count_x = ((texture_size.x) / (tile_size.x + tile_spacing.x)).floor();
                let tile_count_y = ((texture_size.y) / (tile_size.y + tile_spacing.y)).floor();
                (
                    (tile_count_x * tile_count_y) as u32,
                    texture_size,
                    image.texture_descriptor.format,
                )
            }
            #[cfg(not(feature = "atlas"))]
            TilemapTexture::Vector(handles) => {
                for handle in handles {
                    let image = image_assets.get(handle).expect(
                        "Expected image to have finished loading if \
                        it is being extracted as a texture!",
                    );
                    let this_tile_size: TilemapTileSize = image.size_f32().into();
                    if this_tile_size != tile_size {
                        panic!(
                            "Expected all provided image assets to have size {tile_size:?}, \
                                    but found image with size: {this_tile_size:?}",
                        );
                    }
                }
                let first_format = image_assets
                    .get(handles.first().unwrap())
                    .unwrap()
                    .texture_descriptor
                    .format;

                for handle in handles {
                    let image = image_assets.get(handle).unwrap();
                    if image.texture_descriptor.format != first_format {
                        panic!(
                            "Expected all provided image assets to have a format of: {:?} but found image with format: {:?}",
                            first_format, image.texture_descriptor.format
                        );
                    }
                }

                (handles.len() as u32, tile_size.into(), first_format)
            }
            #[cfg(not(feature = "atlas"))]
            TilemapTexture::TextureContainer(image_handle) => {
                let image = image_assets.get(image_handle).expect(
                    "Expected image to have finished loading if \
                        it is being extracted as a texture!",
                );
                let tile_size: TilemapTileSize = image.size_f32().into();
                (
                    image.texture_descriptor.array_layer_count(),
                    tile_size.into(),
                    image.texture_descriptor.format,
                )
            }
        };

        ExtractedTilemapTexture {
            tilemap_id: TilemapId(tilemap_entity),
            texture,
            tile_size,
            tile_spacing,
            filtering,
            tile_count,
            texture_size,
            format,
        }
    }
}

#[derive(Bundle)]
pub(crate) struct ExtractedTilemapTextureBundle {
    data: ExtractedTilemapTexture,
    changed: ChangedInMainWorld,
}

#[derive(Component, Debug)]
pub struct ExtractedFrustum {
    frustum: Frustum,
}

impl ExtractedFrustum {
    pub fn intersects_obb(&self, aabb: &Aabb, transform_matrix: &Mat4) -> bool {
        self.frustum
            .intersects_obb(aabb, &Affine3A::from_mat4(*transform_matrix), true, false)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn extract(
    mut commands: Commands,
    default_image_settings: Res<DefaultSampler>,
    changed_tiles_query: Extract<
        Query<
            (
                &RenderEntity,
                &TilePos,
                &TilePosOld,
                &TilemapId,
                &TileTextureIndex,
                Option<&TileVisible>,
                Option<&TileFlip>,
                Option<&TileColor>,
                Option<&AnimatedTile>,
            ),
            Or<(
                Changed<TilePos>,
                Changed<TileVisible>,
                Changed<TileTextureIndex>,
                Changed<TileFlip>,
                Changed<TileColor>,
                Changed<AnimatedTile>,
            )>,
        >,
    >,
    tilemap_query: Extract<
        Query<(
            &RenderEntity,
            &GlobalTransform,
            &TilemapTileSize,
            &TilemapSpacing,
            &TilemapGridSize,
            &TilemapType,
            &TilemapTexture,
            &TilemapSize,
            &InheritedVisibility,
            &FrustumCulling,
            &TilemapRenderSettings,
            &TilemapAnchor,
        )>,
    >,
    changed_tilemap_query: Extract<
        Query<
            Entity,
            Or<(
                Added<TilemapType>,
                Changed<TilemapType>,
                Changed<GlobalTransform>,
                Changed<TilemapTexture>,
                Changed<TilemapTileSize>,
                Changed<TilemapSpacing>,
                Changed<TilemapGridSize>,
                Changed<TilemapSize>,
                Changed<InheritedVisibility>,
                Changed<FrustumCulling>,
                Changed<TilemapRenderSettings>,
                Changed<TilemapAnchor>,
            )>,
        >,
    >,
    camera_query: Extract<Query<(&RenderEntity, &Frustum), With<Camera>>>,
    images: Extract<Res<Assets<Image>>>,
    // Main-world tilemaps whose texture component was just added or changed.
    changed_texture_query: Extract<Query<Entity, Changed<TilemapTexture>>>,
    // What texture resolution needs, without the cost of the full 12-tuple
    // `tilemap_query` and with the main-world entity available for the change
    // set above.
    texture_source_query: Extract<
        Query<(
            Entity,
            &RenderEntity,
            &TilemapTileSize,
            &TilemapSpacing,
            &TilemapTexture,
        )>,
    >,
    // Render-world: which tilemaps already have a resolved texture.
    resolved_textures: Query<&ExtractedTilemapTexture>,
    modified_image_ids: Res<ModifiedImageIds>,
    mut texture_sizes: ResMut<TilemapTextureSizes>,
) {
    let mut extracted_tiles = Vec::new();
    let mut extracted_tilemaps = <HashMap<_, _>>::default();
    let mut extracted_tilemap_textures = Vec::new();

    // Maps a tilemap's main-world entity to its render-world entity. Tiles reference
    // their tilemap by main-world entity, and the tile loop below can run tens of
    // thousands of times per frame for a handful of tilemaps, so cache the lookup
    // instead of re-querying the 12-tuple for every single tile.
    let mut tilemap_render_entities: HashMap<Entity, Entity> = <HashMap<_, _>>::default();

    // Extract changed tilemaps *before* the tile loop so the cache above is warm and
    // the tile loop does not have to re-extract a tilemap that is already present.
    for tilemap_entity in changed_tilemap_query.iter() {
        if let Ok(data) = tilemap_query.get(tilemap_entity) {
            let render_entity = data.0.id();
            tilemap_render_entities.insert(tilemap_entity, render_entity);
            extracted_tilemaps.insert(
                render_entity,
                (
                    render_entity,
                    ExtractedTilemapBundle {
                        transform: *data.1,
                        tile_size: *data.2,
                        spacing: *data.3,
                        grid_size: *data.4,
                        map_type: *data.5,
                        texture: data.6.clone(),
                        map_size: *data.7,
                        visibility: *data.8,
                        frustum_culling: *data.9,
                        render_settings: *data.10,
                        changed: ChangedInMainWorld,
                        anchor: *data.11,
                    },
                ),
            );
        }
    }

    // Process all tiles
    for (
        render_entity,
        tile_pos,
        tile_pos_old,
        tilemap_id,
        tile_texture,
        visible,
        flip,
        color,
        animated,
    ) in changed_tiles_query.iter()
    {
        // flipping and rotation packed in bits
        // bit 0 : flip_x
        // bit 1 : flip_y
        // bit 2 : flip_d (anti diagonal)
        let flip = flip.copied().unwrap_or_default();
        let tile_flip_bits = flip.x as i32 | ((flip.y as i32) << 1) | ((flip.d as i32) << 2);

        let mut position = Vec4::new(tile_pos.x as f32, tile_pos.y as f32, 0.0, 0.0);
        let mut texture = Vec4::new(tile_texture.0 as f32, tile_flip_bits as f32, 0.0, 0.0);
        if let Some(animation_data) = animated {
            position.z = animation_data.speed;
            texture.z = animation_data.start as f32;
            texture.w = animation_data.end as f32;
        } else {
            texture.z = tile_texture.0 as f32;
            texture.w = tile_texture.0 as f32;
        }

        let tile = PackedTileData {
            visible: visible.is_none_or(|v| v.0),
            position,
            texture,
            color: color.map_or_else(
                || Color::WHITE.to_linear().to_f32_array(),
                |c| c.0.to_linear().to_f32_array(),
            ),
        };

        let tilemap_render_entity = match tilemap_render_entities.get(&tilemap_id.0) {
            Some(render_entity) => *render_entity,
            None => {
                // First tile we see for this tilemap this frame. Extract it once and
                // remember the mapping for the remaining tiles.
                let data = tilemap_query.get(tilemap_id.0).unwrap();
                let render_entity = data.0.id();
                tilemap_render_entities.insert(tilemap_id.0, render_entity);
                extracted_tilemaps.insert(
                    render_entity,
                    (
                        render_entity,
                        ExtractedTilemapBundle {
                            transform: *data.1,
                            tile_size: *data.2,
                            spacing: *data.3,
                            grid_size: *data.4,
                            map_type: *data.5,
                            texture: data.6.clone(),
                            map_size: *data.7,
                            visibility: *data.8,
                            frustum_culling: *data.9,
                            render_settings: *data.10,
                            changed: ChangedInMainWorld,
                            anchor: *data.11,
                        },
                    ),
                );
                render_entity
            }
        };

        extracted_tiles.push((
            render_entity.id(),
            ExtractedTileBundle {
                tile: ExtractedTile {
                    entity: render_entity.id(),
                    position: *tile_pos,
                    old_position: *tile_pos_old,
                    tile,
                    tilemap_id: TilemapId(tilemap_render_entity),
                },
                changed: ChangedInMainWorld,
            },
        ));
    }

    let extracted_tilemaps: Vec<_> = extracted_tilemaps.drain().map(|(_, val)| val).collect();

    // Extracts tilemap textures, but only where something actually changed.
    //
    // Resolving a texture re-reads every image handle the tilemap's texture holds
    // and re-inserts `ExtractedTilemapTexture`, and a `TilemapTexture::Vector`
    // can hold thousands of handles. Doing that unconditionally meant paying
    // `O(tilemaps * handles)` asset lookups every frame for data that almost
    // never changes.
    let mut changed_textures: HashSet<Entity> = HashSet::default();
    for entity in changed_texture_query.iter() {
        changed_textures.insert(entity);
    }
    // Almost always empty. It is only worth walking a texture's handles to ask
    // whether any of them was modified when something was modified at all.
    let any_image_modified = !modified_image_ids.0.is_empty();

    for (entity, render_entity, tile_size, tile_spacing, texture) in texture_source_query.iter() {
        let needs_resolution = if resolved_textures.get(render_entity.id()).is_err() {
            // No resolved texture yet. The image may still be loading, so this
            // has to keep retrying rather than waiting for the component to
            // change again.
            true
        } else if changed_textures.contains(&entity) {
            true
        } else {
            // Covers hot-reloading an image in place.
            any_image_modified && modified_image_ids.is_texture_modified(texture)
        };

        if !needs_resolution {
            continue;
        }

        if texture.verify_ready(&images) {
            let data = ExtractedTilemapTexture::new(
                render_entity.id(),
                texture.clone(),
                *tile_size,
                *tile_spacing,
                default_image_settings.0.min_filter.into(),
                &images,
            );

            // Chunks created from now on read the size straight from here, and
            // `prepare` replays the change onto the chunks that already exist.
            texture_sizes.record(render_entity.id(), data.texture_size.into());

            extracted_tilemap_textures.push((
                render_entity.id(),
                ExtractedTilemapTextureBundle {
                    data,
                    changed: ChangedInMainWorld,
                },
            ))
        }
    }

    for (render_entity, frustum) in camera_query.iter() {
        commands
            .entity(render_entity.id())
            .insert(ExtractedFrustum { frustum: *frustum });
    }

    commands.insert_batch(extracted_tiles);
    commands.insert_batch(extracted_tilemaps);
    commands.insert_batch(extracted_tilemap_textures);
}
