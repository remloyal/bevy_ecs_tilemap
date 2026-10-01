use std::marker::PhantomData;

use crate::anchor::TilemapAnchor;
use crate::map::{
    TilemapId, TilemapSize, TilemapSpacing, TilemapTexture, TilemapTileSize, TilemapType,
};
use crate::prelude::TilemapRenderSettings;
use crate::render::extract::ExtractedFrustum;
use crate::{FrustumCulling, prelude::TilemapGridSize, render::RenderChunkSize};
use bevy::prelude::{Added, Changed, InheritedVisibility, Or, Resource, Transform};
use bevy::render::sync_world::TemporaryRenderEntity;
use bevy::{log::trace, mesh::MeshVertexBufferLayouts};
use bevy::{
    math::{Mat4, UVec4},
    platform::collections::HashMap,
    prelude::{Commands, Component, Entity, GlobalTransform, Query, Res, ResMut, Vec2},
    render::{
        render_resource::{DynamicUniformBuffer, ShaderType},
        renderer::{RenderDevice, RenderQueue},
    },
};

use super::extract::ChangedInMainWorld;
use super::{
    DynamicUniformIndex,
    chunk::{
        ChunkId, PackedTileData, RenderChunk2dStorage, TilemapTextureSizes, TilemapUniformData,
    },
    extract::ExtractedTile,
};
use super::{RemovedMapEntity, RemovedTileEntity};

#[derive(Resource, Default)]
pub struct MeshUniformResource(pub DynamicUniformBuffer<MeshUniform>);

#[derive(Resource, Default)]
pub struct TilemapUniformResource(pub DynamicUniformBuffer<TilemapUniformData>);

#[derive(ShaderType, Component, Clone)]
pub struct MeshUniform {
    pub transform: Mat4,
}

/// The render state of one tilemap, snapshotted once per frame.
///
/// Every changed tile needs its tilemap's parameters, and reading them straight
/// from [`prepare`]'s query means fetching a 13-component item. The tile loop can
/// run tens of thousands of times per frame for a handful of tilemaps, so the
/// snapshot is taken once per tilemap and then indexed by entity. This mirrors
/// the caching already done on the extract side.
struct TilemapRenderParams<'a> {
    transform: &'a GlobalTransform,
    tile_size: TilemapTileSize,
    texture_size: Vec2,
    spacing: Vec2,
    grid_size: TilemapGridSize,
    map_type: TilemapType,
    texture: &'a TilemapTexture,
    map_size: TilemapSize,
    visibility: &'a InheritedVisibility,
    frustum_culling: &'a FrustumCulling,
    chunk_size: RenderChunkSize,
    y_sort: bool,
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn prepare(
    mut commands: Commands,
    mut chunk_storage: ResMut<RenderChunk2dStorage>,
    mut mesh_uniforms: ResMut<MeshUniformResource>,
    mut tilemap_uniforms: ResMut<TilemapUniformResource>,
    extracted_tiles: Query<
        &ExtractedTile,
        Or<(Added<ChangedInMainWorld>, Changed<ChangedInMainWorld>)>,
    >,
    extracted_tilemaps: Query<
        (
            Entity,
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
        ),
        Or<(Added<ChangedInMainWorld>, Changed<ChangedInMainWorld>)>,
    >,
    extracted_frustum_query: Query<&ExtractedFrustum>,
    // Resolved texture sizes. Read for every chunk that gets built, and drained
    // to refresh the chunks that already exist when a size changes.
    mut texture_sizes: ResMut<TilemapTextureSizes>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut mesh_vertex_buffer_layouts: ResMut<MeshVertexBufferLayouts>,
) {
    // Snapshot each tilemap's render parameters once. Previously the loop below
    // did `extracted_tilemaps.get(...)` per changed tile, re-walking the
    // archetype and re-fetching all 13 components every time.
    //
    // The tilemaps iterated here are exactly those that changed this frame, so
    // this is also what keeps the later `extracted_tilemaps` loop correct: a
    // tilemap that did not change contributes no chunks to update.
    let mut tilemap_params: HashMap<Entity, TilemapRenderParams<'_>> = HashMap::default();
    for (
        entity,
        transform,
        tile_size,
        spacing,
        grid_size,
        map_type,
        texture,
        map_size,
        visibility,
        frustum_culling,
        tilemap_render_settings,
        _anchor,
    ) in extracted_tilemaps.iter()
    {
        tilemap_params.insert(
            entity,
            TilemapRenderParams {
                transform,
                tile_size: *tile_size,
                // A tilemap whose texture has not loaded yet has no entry, and
                // its chunks are built with a zero size. That is corrected when
                // the texture resolves, which reports a change below.
                texture_size: texture_sizes.get(entity).unwrap_or(Vec2::ZERO),
                spacing: (*spacing).into(),
                grid_size: *grid_size,
                map_type: *map_type,
                texture,
                map_size: *map_size,
                visibility,
                frustum_culling,
                chunk_size: RenderChunkSize(tilemap_render_settings.render_chunk_size),
                y_sort: tilemap_render_settings.y_sort,
            },
        );
    }

    for tile in extracted_tiles.iter() {
        // First if the tile position has changed remove the tile from the old location.
        if tile.position != tile.old_position.0 {
            chunk_storage.remove_tile_with_entity(tile.entity);
        }

        let params = tilemap_params
            .get(&tile.tilemap_id.0)
            .expect("a changed tile's tilemap must have been extracted this frame");
        let chunk_size = params.chunk_size;
        let chunk_index = chunk_size.map_tile_to_chunk(&tile.position);

        let chunk_data = UVec4::new(
            chunk_index.x,
            chunk_index.y,
            params.transform.translation().z as u32,
            tile.tilemap_id.0.index_u32(),
        );

        let in_chunk_tile_index = chunk_size.map_tile_to_chunk_tile(&tile.position, &chunk_index);
        let chunk = chunk_storage.get_or_add(
            tile.entity,
            in_chunk_tile_index,
            tile.tilemap_id.0,
            &chunk_data,
            *chunk_size,
            params.map_type,
            params.tile_size,
            params.texture_size,
            params.spacing,
            params.grid_size,
            params.texture,
            params.map_size,
            params.transform,
            params.visibility,
            params.frustum_culling,
            chunk_size,
            params.y_sort,
        );
        // `in_chunk_tile_index` was already computed above to place the tile in
        // this chunk. Reuse it for the packed vertex position instead of running
        // the same divide/subtract a second time for every changed tile.
        let in_chunk_tile_pos = in_chunk_tile_index.as_vec2();
        chunk.set(
            &in_chunk_tile_index.into(),
            Some(PackedTileData {
                position: in_chunk_tile_pos
                    .extend(tile.tile.position.z)
                    .extend(tile.tile.position.w),
                ..tile.tile
            }),
        );
    }

    // Copies transform changes from tilemap to chunks.
    for (
        entity,
        global_transform,
        tile_size,
        spacing,
        grid_size,
        map_type,
        texture,
        map_size,
        visibility,
        frustum_culling,
        _,
        anchor,
    ) in extracted_tilemaps.iter()
    {
        let texture_size = texture_sizes.get(entity).unwrap_or(Vec2::ZERO);
        let chunks = chunk_storage.get_chunk_storage(&UVec4::new(0, 0, 0, entity.index_u32()));
        for chunk in chunks.values_mut() {
            chunk.texture = texture.clone();
            chunk.map_size = *map_size;
            chunk.texture_size = texture_size;
            chunk.spacing = (*spacing).into();
            chunk.visible = visibility.get();
            chunk.frustum_culling = **frustum_culling;
            let anchor_offset: Vec2 = anchor.as_offset(map_size, grid_size, tile_size, map_type);
            // The following code that merely adds a vector would be faster and
            // work in most usecases.
            //
            // ```
            // let mut transform: Transform = (*global_transform).into();
            // transform.translation += anchor_offset.extend(0.0);
            // ```
            //
            // But multiplying by the transform is more general and allows the
            // user to rotate and scale their map transforms.
            let transform =
                *global_transform * Transform::from_translation(anchor_offset.extend(0.0));
            chunk.update_geometry(transform.into(), *grid_size, *tile_size, *map_type);
        }
    }

    // Refresh chunks whose tilemap texture resolved or changed size.
    //
    // Chunks are built with whatever size was known at the time, so a chunk made
    // before its texture finished loading holds a zero size. `extract` reports
    // every size it resolves or changes here, and this replays that onto the
    // chunks that already exist. Chunks created later read the current size
    // directly, so they never need this loop.
    for (entity, texture_size) in texture_sizes.take_changed() {
        let chunks = chunk_storage.get_chunk_storage(&UVec4::new(0, 0, 0, entity.index_u32()));
        for chunk in chunks.values_mut() {
            chunk.texture_size = texture_size;
        }
    }

    mesh_uniforms.0.clear();
    tilemap_uniforms.0.clear();

    // Resolve the camera frustums once up front. Testing a chunk against them
    // used to restart this query's iteration for every chunk in the storage, so
    // the cost was `chunks * cameras` archetype walks per frame.
    let frustums: Vec<&ExtractedFrustum> = extracted_frustum_query.iter().collect();

    for chunk in chunk_storage.iter_mut() {
        if !chunk.visible {
            trace!("Visibility culled chunk: {:?}", chunk.get_index());
            continue;
        }

        if chunk.frustum_culling
            && !frustums
                .iter()
                .any(|frustum| chunk.intersects_frustum(frustum))
        {
            trace!("Frustum culled chunk: {:?}", chunk.get_index());
            continue;
        }

        chunk.prepare(&render_device, &mut mesh_vertex_buffer_layouts);

        let chunk_uniform: TilemapUniformData = chunk.into();

        commands.spawn((
            chunk.texture.clone(),
            chunk.get_transform(),
            ChunkId(chunk.get_index()),
            chunk.get_map_type(),
            TilemapId(Entity::from_bits(chunk.tilemap_id)),
            DynamicUniformIndex::<MeshUniform> {
                index: mesh_uniforms.0.push(&MeshUniform {
                    transform: chunk.get_transform_matrix(),
                }),
                marker: PhantomData,
            },
            DynamicUniformIndex::<TilemapUniformData> {
                index: tilemap_uniforms.0.push(&chunk_uniform),
                marker: PhantomData,
            },
            TemporaryRenderEntity::default(),
        ));
    }

    mesh_uniforms.0.write_buffer(&render_device, &render_queue);
    tilemap_uniforms
        .0
        .write_buffer(&render_device, &render_queue);
}

pub fn prepare_removal(
    mut chunk_storage: ResMut<RenderChunk2dStorage>,
    mut texture_sizes: ResMut<TilemapTextureSizes>,
    removed_tiles: Query<&RemovedTileEntity>,
    removed_maps: Query<&RemovedMapEntity>,
) {
    for removed_tile in removed_tiles.iter() {
        chunk_storage.remove_tile_with_entity(removed_tile.0.id())
    }

    for removed_map in removed_maps.iter() {
        let tilemap = removed_map.0.id();
        chunk_storage.remove_map(tilemap);
        // Otherwise a streaming app that spawns and despawns tilemaps would grow
        // this map without bound.
        texture_sizes.remove(tilemap);
    }
}
