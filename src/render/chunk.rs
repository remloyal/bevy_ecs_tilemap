use std::hash::{Hash, Hasher};

use bevy::{
    asset::RenderAssetUsages,
    mesh::{BaseMeshPipelineKey, Indices, PrimitiveTopology},
    platform::collections::HashMap,
};
use bevy::{camera::primitives::Aabb, math::Mat4};
use bevy::{
    math::{UVec2, UVec3, UVec4, Vec2, Vec3Swizzles, Vec4, Vec4Swizzles},
    prelude::{Component, Entity, GlobalTransform, Mesh},
    render::{
        mesh::{RenderMesh, RenderMeshBufferInfo},
        render_resource::{BufferInitDescriptor, BufferUsages, ShaderType},
        renderer::RenderDevice,
    },
};
use bevy::{
    mesh::MeshVertexBufferLayouts,
    prelude::{InheritedVisibility, Resource, Transform},
};
use bevy::{
    mesh::{MeshVertexAttribute, VertexAttributeValues},
    render::render_resource::Buffer,
};

use crate::prelude::helpers::transform::{chunk_aabb, chunk_index_to_world_space};
use crate::render::extract::ExtractedFrustum;
use crate::{
    FrustumCulling, TilemapGridSize, TilemapTileSize,
    map::{TilemapSize, TilemapTexture, TilemapType},
    tiles::TilePos,
};

use super::RenderChunkSize;

#[derive(Resource, Default, Clone, Debug)]
pub struct RenderChunk2dStorage {
    chunks: HashMap<u32, HashMap<UVec3, RenderChunk2d>>,
    entity_to_chunk_tile: HashMap<Entity, (u32, UVec3, UVec2)>,
}

#[derive(Default, Component, Clone, Copy, Debug)]
pub struct ChunkId(pub UVec3);

impl RenderChunk2dStorage {
    #[allow(clippy::too_many_arguments)]
    pub fn get_or_add(
        &mut self,
        tile_entity: Entity,
        tile_pos: UVec2,
        chunk_entity: Entity,
        position: &UVec4,
        chunk_size: UVec2,
        mesh_type: TilemapType,
        tile_size: TilemapTileSize,
        texture_size: Vec2,
        spacing: Vec2,
        grid_size: TilemapGridSize,
        texture: &TilemapTexture,
        map_size: TilemapSize,
        transform: &GlobalTransform,
        visibility: &InheritedVisibility,
        frustum_culling: &FrustumCulling,
        render_size: RenderChunkSize,
        y_sort: bool,
    ) -> &mut RenderChunk2d {
        let pos = position.xyz();

        self.entity_to_chunk_tile
            .insert(tile_entity, (position.w, pos, tile_pos));

        let chunk_storage = self.chunks.entry(position.w).or_default();

        // A single `entry` lookup resolves both the "already exists" and "needs
        // creating" cases. The previous `contains_key` + `get_mut` pair hashed
        // `pos` twice and walked the bucket list twice for every changed tile.
        // The closure only runs when the chunk is genuinely absent, so the
        // `RenderChunk2d::new` cost stays off the common path.
        chunk_storage.entry(pos).or_insert_with(|| {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            position.hash(&mut hasher);

            RenderChunk2d::new(
                hasher.finish(),
                chunk_entity.to_bits(),
                &pos,
                chunk_size,
                mesh_type,
                tile_size,
                spacing,
                grid_size,
                texture.clone(),
                texture_size,
                map_size,
                *transform,
                visibility.get(),
                **frustum_culling,
                render_size,
                y_sort,
            )
        })
    }

    pub fn get(&self, position: &UVec4) -> Option<&RenderChunk2d> {
        if let Some(chunk_storage) = self.chunks.get(&position.w) {
            return chunk_storage.get(&position.xyz());
        }
        None
    }

    pub fn get_mut(&mut self, position: &UVec4) -> &mut RenderChunk2d {
        let chunk_storage = self.chunks.get_mut(&position.w).unwrap();
        chunk_storage.get_mut(&position.xyz()).unwrap()
    }

    pub fn remove_tile_with_entity(&mut self, entity: Entity) {
        if let Some((chunk, tile_pos)) = self.get_mut_from_entity(entity) {
            chunk.set(&tile_pos.into(), None);
        }

        self.entity_to_chunk_tile.remove(&entity);
    }

    pub fn get_mut_from_entity(&mut self, entity: Entity) -> Option<(&mut RenderChunk2d, UVec2)> {
        let (tilemap_id, chunk_pos, tile_pos) = self.entity_to_chunk_tile.get(&entity)?;

        let chunk_storage = self.chunks.get_mut(tilemap_id)?;
        let chunk = chunk_storage.get_mut(&chunk_pos.xyz())?;
        Some((chunk, *tile_pos))
    }

    pub fn get_chunk_storage(&mut self, position: &UVec4) -> &mut HashMap<UVec3, RenderChunk2d> {
        self.chunks.entry(position.w).or_default()
    }

    pub fn remove(&mut self, position: &UVec4) {
        let chunk_storage = self.get_chunk_storage(position);

        let pos = position.xyz();

        chunk_storage.remove(&pos);
    }

    pub fn count(&self) -> usize {
        self.chunks.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &RenderChunk2d> {
        self.chunks.iter().flat_map(|(_, x)| x.iter().map(|x| x.1))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut RenderChunk2d> {
        self.chunks
            .iter_mut()
            .flat_map(|(_, x)| x.iter_mut().map(|x| x.1))
    }

    pub fn remove_map(&mut self, entity: Entity) {
        self.chunks.remove(&entity.index_u32());
    }
}

/// Resolved texture size, in pixels, for each tilemap that is ready to render.
///
/// The shader turns a tile index into atlas or array UVs using this size, so a
/// chunk cannot be built correctly without it. It is only knowable in the render
/// world, after the image asset has loaded and its dimensions can be read.
///
/// It lives in a resource rather than on the tilemap entity because
/// `ExtractedTilemapBundle` is re-inserted every time the tilemap changes and
/// would reset the component to a value the main world cannot compute. Keeping
/// the value here means it is *only* ever written with a real, resolved size —
/// an absent entry means "not resolved yet", which callers read as `None`. That
/// is deliberately different from storing a zero placeholder, which is
/// indistinguishable from a genuinely zero-sized texture.
#[derive(Resource, Default, Debug)]
pub struct TilemapTextureSizes {
    sizes: HashMap<Entity, Vec2>,
    /// Tilemaps whose size changed since the last [`take_changed`](Self::take_changed).
    changed: Vec<(Entity, Vec2)>,
}

impl TilemapTextureSizes {
    /// Records the resolved size of a tilemap's texture.
    ///
    /// Re-recording an identical size is a no-op. Resolution runs again whenever
    /// a tilemap's texture is re-resolved, which includes cases where nothing
    /// actually changed, and callers rely on the change list to decide whether
    /// the chunks that already exist need updating. Reporting an unchanged size
    /// would make every such resolution walk every chunk to write the same value.
    pub fn record(&mut self, tilemap: Entity, size: Vec2) {
        if self.sizes.get(&tilemap) == Some(&size) {
            return;
        }

        self.sizes.insert(tilemap, size);
        self.changed.push((tilemap, size));
    }

    /// The resolved size for `tilemap`, or `None` if its texture is not ready.
    pub fn get(&self, tilemap: Entity) -> Option<Vec2> {
        self.sizes.get(&tilemap).copied()
    }

    /// Takes the tilemaps whose size changed since the previous call.
    pub fn take_changed(&mut self) -> Vec<(Entity, Vec2)> {
        std::mem::take(&mut self.changed)
    }

    /// Forgets a tilemap, so despawned maps do not accumulate.
    pub fn remove(&mut self, tilemap: Entity) {
        self.sizes.remove(&tilemap);
        self.changed.retain(|(entity, _)| *entity != tilemap);
    }
}

#[cfg(test)]
mod texture_size_tests {
    use super::{RenderChunk2dStorage, TilemapTextureSizes};
    use crate::{
        FrustumCulling, TilemapGridSize, TilemapTileSize,
        map::{TilemapSize, TilemapTexture, TilemapType},
    };
    use bevy::{
        asset::Handle,
        image::Image,
        math::{UVec2, UVec4, Vec2},
        prelude::{Entity, GlobalTransform, InheritedVisibility},
    };

    use crate::render::RenderChunkSize;

    fn tilemap(index: u32) -> Entity {
        Entity::from_raw_u32(index).unwrap()
    }

    /// Mirrors what `prepare`'s tile loop does when it creates a chunk.
    fn add_tile(
        storage: &mut RenderChunk2dStorage,
        tile: Entity,
        tilemap: Entity,
        chunk_x: u32,
        texture_size: Vec2,
    ) -> UVec4 {
        let position = UVec4::new(chunk_x, 0, 0, tilemap.index_u32());
        let render_size = RenderChunkSize::new(UVec2::splat(64));
        storage.get_or_add(
            tile,
            UVec2::ZERO,
            tilemap,
            &position,
            render_size.0,
            TilemapType::Square,
            TilemapTileSize::new(16.0, 16.0),
            texture_size,
            Vec2::ZERO,
            TilemapGridSize::new(16.0, 16.0),
            &TilemapTexture::Single(Handle::<Image>::default()),
            TilemapSize::new(64, 64),
            &GlobalTransform::IDENTITY,
            &InheritedVisibility::default(),
            &FrustumCulling(true),
            render_size,
            false,
        );
        position
    }

    /// Mirrors what `prepare` does when it replays a size change.
    fn refresh_changed_sizes(storage: &mut RenderChunk2dStorage, sizes: &mut TilemapTextureSizes) {
        for (entity, texture_size) in sizes.take_changed() {
            let chunks = storage.get_chunk_storage(&UVec4::new(0, 0, 0, entity.index_u32()));
            for chunk in chunks.values_mut() {
                chunk.texture_size = texture_size;
            }
        }
    }

    /// The common case: chunks are built from this value, so it has to survive.
    #[test]
    fn records_and_reads_back_a_size() {
        let mut sizes = TilemapTextureSizes::default();
        assert_eq!(sizes.get(tilemap(1)), None);

        sizes.record(tilemap(1), Vec2::new(64.0, 32.0));

        assert_eq!(sizes.get(tilemap(1)), Some(Vec2::new(64.0, 32.0)));
    }

    /// Sizes are per-tilemap, and an unresolved neighbour must stay `None` rather
    /// than borrowing the resolved one's size.
    #[test]
    fn sizes_are_tracked_per_tilemap() {
        let mut sizes = TilemapTextureSizes::default();
        sizes.record(tilemap(1), Vec2::new(64.0, 64.0));

        assert_eq!(sizes.get(tilemap(2)), None);
    }

    /// Re-resolving to the same size must not be reported, or every resolution
    /// would walk every chunk of the tilemap to rewrite an identical value.
    #[test]
    fn re_recording_the_same_size_reports_nothing() {
        let mut sizes = TilemapTextureSizes::default();
        sizes.record(tilemap(1), Vec2::new(64.0, 64.0));
        sizes.take_changed();

        sizes.record(tilemap(1), Vec2::new(64.0, 64.0));

        assert!(sizes.take_changed().is_empty());
    }

    /// A genuine change has to be reported exactly once, since it is what tells
    /// `prepare` that already-built chunks are stale.
    #[test]
    fn changing_a_size_is_reported_once_then_drained() {
        let mut sizes = TilemapTextureSizes::default();
        sizes.record(tilemap(1), Vec2::new(64.0, 64.0));
        sizes.take_changed();

        sizes.record(tilemap(1), Vec2::new(128.0, 64.0));

        assert_eq!(
            sizes.take_changed(),
            vec![(tilemap(1), Vec2::new(128.0, 64.0))]
        );
        assert!(sizes.take_changed().is_empty());
        assert_eq!(sizes.get(tilemap(1)), Some(Vec2::new(128.0, 64.0)));
    }

    /// Tilemaps come and go (streaming spawns and despawns them), so forgetting
    /// one must clear both the value and any pending change for it.
    #[test]
    fn remove_forgets_value_and_pending_change() {
        let mut sizes = TilemapTextureSizes::default();
        sizes.record(tilemap(1), Vec2::new(64.0, 64.0));

        sizes.remove(tilemap(1));

        assert_eq!(sizes.get(tilemap(1)), None);
        assert!(sizes.take_changed().is_empty());
    }

    /// This is the sequence the refactor had to get right.
    ///
    /// A chunk built before its texture finished loading has no size to be given,
    /// and the shader divides by it (`0.5 / texture_size.x` in
    /// `shaders/common.wesl`), so leaving one at zero is a real rendering fault,
    /// not a cosmetic one. Correctness used to come from re-extracting every
    /// texture and rewriting every chunk every frame; now the resolution has to
    /// explicitly reach chunks that already exist.
    ///
    /// Drives the real storage types, and needs no GPU device.
    #[test]
    fn chunk_built_before_its_texture_resolves_is_fixed_when_it_does() {
        let mut sizes = TilemapTextureSizes::default();
        let mut storage = RenderChunk2dStorage::default();
        let map = tilemap(1);

        // Texture not loaded yet: there is nothing to give the chunk.
        assert_eq!(sizes.get(map), None);
        let early = add_tile(
            &mut storage,
            tilemap(2),
            map,
            0,
            sizes.get(map).unwrap_or(Vec2::ZERO),
        );
        assert_eq!(storage.get(&early).unwrap().texture_size, Vec2::ZERO);

        // The image loads and `extract` records the resolved size.
        sizes.record(map, Vec2::new(128.0, 64.0));
        refresh_changed_sizes(&mut storage, &mut sizes);

        // The chunk that already existed has to have been corrected.
        assert_eq!(
            storage.get(&early).unwrap().texture_size,
            Vec2::new(128.0, 64.0)
        );

        // A chunk created afterwards reads the size directly, so it never
        // depends on the refresh having run.
        let late = add_tile(
            &mut storage,
            tilemap(3),
            map,
            1,
            sizes.get(map).unwrap_or(Vec2::ZERO),
        );
        assert_eq!(
            storage.get(&late).unwrap().texture_size,
            Vec2::new(128.0, 64.0)
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PackedTileData {
    pub visible: bool,
    pub position: Vec4,
    pub texture: Vec4,
    pub color: [f32; 4],
}

#[derive(Clone, Debug)]
pub struct RenderChunk2d {
    pub id: u64,
    pub tilemap_id: u64,
    /// The index of the chunk. It is equivalent to the position of the chunk in "chunk
    /// coordinates".
    index: UVec3,
    /// The position of this chunk, in world space,
    position: Vec2,
    /// Size of the chunk, in tiles.
    pub size_in_tiles: UVec2,
    /// [`TilemapSize`] of the map this chunk belongs to.
    pub map_size: TilemapSize,
    /// [`TilemapType`] of the map this chunk belongs to.
    map_type: TilemapType,
    /// The grid size of the map this chunk belongs to.
    pub grid_size: TilemapGridSize,
    /// The tile size of the map this chunk belongs to.
    pub tile_size: TilemapTileSize,
    /// The [`Aabb`] of this chunk, based on the map type, grid size, and tile size. It is not
    /// transformed by the `global_transform` or [`local_transform`]
    aabb: Aabb,
    local_transform: Transform,
    /// The [`GlobalTransform`] of this chunk, stored as a [`Transform`].
    global_transform: Transform,
    /// The product of the local and global transforms.
    transform: Transform,
    /// The matrix computed from this chunk's `transform`.
    transform_matrix: Mat4,
    pub spacing: Vec2,
    pub tiles: Vec<Option<PackedTileData>>,
    pub texture: TilemapTexture,
    pub texture_size: Vec2,
    pub mesh: Mesh,
    pub render_mesh: Option<RenderMesh>,
    pub vertex_buffer: Option<Buffer>,
    pub index_buffer: Option<Buffer>,
    pub dirty_mesh: bool,
    pub visible: bool,
    pub frustum_culling: bool,
    pub render_size: RenderChunkSize,
    pub y_sort: bool,
    /// Number of visible tiles the index buffer currently uploaded to the GPU
    /// was built for.
    ///
    /// The index pattern is a pure function of that count, so while the count is
    /// unchanged the uploaded buffer is still byte-for-byte correct and does not
    /// need rebuilding or re-uploading.
    indexed_visible_tiles: Option<usize>,
}

/// Take the mesh's `id` vertex vector out, cleared and ready to be refilled.
///
/// Filling the mesh's *own* allocation in place is what keeps a rebuild from
/// reallocating: after the first build the vector already holds a previous
/// frame's worth of vertices, so its capacity is exactly what the next rebuild
/// needs and `reserve` is a no-op.
///
/// An earlier revision instead swapped the mesh's buffer with a separate
/// `scratch_*` field. That kept the previous frame's vertices alive alongside
/// the new ones, doubling steady-state vertex memory for no gain — on the
/// 1280x1280 benchmark, an extra ~330 MB, since every byte uploaded to the GPU
/// was also retained on the CPU.
fn take_attribute(mesh: &mut Mesh, id: MeshVertexAttribute, size: usize) -> Vec<[f32; 4]> {
    let mut values = match mesh.attribute_mut(id) {
        Some(VertexAttributeValues::Float32x4(existing)) => std::mem::take(existing),
        // Attribute not present yet: the first build has nothing to reuse.
        _ => Vec::new(),
    };

    values.clear();
    values.reserve(size);
    values
}

/// Hand a refilled attribute vector back to the mesh.
fn put_attribute(mesh: &mut Mesh, id: MeshVertexAttribute, values: Vec<[f32; 4]>) {
    mesh.insert_attribute(id, VertexAttributeValues::Float32x4(values));
}

impl RenderChunk2d {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: u64,
        tilemap_id: u64,
        index: &UVec3,
        size_in_tiles: UVec2,
        map_type: TilemapType,
        tile_size: TilemapTileSize,
        spacing: Vec2,
        grid_size: TilemapGridSize,
        texture: TilemapTexture,
        texture_size: Vec2,
        map_size: TilemapSize,
        global_transform: GlobalTransform,
        visible: bool,
        frustum_culling: bool,
        render_size: RenderChunkSize,
        y_sort: bool,
    ) -> Self {
        let position = chunk_index_to_world_space(index.xy(), size_in_tiles, &grid_size, &map_type);
        let local_transform = Transform::from_translation(position.extend(0.0));
        let global_transform: Transform = global_transform.into();
        let transform = local_transform * global_transform;
        let transform_matrix = transform.to_matrix();
        let aabb = chunk_aabb(size_in_tiles, &grid_size, &tile_size, &map_type);
        Self {
            dirty_mesh: true,
            render_mesh: None,
            id,
            index: *index,
            position,
            size_in_tiles,
            map_size,
            map_type,
            grid_size,
            tile_size,
            aabb,
            local_transform,
            global_transform,
            transform,
            transform_matrix,
            mesh: Mesh::new(
                bevy::render::render_resource::PrimitiveTopology::TriangleList,
                RenderAssetUsages::default(),
            ),
            vertex_buffer: None,
            index_buffer: None,
            spacing,
            texture_size,
            texture,
            tilemap_id,
            tiles: vec![None; (size_in_tiles.x * size_in_tiles.y) as usize],
            visible,
            frustum_culling,
            render_size,
            y_sort,
            indexed_visible_tiles: None,
        }
    }

    pub fn get(&self, tile_pos: &TilePos) -> &Option<PackedTileData> {
        &self.tiles[tile_pos.to_index(&self.size_in_tiles.into())]
    }

    pub fn get_mut(&mut self, tile_pos: &TilePos) -> &mut Option<PackedTileData> {
        self.dirty_mesh = true;
        &mut self.tiles[tile_pos.to_index(&self.size_in_tiles.into())]
    }

    pub fn set(&mut self, tile_pos: &TilePos, tile: Option<PackedTileData>) {
        let index = tile_pos.to_index(&self.size_in_tiles.into());

        // Bevy reports a component as changed whenever it is written through
        // `DerefMut`, even if the value is identical. A system that re-assigns
        // the same texture index or color every frame would otherwise dirty this
        // chunk on every frame and pay a full vertex regeneration plus a ~12 MB
        // upload for a 256x256 chunk, with nothing on screen changing.
        if self.tiles[index] == tile {
            return;
        }

        self.tiles[index] = tile;
        self.dirty_mesh = true;
    }

    pub fn get_index(&self) -> UVec3 {
        self.index
    }

    pub fn get_map_type(&self) -> TilemapType {
        self.map_type
    }

    pub fn get_transform(&self) -> Transform {
        self.transform
    }

    pub fn get_transform_matrix(&self) -> Mat4 {
        self.transform_matrix
    }

    pub fn intersects_frustum(&self, frustum: &ExtractedFrustum) -> bool {
        frustum.intersects_obb(&self.aabb, &self.transform_matrix)
    }

    pub fn update_geometry(
        &mut self,
        global_transform: Transform,
        grid_size: TilemapGridSize,
        tile_size: TilemapTileSize,
        map_type: TilemapType,
    ) {
        let mut dirty_local_transform = false;

        if self.grid_size != grid_size || self.tile_size != tile_size || self.map_type != map_type {
            self.grid_size = grid_size;
            self.map_type = map_type;
            self.tile_size = tile_size;

            self.position = chunk_index_to_world_space(
                self.index.xy(),
                self.size_in_tiles,
                &self.grid_size,
                &self.map_type,
            );

            self.local_transform = Transform::from_translation(self.position.extend(0.0));
            dirty_local_transform = true;

            self.aabb = chunk_aabb(
                self.size_in_tiles,
                &self.grid_size,
                &self.tile_size,
                &self.map_type,
            );
        }

        let mut dirty_global_transform = false;
        if self.global_transform != global_transform {
            self.global_transform = global_transform;
            dirty_global_transform = true;
        }

        if dirty_local_transform || dirty_global_transform {
            self.transform = global_transform * self.local_transform;
            self.transform_matrix = self.transform.to_matrix();
        }
    }

    pub fn prepare(
        &mut self,
        device: &RenderDevice,
        mesh_vertex_buffer_layouts: &mut MeshVertexBufferLayouts,
    ) {
        if self.dirty_mesh {
            let size = ((self.size_in_tiles.x * self.size_in_tiles.y) * 4) as usize;

            // Fill the mesh's own attribute allocations. Taking them out and
            // putting them back keeps a single live copy per attribute, so a
            // rebuild costs no allocation but also does not retain a second
            // copy of the geometry.
            let mut positions =
                take_attribute(&mut self.mesh, crate::render::ATTRIBUTE_POSITION, size);
            let mut textures =
                take_attribute(&mut self.mesh, crate::render::ATTRIBUTE_TEXTURE, size);
            let mut colors = take_attribute(&mut self.mesh, crate::render::ATTRIBUTE_COLOR, size);

            let mut visible_tiles: usize = 0;

            // Convert tile into mesh data.
            for tile in self.tiles.iter().filter_map(|x| x.as_ref()) {
                if !tile.visible {
                    continue;
                }
                visible_tiles += 1;

                let position: [f32; 4] = tile.position.to_array();
                positions.extend([
                    // X, Y
                    position,
                    // X, Y + 1
                    //[tile_pos.x, tile_pos.y + 1.0, animation_speed],
                    position,
                    // X + 1, Y + 1
                    //[tile_pos.x + 1.0, tile_pos.y + 1.0, animation_speed],
                    position,
                    // X + 1, Y
                    //[tile_pos.x + 1.0, tile_pos.y, animation_speed],
                    position,
                ]);

                colors.extend(std::iter::repeat_n(tile.color, 4));

                // flipping and rotation packed in bits
                // bit 0 : flip_x
                // bit 1 : flip_y
                // bit 2 : flip_d (anti diagonal)

                // let tile_flip_bits =
                //     tile.flip_x as i32 | (tile.flip_y as i32) << 1 | (tile.flip_d as i32) << 2;

                //let texture: [f32; 4] = tile.texture.xyxx().into();
                let texture: [f32; 4] = tile.texture.to_array();
                textures.extend([texture, texture, texture, texture]);
            }

            put_attribute(&mut self.mesh, crate::render::ATTRIBUTE_POSITION, positions);
            put_attribute(&mut self.mesh, crate::render::ATTRIBUTE_TEXTURE, textures);
            put_attribute(&mut self.mesh, crate::render::ATTRIBUTE_COLOR, colors);

            // The index pattern is a pure function of the number of visible tiles,
            // so while that count is unchanged the indices already in the mesh and
            // the buffer already on the GPU stay valid and are left alone.
            let rebuild_indices = self.indexed_visible_tiles != Some(visible_tiles);
            if rebuild_indices {
                let mut indices: Vec<u32> = Vec::with_capacity(visible_tiles * 6);
                for i in (0..(visible_tiles as u32) * 4).step_by(4) {
                    indices.extend_from_slice(&[i, i + 2, i + 1, i, i + 3, i + 2]);
                }
                self.mesh.insert_indices(Indices::U32(indices));
                self.indexed_visible_tiles = Some(visible_tiles);
            }

            let vertex_buffer_data = self.mesh.create_packed_vertex_buffer_data();
            let vertex_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
                usage: BufferUsages::VERTEX,
                label: Some("Mesh Vertex Buffer"),
                contents: &vertex_buffer_data,
            });

            if rebuild_indices {
                let index_buffer = device.create_buffer_with_data(&BufferInitDescriptor {
                    usage: BufferUsages::INDEX,
                    contents: self.mesh.get_index_buffer_bytes().unwrap(),
                    label: Some("Mesh Index Buffer"),
                });
                self.index_buffer = Some(index_buffer);
            }

            let buffer_info = RenderMeshBufferInfo::Indexed {
                count: self.mesh.indices().unwrap().len() as u32,
                index_format: self.mesh.indices().unwrap().into(),
            };

            let mesh_vertex_buffer_layout = self
                .mesh
                .get_mesh_vertex_buffer_layout(mesh_vertex_buffer_layouts);
            self.render_mesh = Some(RenderMesh {
                vertex_count: self.mesh.count_vertices() as u32,
                aabb_center: Default::default(),
                buffer_info,
                layout: mesh_vertex_buffer_layout,
                key_bits: BaseMeshPipelineKey::from_primitive_topology_and_strip_index(
                    PrimitiveTopology::TriangleList,
                    None,
                ),
            });
            self.vertex_buffer = Some(vertex_buffer);
            self.dirty_mesh = false;
        }
    }
}

// Used to transfer info to the GPU for tile building.
#[derive(Debug, Default, Copy, Component, Clone, ShaderType)]
pub struct TilemapUniformData {
    pub texture_size: Vec2,
    pub tile_size: Vec2,
    pub grid_size: Vec2,
    pub spacing: Vec2,
    pub chunk_pos: Vec2,
    pub map_size: Vec2,
}

impl From<&RenderChunk2d> for TilemapUniformData {
    fn from(chunk: &RenderChunk2d) -> Self {
        let chunk_ix: Vec2 = chunk.index.xy().as_vec2();
        let chunk_size: Vec2 = chunk.size_in_tiles.as_vec2();
        let map_size: Vec2 = chunk.map_size.into();
        let tile_size: Vec2 = chunk.tile_size.into();
        Self {
            texture_size: chunk.texture_size,
            tile_size,
            grid_size: chunk.grid_size.into(),
            spacing: chunk.spacing,
            chunk_pos: chunk_ix * chunk_size,
            map_size: map_size * tile_size,
        }
    }
}

impl From<&mut RenderChunk2d> for TilemapUniformData {
    fn from(chunk: &mut RenderChunk2d) -> Self {
        let chunk_pos: Vec2 = chunk.index.xy().as_vec2();
        let chunk_size: Vec2 = chunk.size_in_tiles.as_vec2();
        let map_size: Vec2 = chunk.map_size.into();
        let tile_size: Vec2 = chunk.tile_size.into();
        Self {
            texture_size: chunk.texture_size,
            tile_size,
            grid_size: chunk.grid_size.into(),
            spacing: chunk.spacing,
            chunk_pos: chunk_pos * chunk_size,
            map_size: map_size * tile_size,
        }
    }
}
