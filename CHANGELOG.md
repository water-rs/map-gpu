# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0](https://github.com/water-rs/map-gpu/compare/v0.3.0...v0.4.0) - 2026-11-14

### Added

- render the map as GpuContent
- drive map scrolling from the surface scroll phase
- port to the render-target-neutral recording contract (water-rs/waterui#1435) ([#52](https://github.com/water-rs/map-gpu/pull/52))
- *(render)* port the map renderer from Vello to Cherenkov

### Fixed

- *(gpu)* build the shared device from the setup context instance
- *(scene)* implement SceneContent::rebuild_for_engine for the 1406 contract
- repin hydrolysis to 14d5e231, m3 to fcb8b9e8; patch all registry waterkit crates to 123b0581
- *(ci)* pin the toolchain without disarming the MSRV gate ([#42](https://github.com/water-rs/map-gpu/pull/42))

### Other

- *(deps)* point migrated Batch 2 git dependencies to main ([#78](https://github.com/water-rs/map-gpu/pull/78))
- update release-plz rev for Rust 1.100 compatibility
- release from main and retire dev ([#73](https://github.com/water-rs/map-gpu/pull/73))
- restore the Hydrolysis test fonts from the Actions cache
- *(deps)* point migrated Batch 1 git dependencies to main ([#69](https://github.com/water-rs/map-gpu/pull/69))
- name GpuContent and input types by their module paths
- follow waterui dev again
- resolve gpu-allocator against the windows version wgpu-hal uses
- lock the wgpu fork at the revision the framework builds DX12 with
- follow waterui dev's toolchain floor and windows resolution
- read waterui structs through their accessors
- consume the framework from its dev branch
- lock the framework at its rebased nami 0.13 head
- [**breaking**] move to nami 0.13 and the migrated framework
- *(render)* hash style fingerprints with rustc-hash
- *(render)* fingerprint styles and cache labels, heatmap and shaded tiles
- *(render)* record one picture per tile and stop batches at the tile boundary
- pass the release pull request's gate without CI
- [**breaking**] move the patch table to nami 0.12
- require nami 0.12
- *(deps)* pin waterui to dev at 46552dd30 ([#53](https://github.com/water-rs/map-gpu/pull/53))
- install Hydrolysis test fonts before tests
- *(deps)* repin water-rs/waterui to 1e359bca
- *(deps)* repin the engine stack to in-tree Cherenkov at waterui 4dacae48
- *(deps)* pin hydrolysis, hydrolysis-m3 and video-gpu at the repinned heads
- pin cherenkov to 52236561
- install SPIRV-Tools through the shared Linux deps action
- install SPIRV-Tools for cherenkov-gpu's shader build on non-Apple legs
- repin to the cherenkov cutover and restore the mounted test
- repin cherenkov to cb2af643
- fix lint findings in the Manhattan export test
- *(render)* render the Manhattan export on the GPU and CPU backends
- *(render)* restore the layer-wide path merge for opaque paints
- repin cherenkov to 283124ff and waterui to e5801bad
- pin water-rs dependencies to their dev merges ([#48](https://github.com/water-rs/map-gpu/pull/48))
- *(render)* size bump buffers per tile and verify readbacks demand-sized ([#47](https://github.com/water-rs/map-gpu/pull/47))
- repin the vello fork to the glifo canonical-location fix
- repin water-rs dependencies to merged dev heads
- repin water-rs dependencies to unified migration heads
- repin hydrolysis to accessibility import fix
- refresh lockfile for final branch revs
- repin migrated dependency branch heads
- rustfmt on migrated files
- [**breaking**] migrate to nami snapshot API
- consume the 0.5.0 framework wave from crates.io ([#41](https://github.com/water-rs/map-gpu/pull/41))

## [0.2.0](https://github.com/water-rs/map-gpu/compare/v0.1.0...v0.2.0) - 2026-09-11

### Added

- *(graphics)* render scenes on the CPU/GPU split engine where compute is missing
- *(cli)* launcher icons for self-drawn desktop targets
- *(cli)* warn when a dependency needs a permission the app has not enabled
- *(ffi)* [**breaking**] carry user location and load status across the map boundary
- *(map-gpu)* give the map surface an accessibility identity
- *(map)* report load status and draw the compass and scale bar
- *(map-gpu)* [**breaking**] resolve satellite and hybrid styles instead of panicking
- *(map-gpu)* implement the missing MapLibre layer types
- [**breaking**] close the issue #134-#148 audit sweep

### Fixed

- *(ci)* install the same Linux packages for the release preflight
- *(map)* pinch zoom and committed pans on the surface-gesture path
- *(map)* declare the network access the GPU map realization needs
- *(map)* make the GPU map actually reach the screen on Android

### Other

- *(deps)* lock zenwave 0.6.1
- *(deps)* zenwave 0.6.1 with the hyper backend enabled explicitly
- link the test graph to the waterui 0.4 release commit
- *(deps)* waterui-graphics 0.4 (and waterui-text/-testing 0.4 where used)
- depend on the released waterui crates instead of the monorepo dev branch
- update Linux package matrix and add dxc on Windows
- setup standalone crate files, CI workflows, and release-plz
- [**breaking**] depend on the self-drawn component crates directly
- consolidate GPU glue into waterui-graphics helpers
- [**breaking**] ungate Scene2D from the GPU stack and drop its Vello escape hatches
- ship the licence texts in every published crate
- depend on shaderloom directly, and give the icon codegen its own name
- clear the rustdoc warnings
- Format the workspace
- Fix complete Hydrolysis Web build
- *(map-gpu)* assert the projection puts north up and west left
- clear pre-existing lint debt surfaced while checking the GPU fixes
- Use fused math for map label bounds
- Fix workspace CI failures
- Add production GPU vector map fallback
