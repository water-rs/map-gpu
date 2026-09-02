# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0](https://github.com/water-rs/map-gpu/releases/tag/v0.1.0) - 2026-09-02

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

- *(map)* pinch zoom and committed pans on the surface-gesture path
- *(map)* declare the network access the GPU map realization needs
- *(map)* make the GPU map actually reach the screen on Android

### Other

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
