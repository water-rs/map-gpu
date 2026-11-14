//! The map as [`GpuContent`]: a retained Cherenkov surface of its own, drawn
//! into the host's content texture.
//!
//! The content has two halves. The UI-thread half ([`Host`]) owns everything
//! the map already owned — style, tiles, caches, label placement, the camera
//! and its gestures — plus an engine and a retained surface on the host's
//! device. Each frame, before the content renders, it brings that surface up
//! to date and renders it. The render-thread half ([`MapContent`]) only
//! presents: it copies the surface's texture into the frame the host hands it.
//! The halves meet in [`Shared`], a mailbox of latest values; nothing blocks
//! on it.
//!
//! The camera is state of the retained recording, not a reason to record
//! again: tiles, rasters and the heatmap are placed by transforms that follow
//! the camera signal, so a pan or zoom moves them in place. Only what a
//! camera move changes the *choice* of — label placement, the pins and chrome,
//! and layers crossing a zoom threshold — records again, each in its own
//! region.

use std::{
    cell::RefCell,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
};

use arc_swap::{ArcSwap, ArcSwapOption};
use cherenkov::{Display, Engine, FrameTime, Surface};
use cherenkov_gpu::{
    Gpu, GpuConfig,
    interop::{SharedDevice, TextureTarget, shader_module},
};
use waterui_graphics::{
    gpu::{Context, Frame, GpuContent, GpuContentView, RedrawHandle, wgpu},
    input::SurfaceInputEvent,
};

use crate::render::{MapScene, Retained};

/// The host's device handles, as `setup` received them.
#[derive(Debug)]
struct Handles {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

/// The content texture's extent and scale, as the last frame saw it.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct View {
    pixels: (u32, u32),
    scale: f32,
}

/// The surface's texture and the extent it was rendered at.
#[derive(Debug)]
struct Output {
    texture: wgpu::Texture,
    pixels: (u32, u32),
}

/// The mailbox between the UI-thread half and the render-thread half.
///
/// Every field is the latest value of one fact, written by one side and read
/// by the other.
#[derive(Debug, Default)]
pub struct Shared {
    /// The host's wake, known once the content is set up.
    redraw: ArcSwapOption<RedrawHandle>,
    /// The host's device, replaced when the host rebuilds it.
    device: ArcSwapOption<Handles>,
    /// What the render side last drew into.
    view: ArcSwap<View>,
    /// What the UI side last rendered.
    output: ArcSwapOption<Output>,
    /// Whether the camera is moving and the next frame is wanted.
    animating: AtomicBool,
}

impl Shared {
    /// Wakes an idle host for another frame.
    pub(crate) fn wake(&self) {
        if let Some(redraw) = &*self.redraw.load() {
            redraw.request_redraw();
        }
    }

    /// The extent of the surface the UI half last rendered.
    #[cfg(test)]
    pub(crate) fn rendered_pixels(&self) -> Option<(u32, u32)> {
        self.output.load().as_ref().map(|output| output.pixels)
    }
}

/// The pipeline that draws the surface's texture into the host's.
struct Blit {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
}

impl Blit {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let module = shader_module(device, "map present", include_str!("blit.wgsl"));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("map present"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("map present"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("map present"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self { pipeline, layout }
    }
}

/// The render-thread half: presents the surface's texture.
struct MapContent {
    shared: Arc<Shared>,
    blit: Option<Blit>,
}

impl GpuContent for MapContent {
    fn setup(&mut self, gpu: &Context<'_>) {
        self.blit = Some(Blit::new(gpu.device(), gpu.format()));
        self.shared
            .redraw
            .store(Some(Arc::new(gpu.redraw().clone())));
        self.shared.device.store(Some(Arc::new(Handles {
            instance: gpu.instance().clone(),
            adapter: gpu.adapter().clone(),
            device: gpu.device().clone(),
            queue: gpu.queue().clone(),
        })));
    }

    fn render(&mut self, frame: &mut Frame<'_>) {
        let view = View {
            pixels: (frame.width, frame.height),
            scale: frame.scale,
        };
        if **self.shared.view.load() != view {
            // The UI half sizes its surface from this, so it needs another
            // frame to catch up.
            self.shared.view.store(Arc::new(view));
            frame.request_redraw();
        }
        let blit = self.blit.as_ref().expect("the map rendered before setup");
        let output = self.shared.output.load();
        let output = output.as_deref().filter(|o| o.pixels == view.pixels);
        if let Some(output) = output {
            assert_eq!(
                output.texture.format(),
                frame.format,
                "the map surface and the content texture must share the working format"
            );
        }
        let mut encoder = frame
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("map present"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("map present"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: frame.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            // Until the UI half has rendered at this extent the frame is clear.
            if let Some(output) = output {
                let source = output
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let bind_group = frame.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("map present"),
                    layout: &blit.layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&source),
                    }],
                });
                pass.set_pipeline(&blit.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        frame.queue.submit([encoder.finish()]);
        if self.shared.animating.load(Ordering::Relaxed) {
            frame.request_redraw();
        }
    }
}

/// The engine and retained surface on the host's current device.
struct Drawing {
    device: Arc<Handles>,
    engine: Engine<Gpu>,
    retained: Retained<Gpu>,
    textures: mpsc::Receiver<wgpu::Texture>,
    texture: Option<wgpu::Texture>,
    view: View,
}

impl Drawing {
    fn new(device: Arc<Handles>, view: View, redraw: Option<Arc<RedrawHandle>>) -> Self {
        let engine = Engine::<Gpu>::new(GpuConfig {
            device: Some(SharedDevice {
                instance: device.instance.clone(),
                adapter: device.adapter.clone(),
                device: device.device.clone(),
                queue: device.queue.clone(),
            }),
            ..GpuConfig::default()
        })
        .expect("the map engine must start on the host's device");
        let (target, textures) = TextureTarget::new(view.pixels);
        let surface: Surface<Gpu> = engine
            .surface(target, move || {
                if let Some(redraw) = &redraw {
                    redraw.request_redraw();
                }
            })
            .expect("the map surface must open on the host's device");
        let texture = textures
            .try_recv()
            .expect("a texture target publishes its texture when its surface opens");
        Self {
            device,
            engine,
            retained: Retained::new(surface),
            textures,
            texture: Some(texture),
            view,
        }
    }
}

/// The UI-thread half: the map's state and, once the host's device is known,
/// its drawing.
struct Host {
    shared: Arc<Shared>,
    scene: Rc<RefCell<MapScene>>,
    drawing: Option<Drawing>,
}

impl Host {
    /// Brings the surface up to date for this frame and renders it.
    fn drive(&mut self) {
        let Some(device) = self.shared.device.load_full() else {
            return;
        };
        let view = **self.shared.view.load();
        if view.pixels.0 == 0 || view.pixels.1 == 0 {
            return;
        }
        let drawing = match self.drawing.take() {
            Some(drawing) if Arc::ptr_eq(&drawing.device, &device) => drawing,
            // A device the host replaced took the old engine's resources with
            // it: forget them, and draw again on the new one.
            previous => {
                if previous.is_some() {
                    self.scene.borrow_mut().engine_replaced();
                }
                Drawing::new(device, view, self.shared.redraw.load_full())
            }
        };
        let drawing = self.drawing.insert(drawing);
        if drawing.view != view {
            if drawing.view.pixels != view.pixels {
                drawing
                    .retained
                    .surface()
                    .resize(view.pixels)
                    .expect("the map surface must resize with its content");
            }
            drawing.view = view;
        }
        drawing
            .retained
            .surface()
            .display(Display {
                scale: f64::from(view.scale),
                headroom: 1.0,
            })
            .expect("the map surface must accept its display scale");
        let animating = drawing.retained.update(
            &drawing.engine,
            &mut self.scene.borrow_mut(),
            view.pixels,
            view.scale,
        );
        drawing
            .engine
            .render(FrameTime::now())
            .expect("the map surface must render");
        while let Ok(texture) = drawing.textures.try_recv() {
            drawing.texture = Some(texture);
        }
        let texture = drawing
            .texture
            .clone()
            .expect("a rendered surface has a texture");
        self.shared.output.store(Some(Arc::new(Output {
            texture,
            pixels: view.pixels,
        })));
        self.shared.animating.store(animating, Ordering::Relaxed);
    }
}

/// `scene` as a view drawn by [`GpuContent`].
pub fn view(scene: MapScene) -> GpuContentView {
    let shared = scene.shared();
    let interactive = scene.wants_input_events();
    let scene = Rc::new(RefCell::new(scene));
    let host = RefCell::new(Host {
        shared: Arc::clone(&shared),
        scene: Rc::clone(&scene),
        drawing: None,
    });
    let view = GpuContentView::new(MapContent { shared, blit: None })
        .on_frame(move || host.borrow_mut().drive());
    if interactive {
        view.on_input(move |event: &SurfaceInputEvent| scene.borrow_mut().input(event))
    } else {
        view
    }
}
