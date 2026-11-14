// The light of a galaxy as a volume to turn (see Volume): rendered by ray
// marching on the graphics card (three.js, WebGL2), each ray through the
// box of the volume adding up the light of its voxels, each in the colour
// of its brightness (the colour map in use), as a glowing gas. A drag
// turns it about its centre, the wheel zooms; z, the axis of the galaxy,
// is up.
import { useElementSize } from '@mantine/hooks';
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { GalaxyVolume } from '../../api/queries';
import { rampColour, rampStops, useColourMap } from '../../components/colours';

// The most steps of a ray through the box
const STEPS = 320;

const VERTEX = /* glsl */ `
uniform vec3 eye;
out vec3 vOrigin;
out vec3 vDirection;
void main() {
  vOrigin = eye;
  vDirection = position - eye;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const FRAGMENT = /* glsl */ `
precision highp float;
precision highp sampler3D;
in vec3 vOrigin;
in vec3 vDirection;
out vec4 colour;
uniform sampler3D volume;
uniform sampler2D ramp;
uniform vec3 box;
uniform float steps;
uniform float gain;

// The distances along a ray where it enters and leaves the box
vec2 hits(vec3 origin, vec3 direction) {
  vec3 inverse = 1.0 / direction;
  vec3 t0 = (-0.5 * box - origin) * inverse;
  vec3 t1 = (0.5 * box - origin) * inverse;
  vec3 near = min(t0, t1);
  vec3 far = max(t0, t1);
  return vec2(max(max(near.x, near.y), near.z), min(min(far.x, far.y), far.z));
}

void main() {
  vec3 direction = normalize(vDirection);
  vec2 span = hits(vOrigin, direction);
  if (span.x > span.y) discard;
  span.x = max(span.x, 0.0);
  float dt = length(box) / steps;
  vec3 p = vOrigin + span.x * direction;
  // (the light of the voxels along the ray, each in the colour of its
  // brightness, added up; faint light fades out rather than darkens)
  vec3 sum = vec3(0.0);
  for (float t = span.x; t < span.y; t += dt) {
    float d = texture(volume, p / box + 0.5).r;
    sum += texture(ramp, vec2(0.3 + 0.7 * d, 0.5)).rgb * d * gain * dt;
    p += direction * dt;
  }
  vec3 shown = vec3(1.0) - exp(-sum);
  float alpha = max(max(shown.r, shown.g), shown.b);
  if (alpha < 0.002) discard;
  colour = vec4(shown / alpha, alpha);
}
`;

export default function Volume3D({ volume }: { volume: GalaxyVolume }) {
  const { ref: box, width, height } = useElementSize();
  const holder = useRef<HTMLDivElement>(null);
  const scene = useRef<{ renderer: THREE.WebGLRenderer; camera: THREE.PerspectiveCamera; controls: OrbitControls; material: THREE.ShaderMaterial; mesh: THREE.Mesh; draw: () => void } | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const colourMap = useColourMap();

  // (the renderer, camera and controls, once)
  useEffect(() => {
    const element = holder.current;
    if (!element) return;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch (error) {
      setFailed(error instanceof Error ? error.message : String(error));
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    element.appendChild(renderer.domElement);
    const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
    camera.up.set(0, 0, 1);
    camera.position.set(0, -1.7, 0.85);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = false;
    const material = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader: VERTEX,
      fragmentShader: FRAGMENT,
      transparent: true,
      side: THREE.BackSide,
      uniforms: {
        volume: { value: null },
        ramp: { value: null },
        box: { value: new THREE.Vector3(1, 1, 1) },
        eye: { value: new THREE.Vector3() },
        steps: { value: STEPS },
        gain: { value: 24 },
      },
    });
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material);
    const world = new THREE.Scene();
    world.add(mesh);
    const draw = () => {
      material.uniforms.eye.value.copy(camera.position);
      renderer.render(world, camera);
    };
    controls.addEventListener('change', draw);
    scene.current = { renderer, camera, controls, material, mesh, draw };
    return () => {
      controls.dispose();
      material.dispose();
      mesh.geometry.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      scene.current = null;
    };
  }, []);

  // (the volume, as a texture of bytes of the square root of its light, of
  // its peak; the box in the proportions of its voxels)
  useEffect(() => {
    const s = scene.current;
    if (!s) return;
    const [nz, ny, nx] = volume.shape;
    let peak = 0;
    for (const v of volume.values) if (v > peak) peak = v;
    const bytes = new Uint8Array(volume.values.length);
    for (let i = 0; i < bytes.length; i++) bytes[i] = peak > 0 ? Math.round(255 * Math.sqrt(Math.max(0, volume.values[i]) / peak)) : 0;
    const texture = new THREE.Data3DTexture(bytes, nx, ny, nz);
    texture.format = THREE.RedFormat;
    texture.type = THREE.UnsignedByteType;
    texture.minFilter = texture.magFilter = THREE.LinearFilter;
    texture.unpackAlignment = 1;
    texture.needsUpdate = true;
    const size = new THREE.Vector3(nx * volume.step, ny * volume.step, Math.max(nz, 1) * volume.depthStep);
    size.divideScalar(Math.max(size.x, size.y, size.z));
    // (the gain of the light: the brightest column along z, face-on, about
    // white, however thin the galaxy is in its box)
    let column = 0;
    for (let i = 0; i < nx * ny; i++) {
      let sum = 0;
      for (let z = 0; z < nz; z++) sum += bytes[z * nx * ny + i];
      column = Math.max(column, sum / 255);
    }
    const depth = column * (size.z / Math.max(nz, 1));
    s.material.uniforms.gain.value = depth > 0 ? 3 / depth : 24;
    s.material.uniforms.volume.value?.dispose();
    s.material.uniforms.volume.value = texture;
    s.material.uniforms.box.value.copy(size);
    s.mesh.scale.copy(size);
    s.draw();
  }, [volume]);

  // (the colours of the colour map in use)
  useEffect(() => {
    const s = scene.current;
    if (!s) return;
    const stops = rampStops(false);
    const data = new Uint8Array(256 * 4);
    for (let i = 0; i < 256; i++) {
      const [r, g, b] = rampColour(stops, i / 255);
      data.set([r, g, b, 255], i * 4);
    }
    const ramp = new THREE.DataTexture(data, 256, 1, THREE.RGBAFormat);
    ramp.minFilter = ramp.magFilter = THREE.LinearFilter;
    ramp.needsUpdate = true;
    s.material.uniforms.ramp.value?.dispose();
    s.material.uniforms.ramp.value = ramp;
    s.draw();
  }, [colourMap]);

  // (as large as its place)
  useEffect(() => {
    const s = scene.current;
    if (!s || width === 0 || height === 0) return;
    s.renderer.setSize(width, height);
    s.camera.aspect = width / height;
    s.camera.updateProjectionMatrix();
    s.draw();
  }, [width, height]);

  return (
    <div className="gx-volume" ref={box}>
      <div className="gx-volume-canvas" ref={holder} />
      {failed ? <p className="gx-error gx-view-error">The 3d view needs WebGL2: {failed}</p> : <span className="gx-volume-hint">drag: turn · wheel: zoom</span>}
    </div>
  );
}
