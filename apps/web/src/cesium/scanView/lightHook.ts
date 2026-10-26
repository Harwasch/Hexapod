/** Research format LL01: 48 B/splat, three RGBA32UI streams. See light_bake.py. */
export const LIGHT_STREAMS = ["hexapodLight0", "hexapodLight1", "hexapodLight2"] as const;
export const LIGHT_SOURCES = ["splatLight0", "splatLight1", "splatLight2"] as const;
export const LIGHT_MAGIC = 0x31304c4c;

/** Exact source order -> PlayCanvas Morton order; reject truncated or foreign sidecars. */
export function lightStreamData(buffer: ArrayBuffer, order: Uint32Array): Uint32Array[] {
  if (buffer.byteLength !== order.length * 48) throw new Error("Lighting count mismatch");
  const input = new Uint32Array(buffer);
  const streams = Array.from({ length: 3 }, () => new Uint32Array(order.length * 4));
  for (let i = 0; i < order.length; i++) {
    const source = order[i];
    if (source === undefined || source >= order.length || input[source * 12 + 10] !== LIGHT_MAGIC)
      throw new Error("Invalid LL01 lighting sidecar");
    for (let s = 0; s < 3; s++)
      streams[s]?.set(input.subarray(source * 12 + s * 4, source * 12 + s * 4 + 4), i * 4);
  }
  return streams;
}

const EMPTY = `void modifySplatCenter(inout vec3 center) {}
void modifySplatRotationScale(vec3 a, vec3 b, inout vec4 r, inout vec3 s) {}
void modifySplatColor(vec3 center, inout vec4 color) {}`;
/** Copies once into the work buffer. Missing / non-scan tiles explicitly write zeros. */
export function lightCopyGlsl(base: string | undefined, bound: boolean): string {
  const code = (base ?? EMPTY).replace(/\bmodifySplatColor\b/g, "lightPreviousCopyColor");
  return `${code}
void modifySplatColor(vec3 center, inout vec4 color) {
  writeHexapodLight0(${bound ? "loadSplatLight0()" : "uvec4(0u)"});
  writeHexapodLight1(${bound ? "loadSplatLight1()" : "uvec4(0u)"});
  writeHexapodLight2(${bound ? "loadSplatLight2()" : "uvec4(0u)"});
  lightPreviousCopyColor(center, color);
}`;
}

/** Replaces captured RGB in linear light; never uses the old affine look-of-black trick.
 * Research arms require the Ground truth look and static geometry; alpha stays measured.
 */
export const LIGHT_HOOK = `
uniform vec4 uLightSun;
uniform vec4 uLightParams; // day bin, B, cloud strength, seconds
uniform highp sampler2D uLightCloud;
void modifySplatCenter(inout vec3 center) {}
void modifySplatRotationScale(vec3 a, vec3 b, inout vec4 r, inout vec3 s) {}
vec4 lightBytes(uint word) {
  return vec4(float(word & 255u), float((word >> 8u) & 255u),
              float((word >> 16u) & 255u), float(word >> 24u)) / 255.0;
}
vec3 lightLinear(vec3 c) {
  return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}
vec3 lightSrgb(vec3 c) {
  return mix(c * 12.92, 1.055 * pow(max(c, vec3(0.0)), vec3(1.0/2.4)) - 0.055,
             step(vec3(0.0031308), c));
}
float lightVisibility(uvec4 a, uvec4 b, uvec4 c, int bin) {
  int word = bin / 4;
  uint packed = word < 2 ? a[word + 2] : (word < 6 ? b[word - 2] : c[word - 6]);
  return float((packed >> uint((bin % 4) * 8)) & 255u) / 255.0;
}
void modifySplatColor(vec3 center, inout vec4 color) {
  uvec4 a = loadHexapodLight0();
  uvec4 b = loadHexapodLight1();
  uvec4 c = loadHexapodLight2();
  if (c.z != 825248844u || (a.x >> 24u) == 0u) return;
  vec4 material = lightBytes(a.x);
  vec4 surface = lightBytes(a.y);
  vec3 normal = normalize((surface.xyz * 255.0 - 128.0) / 127.0);
  float visibility = 1.0;
  float sky = 1.0;
  float cloud = 1.0;
  if (uLightParams.y > 0.5) {
    int bin = int(floor(clamp(uLightParams.x, 0.0, 31.0)));
    visibility = mix(lightVisibility(a,b,c,bin), lightVisibility(a,b,c,min(bin+1,31)), fract(uLightParams.x));
    sky = surface.w;
    // World ENU metres, wind towards east/north, unaffected by camera or tile origin.
    float noise = textureLod(uLightCloud, center.xy / 5.0 - vec2(0.035,0.012)*uLightParams.w, 0.0).r;
    cloud = 1.0 - uLightParams.z * smoothstep(0.30,0.70,noise);
  }
  float direct = max(dot(normal, uLightSun.xyz), 0.0) * uLightSun.w * visibility * cloud;
  vec3 illumination = vec3(1.3,1.23,1.1) * direct + vec3(0.32,0.37,0.45) * sky;
  color.rgb = lightSrgb(lightLinear(material.rgb) * illumination);
}
`;
