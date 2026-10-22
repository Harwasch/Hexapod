/**
 * Runs the cell views' one colour code (lib/cellLook.ts `CELL_LOOK_GLSL`) on a real GPU and
 * compares every output with its TypeScript twin (`cellViewColor`): the same inputs must give
 * the same colours and the same hatch, dot and stripe cues, or a renderer would draw one thing
 * while the tests and the key reason about another. The driver for one test of
 * e2e/confidenceViewScene.spec.ts; nothing imports it, so it never reaches the bundle.
 *
 * Each case -- a mode, a cell's two bytes, a colour, a point, a cell size and a threshold -- is
 * one pixel: the inputs ride in a float texture, `hexapodCellLook` writes the pixel, and the
 * pixels are read back as bytes. The points avoid the patterns' exact edges, where a float on
 * the GPU and a double here may round to either side.
 */

import { NEWER_STRIPE_M, CELL_LOOK_GLSL, cellViewColor, type CellViewMode } from "@/lib/cellLook";

export interface LookAgreement {
  cases: number;
  /** The largest difference of any channel, 0 to 1. */
  maxError: number;
  /** The case of the largest difference, and both colours. */
  worst: { input: unknown; gpu: number[]; reference: number[] } | null;
}

interface Case {
  mode: CellViewMode;
  texel: [number, number];
  color: [number, number, number, number];
  p: [number, number];
  cell: number;
  weakBelow: number;
}

const UNSEEN = 16;

function cases(): Case[] {
  const out: Case[] = [];
  const colors: [number, number, number, number][] = [
    [1, 1, 1, 1],
    [0.3, 0.5, 0.2, 0.8],
    [0.05, 0.1, 0.4, 0.5],
  ];
  const points: [number, number][] = [
    [0.2, 0],
    [1.3, 0],
    [1.5, 1.4],
    [0.5, 0.5],
    [0.1, 0],
    [0.4, 0],
    [3.1, 2.2],
  ];
  for (const mode of ["confidence", "data-age"] as const) {
    const scores =
      mode === "confidence"
        ? [0, 1, 50, 99, 100, 127, 128, 140, 160, 191, 192, 230, 255]
        : [0, 1, 2, 3, 4, 5, 6, 7];
    const flags = mode === "confidence" ? [0, 1, UNSEEN, UNSEEN | 2] : [0, 255];
    for (const r of scores)
      for (const g of flags)
        for (const color of colors)
          for (const p of points)
            for (const cell of [1, 2])
              for (const weakBelow of [128, 100])
                out.push({ mode, texel: [r, g], color, p, cell, weakBelow });
  }
  return out;
}

const VERTEX = `#version 300 es
in vec2 position;
void main() { gl_Position = vec4(position, 0.0, 1.0); }
`;

const FRAGMENT = `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D uCases;
out vec4 outColor;
${CELL_LOOK_GLSL}
void main() {
    ivec2 at = ivec2(gl_FragCoord.xy);
    vec4 a = texelFetch(uCases, ivec2(at.x * 3, at.y), 0);
    vec4 b = texelFetch(uCases, ivec2(at.x * 3 + 1, at.y), 0);
    vec4 c = texelFetch(uCases, ivec2(at.x * 3 + 2, at.y), 0);
    outColor = hexapodCellLook(int(a.x + 0.5), ivec2(a.yz + 0.5), b.xy, a.w,
                               vec4(b.z, b.w, ${String(NEWER_STRIPE_M)}, 0.0), c);
}
`;

function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("no shader");
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(`the cell look did not compile: ${gl.getShaderInfoLog(shader) ?? ""}`);
  }
  return shader;
}

/** Draws every case with the GLSL and compares it with `cellViewColor`. */
export function cellLookAgreement(): LookAgreement {
  const all = cases();
  const width = 128;
  const height = Math.ceil(all.length / width);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const gl = canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: false });
  if (!gl) throw new Error("no WebGL2");
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX));
  gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(`the cell look did not link: ${gl.getProgramInfoLog(program) ?? ""}`);
  }
  gl.useProgram(program);

  const inputs = new Float32Array(width * 3 * height * 4);
  all.forEach((c, i) => {
    const at = (Math.floor(i / width) * width * 3 + (i % width) * 3) * 4;
    inputs.set(
      [
        c.mode === "confidence" ? 1 : 2,
        c.texel[0],
        c.texel[1],
        c.cell,
        c.p[0],
        c.p[1],
        c.weakBelow,
        UNSEEN,
        ...c.color,
      ],
      at,
    );
  });
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width * 3, height, 0, gl.RGBA, gl.FLOAT, inputs);
  gl.uniform1i(gl.getUniformLocation(program, "uCases"), 0);

  const quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, Float32Array.of(-1, -1, 3, -1, -1, 3), gl.STATIC_DRAW);
  const position = gl.getAttribLocation(program, "position");
  gl.enableVertexAttribArray(position);
  gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
  gl.viewport(0, 0, width, height);
  gl.disable(gl.BLEND);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  const pixels = new Uint8Array(width * height * 4);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

  let maxError = 0;
  let worst: LookAgreement["worst"] = null;
  all.forEach((c, i) => {
    const reference = cellViewColor(c.mode, c.texel, c.color, {
      weakBelow: c.weakBelow,
      unseenBit: UNSEEN,
      p: c.p,
      cell: c.cell,
    }).map((v) => Math.min(1, Math.max(0, v)));
    const gpu = Array.from(pixels.subarray(i * 4, i * 4 + 4), (v) => v / 255);
    const error = Math.max(...reference.map((v, k) => Math.abs(v - (gpu[k] ?? 0))));
    if (error > maxError) {
      maxError = error;
      worst = { input: c, gpu, reference };
    }
  });
  return { cases: all.length, maxError, worst };
}
