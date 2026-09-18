import { PostProcessStage } from "cesium";

/**
 * A light colour grade on the finished frame. The world's tiles are unlit textures drawn
 * as-is, which reads flatter than a maps app whose renderer adds a tone curve on output; a
 * touch of contrast and saturation brings the two to the same look. Applied after MSAA and
 * FXAA, one full-screen pass at a fixed cost of a fraction of a millisecond.
 */
export interface ColorGrade {
  /** 1 leaves contrast alone; 1.08 is the default. */
  contrast: number;
  /** 1 leaves saturation alone; 1.18 is the default. */
  saturation: number;
}

export const DEFAULT_COLOR_GRADE: ColorGrade = { contrast: 1.08, saturation: 1.18 };

const FRAGMENT_SHADER = /* glsl */ `
uniform sampler2D colorTexture;
uniform float u_contrast;
uniform float u_saturation;
in vec2 v_textureCoordinates;

void main() {
  vec4 color = texture(colorTexture, v_textureCoordinates);
  // Luminance-preserving saturation, then contrast around mid grey, both in display space.
  float luminance = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
  vec3 graded = mix(vec3(luminance), color.rgb, u_saturation);
  graded = (graded - 0.5) * u_contrast + 0.5;
  out_FragColor = vec4(clamp(graded, 0.0, 1.0), color.a);
}
`;

export function createColorGradeStage(grade: ColorGrade = DEFAULT_COLOR_GRADE): PostProcessStage {
  return new PostProcessStage({
    name: "twin_color_grade",
    fragmentShader: FRAGMENT_SHADER,
    uniforms: { u_contrast: grade.contrast, u_saturation: grade.saturation },
  });
}
