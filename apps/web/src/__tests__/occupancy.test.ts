import { describe, expect, it } from "vitest";

import { OccupancyGrid, SOLID, SplatOccupancy, cellFor, type Vec3 } from "@/lib/occupancy";

/** A wall of opaque splats in the plane x = `x`, 4 m square, every 5 cm. */
function wall(x: number): Float32Array {
  const points: number[] = [];
  for (let y = -2; y <= 2; y += 0.05) {
    for (let z = 0; z <= 4; z += 0.05) points.push(x, y, z);
  }
  return new Float32Array(points);
}

const opaque = (): number => 1;

function gridWithWall(x = 5, cell = 0.1): OccupancyGrid {
  const grid = new OccupancyGrid(cell);
  const points = wall(x);
  grid.add("wall", points, 0, points.length / 3, opaque);
  return grid;
}

describe("the occupancy grid", () => {
  it("makes a cell solid from one opaque splat, not from a faint floater", () => {
    const grid = new OccupancyGrid(0.1);
    grid.add("a", [0.05, 0.05, 0.05, 1.05, 0.05, 0.05], 0, 2, (i) => (i === 0 ? 0.9 : 0.1));
    expect(grid.solidAt(0, 0, 0)).toBe(true);
    expect(grid.solidAt(10, 0, 0)).toBe(false);
    // Several translucent splats add up to a surface.
    const haze = new OccupancyGrid(0.1);
    haze.add("h", [0.01, 0, 0, 0.02, 0, 0, 0.03, 0, 0], 0, 3, () => SOLID / 2);
    expect(haze.solidAt(0, 0, 0)).toBe(true);
  });

  it("takes back exactly what a tile added when it is swapped out", () => {
    const grid = gridWithWall();
    const cells = grid.solidCells;
    expect(cells).toBeGreaterThan(1000);
    grid.add("floor", [0.05, 0.05, 0.05], 0, 1, opaque);
    expect(grid.solidCells).toBe(cells + 1);
    grid.remove("wall");
    expect(grid.solidCells).toBe(1);
    grid.remove("floor");
    expect(grid.solidCells).toBe(0);
  });

  it("finds the first surface along a ray, at any angle, and nothing past its reach", () => {
    const grid = gridWithWall(5);
    expect(grid.raycast([0, 0, 1], [1, 0, 0], 100)).toBeCloseTo(5, 1);
    // Obliquely, the hit is on the wall's plane.
    const d: Vec3 = [1, 0.3, 0.2];
    const t = grid.raycast([0, 0, 1], d, 100) ?? Number.NaN;
    const unit = Math.hypot(...d);
    expect((t * d[0]) / unit).toBeCloseTo(5, 1);
    expect(grid.raycast([0, 0, 1], [1, 0, 0], 4)).toBeNull();
    expect(grid.raycast([0, 0, 1], [-1, 0, 0], 100)).toBeNull();
    // Over the top of the wall.
    expect(grid.raycast([0, 0, 6], [1, 0, 0], 100)).toBeNull();
  });

  it("stops a moving camera short of a surface instead of letting it through", () => {
    const grid = gridWithWall(5);
    const radius = 0.25;
    const { position, blocked } = grid.sweep([0, 0, 1], [10, 0, 1], radius);
    expect(blocked).toBe(true);
    expect(position[0]).toBeLessThan(5);
    expect(position[0]).toBeGreaterThan(5 - radius - 0.2);
    // One huge step (a fast flick) cannot tunnel through either.
    expect(grid.sweep([0, 0, 1], [1000, 0, 1], radius).position[0]).toBeLessThan(5);
  });

  it("slides along a surface met at an angle, keeping the motion along it", () => {
    const grid = gridWithWall(5);
    const { position, blocked } = grid.sweep([4.5, 0, 1], [5.5, 1, 1], 0.25);
    expect(blocked).toBe(true);
    expect(position[0]).toBeLessThan(5);
    // Most of the sideways motion survives.
    expect(position[1]).toBeGreaterThan(0.6);
  });

  it("slides along a bumpy surface too, instead of catching on every bump", () => {
    // A wall whose splats wander ±4 cm off its plane: a voxel surface with ridges.
    const points: number[] = [];
    let seed = 7;
    const random = (): number => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;
    for (let y = -3; y <= 3; y += 0.05) {
      for (let z = 0; z <= 4; z += 0.05) points.push(5 + random() * 0.04, y, z);
    }
    const grid = new OccupancyGrid(0.1);
    grid.add("bumpy", points, 0, points.length / 3, opaque);
    let position: Vec3 = [4.6, -2, 1];
    // Pressed into it at 45 degrees, in small moves, as frames do.
    for (let i = 0; i < 60; i++) {
      position = grid.sweep(position, [position[0] + 0.05, position[1] + 0.05, 1], 0.25).position;
    }
    expect(position[0]).toBeLessThan(5);
    expect(position[1]).toBeGreaterThan(0.5);
  });

  it("lets a camera already inside a surface move out, but no deeper", () => {
    const grid = gridWithWall(5);
    const out = grid.sweep([5.02, 0, 1], [3, 0, 1], 0.25);
    expect(out.position[0]).toBeCloseTo(3, 5);
    const deeper = grid.sweep([4.9, 0, 1], [5.05, 0, 1], 0.25);
    expect(deeper.position[0]).toBeLessThanOrEqual(4.9 + 1e-9);
  });

  it("moves freely where there is nothing", () => {
    const grid = gridWithWall(5);
    expect(grid.sweep([0, 0, 1], [-3, 2, 1], 0.25)).toEqual({
      position: [-3, 2, 1],
      blocked: false,
    });
  });
});

describe("sizing cells to the splats", () => {
  it("picks the smallest cell a surface fills: about three splats a cell", () => {
    // 5 cm spacing: 6.25 cm cells hold ~1.6, 12.5 cm hold ~6.
    const points = wall(0);
    expect(cellFor(points, 0, points.length / 3, opaque)).toBe(0.125);
    // Every other row and column: twice the spacing, twice the cell.
    const sparse: number[] = [];
    for (let i = 0; i < points.length / 3; i++) {
      const y = Math.round((points[i * 3 + 1] ?? 0) / 0.05);
      const z = Math.round((points[i * 3 + 2] ?? 0) / 0.05);
      if (y % 2 === 0 && z % 2 === 0) sparse.push(0, y * 0.05, z * 0.05);
    }
    expect(cellFor(sparse, 0, sparse.length / 3, opaque)).toBe(0.25);
  });

  it("sizes a large tile by its real density, not a thinned sample's", () => {
    // 1 cm spacing over 3.2 x 3.2 m: 102,400 splats. 2^-6 m cells hold ~2.4, 2^-5 m ~9.8.
    // A strided sample scaled by its stride read about the stride per cell at any size and
    // gave the finest cell, 4 mm.
    const wall1cm: number[] = [];
    for (let y = 0; y < 320; y++) for (let z = 0; z < 320; z++) wall1cm.push(0, y * 0.01, z * 0.01);
    expect(cellFor(wall1cm, 0, wall1cm.length / 3, opaque)).toBe(2 ** -5);
  });

  it("keeps a sparse tile a surface a ray cannot slip through", () => {
    const occupancy = new SplatOccupancy();
    const sparse: number[] = [];
    for (let y = -2; y <= 2; y += 0.2) for (let z = 0; z <= 4; z += 0.2) sparse.push(5, y, z);
    occupancy.add("coarse", sparse, 0, sparse.length / 3, opaque);
    for (const y of [-1.5, -0.73, 0.11, 0.9, 1.37]) {
      expect(occupancy.raycast([0, y, 2.05], [1, 0, 0], 100), `y = ${String(y)}`).not.toBeNull();
    }
  });

  it("describes each tile at its own level; the nearest hit of any wins", () => {
    const occupancy = new SplatOccupancy();
    const near = wall(3);
    occupancy.add("fine", near, 0, near.length / 3, opaque);
    const far: number[] = [];
    for (let y = -2; y <= 2; y += 0.2) for (let z = 0; z <= 4; z += 0.2) far.push(8, y, z);
    occupancy.add("coarse", far, 0, far.length / 3, opaque);
    expect(occupancy.cells.length).toBe(2);
    expect(occupancy.clearance).toBeCloseTo(0.125 * 1.5);
    expect(occupancy.raycast([0, 0, 1], [1, 0, 0], 100)).toBeCloseTo(3, 0);
    occupancy.remove("fine");
    expect(occupancy.raycast([0, 0, 1], [1, 0, 0], 100)).toBeGreaterThan(7);
    expect(occupancy.sweep([0, 0, 1], [20, 0, 1]).position[0]).toBeLessThan(8);
  });
});
