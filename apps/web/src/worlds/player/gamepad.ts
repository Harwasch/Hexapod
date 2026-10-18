/** Standard gamepad axes: left stick translates, right stick looks. */
export function gamepadMotion(axes: readonly number[], deadZone = 0.15) {
  const axis = (index: number) => {
    const value = axes[index] ?? 0;
    if (!Number.isFinite(value) || Math.abs(value) <= deadZone) return 0;
    return (Math.sign(value) * (Math.min(1, Math.abs(value)) - deadZone)) / (1 - deadZone);
  };
  return { forward: -axis(1), right: axis(0), up: 0, yaw: axis(2), pitch: -axis(3) };
}
