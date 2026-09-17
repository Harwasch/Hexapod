import type { PlanStep } from "@twin/contracts";

import { scheduleLanes } from "@/missions/planDraft";

const LANE_H = 22;
const LABEL_W = 64;
const TOP = 18;

/** Machine lanes across the plan's days: who works where, when (design: PLAN timeline). */
export function PlanSchedule({
  steps,
  startDate,
  colorFor,
}: {
  steps: PlanStep[];
  startDate: string;
  colorFor: (machineId: string) => string;
}) {
  const { lanes, totalDays } = scheduleLanes(steps);
  if (lanes.length === 0) return null;
  const width = 520;
  const plotW = width - LABEL_W - 8;
  const dayW = plotW / totalDays;
  const height = TOP + lanes.length * LANE_H + 6;
  const tickEvery = totalDays > 21 ? 7 : totalDays > 10 ? 2 : 1;
  const start = new Date(`${startDate}T00:00:00`);
  const dayLabel = (day: number) => {
    if (Number.isNaN(start.getTime())) return `D${day + 1}`;
    const d = new Date(start);
    d.setDate(d.getDate() + day);
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  };
  return (
    <svg
      className="mc-schedule"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Schedule: ${lanes.length} machines over ${totalDays} days`}
      data-testid="plan-schedule"
    >
      {Array.from({ length: totalDays + 1 }, (_, day) =>
        day % tickEvery === 0 ? (
          <g key={day}>
            <line
              x1={LABEL_W + day * dayW}
              x2={LABEL_W + day * dayW}
              y1={TOP - 4}
              y2={height - 4}
              className="mc-schedule__grid"
            />
            {day < totalDays && (
              <text x={LABEL_W + day * dayW + 3} y={TOP - 7} className="mc-schedule__tick">
                {dayLabel(day)}
              </text>
            )}
          </g>
        ) : null,
      )}
      {lanes.map((lane, row) => {
        const y = TOP + row * LANE_H;
        return (
          <g key={lane.machineId}>
            <text x={0} y={y + 14} className="mc-schedule__lane">
              {lane.machineId}
            </text>
            {lane.bars.map((bar, i) => (
              <g key={`${i}-${bar.title}`}>
                <rect
                  x={LABEL_W + bar.startDay * dayW + 1}
                  y={y + 3}
                  width={Math.max(bar.days * dayW - 2, 3)}
                  height={LANE_H - 6}
                  rx={3}
                  fill={
                    lane.machineId === "Fleet" ? "rgba(255,255,255,0.35)" : colorFor(lane.machineId)
                  }
                >
                  <title>{`${bar.title} · ${dayLabel(bar.startDay)} · ${bar.days} d`}</title>
                </rect>
                {bar.days * dayW > 46 && (
                  <text
                    x={LABEL_W + bar.startDay * dayW + 6}
                    y={y + 14}
                    className="mc-schedule__bar-label"
                  >
                    {bar.zoneId ?? bar.title}
                  </text>
                )}
              </g>
            ))}
          </g>
        );
      })}
    </svg>
  );
}
