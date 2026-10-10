import type { components } from "@twin/contracts";
type Period = components["schemas"]["VegetationPeriod"];
export function vegetationPeriods(
  months: string[],
  today = new Date().toISOString().slice(0, 10),
): Period[] | null {
  if (!months.length || months.length > 6 || new Set(months).size !== months.length) return null;
  const result: Period[] = [];
  for (const month of [...months].sort()) {
    if (!/^20\d{2}-(0[1-9]|1[0-2])$/.test(month) || month < "2015-07" || month > today.slice(0, 7))
      return null;
    const [year = 0, index = 0] = month.split("-").map(Number);
    const end = new Date(Date.UTC(year, index, 0)).toISOString().slice(0, 10);
    result.push({ startDate: `${month}-01`, endDate: end < today ? end : today });
  }
  return result;
}
