"use client";

/** VRAM + tier comparison visualizations. */

import * as React from "react";
import { SEGMENT_COLORS, cn, fmtGB } from "@/lib/format";
import { VRAM_SEGMENTS, type VramBreakdown } from "@/lib/types";

/**
 * Horizontal stacked VRAM bar. When `capacityGb` is provided, the track is
 * scaled to max(total, capacity) so both headroom (capacity > total) and
 * overflow / OOM (total > capacity) are visible, with a dashed capacity marker.
 */
export function VramBar({
  breakdown,
  total,
  capacityGb,
  height = 26,
  className,
}: {
  breakdown: VramBreakdown;
  total: number;
  capacityGb?: number;
  height?: number;
  className?: string;
}) {
  const scaleMax = Math.max(total, capacityGb ?? 0, 0.001);
  const overflow = capacityGb != null && total > capacityGb;
  const capPct =
    capacityGb != null ? Math.min(100, (capacityGb / scaleMax) * 100) : null;

  return (
    <div className={cn("w-full", className)}>
      <div
        className="relative w-full overflow-hidden rounded-lg border border-line bg-surface-2"
        style={{ height }}
      >
        <div className="flex h-full w-full">
          {VRAM_SEGMENTS.map((seg) => {
            const value = breakdown[seg.key];
            const w = (value / scaleMax) * 100;
            if (w <= 0) return null;
            return (
              <div
                key={seg.key}
                className="h-full transition-[width] duration-300"
                style={{ width: `${w}%`, backgroundColor: SEGMENT_COLORS[seg.key] }}
                title={`${seg.label}: ${fmtGB(value)}`}
              />
            );
          })}
        </div>

        {/* Overflow / OOM region (beyond capacity). */}
        {overflow && capPct != null ? (
          <div
            className="pointer-events-none absolute inset-y-0"
            style={{
              left: `${capPct}%`,
              right: 0,
              backgroundImage:
                "repeating-linear-gradient(45deg, rgba(0,0,0,0.45) 0 6px, rgba(0,0,0,0.15) 6px 12px)",
            }}
          />
        ) : null}

        {/* Capacity marker. */}
        {capPct != null ? (
          <div
            className="pointer-events-none absolute inset-y-0 w-0.5 bg-white/90"
            style={{ left: `calc(${capPct}% - 1px)` }}
          />
        ) : null}
      </div>

      {capacityGb != null ? (
        <div className="mt-1 flex items-center justify-between text-[11px] text-muted">
          <span className="tabular-nums">
            使用 {fmtGB(total)} / 容量 {fmtGB(capacityGb)}
          </span>
          <span
            className={cn(
              "font-medium tabular-nums",
              overflow ? "text-red-400" : "text-emerald-400",
            )}
          >
            {overflow
              ? `超出 ${fmtGB(total - capacityGb)}`
              : `餘裕 ${fmtGB(capacityGb - total)}`}
          </span>
        </div>
      ) : null}
    </div>
  );
}

/** Legend describing each VRAM segment with its current value + share. */
export function VramLegend({
  breakdown,
  total,
  columns = 2,
}: {
  breakdown: VramBreakdown;
  total: number;
  columns?: 2 | 4;
}) {
  return (
    <div
      className={cn(
        "grid gap-x-4 gap-y-2",
        columns === 4 ? "grid-cols-2 sm:grid-cols-4" : "grid-cols-2",
      )}
    >
      {VRAM_SEGMENTS.map((seg) => {
        const value = breakdown[seg.key];
        const pct = total > 0 ? (value / total) * 100 : 0;
        return (
          <div key={seg.key} className="flex items-center gap-2">
            <span
              className="mt-0.5 h-3 w-3 shrink-0 rounded-sm"
              style={{ backgroundColor: SEGMENT_COLORS[seg.key] }}
            />
            <div className="min-w-0 leading-tight">
              <div className="truncate text-xs text-zinc-300" title={seg.hint}>
                {seg.label}
              </div>
              <div className="text-xs font-medium tabular-nums text-white">
                {fmtGB(value)}{" "}
                <span className="font-normal text-muted">
                  ({pct.toFixed(0)}%)
                </span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export interface MetricBarItem {
  id: string;
  label: React.ReactNode;
  value: number;
  display: string;
  color: string;
  highlight?: boolean;
  simulated?: boolean;
}

/**
 * Compact horizontal comparison bars (used for max-population and tokens/s
 * across tiers). The largest value defines the full-width scale.
 */
export function MetricBars({
  items,
  className,
}: {
  items: MetricBarItem[];
  className?: string;
}) {
  const max = Math.max(...items.map((i) => i.value), 1);
  return (
    <div className={cn("space-y-2.5", className)}>
      {items.map((item) => {
        const w = Math.max(2, (item.value / max) * 100);
        return (
          <div key={item.id} className="flex items-center gap-3">
            <div className="flex w-36 shrink-0 items-center gap-1.5 text-xs text-zinc-300">
              <span className="truncate">{item.label}</span>
              {item.simulated ? (
                <span className="rounded bg-amber-500/15 px-1 text-[9px] font-semibold uppercase text-amber-300">
                  sim
                </span>
              ) : null}
            </div>
            <div className="relative h-5 flex-1 overflow-hidden rounded-md bg-surface-2">
              <div
                className={cn(
                  "h-full rounded-md transition-[width] duration-500",
                  item.highlight && "shadow-[0_0_16px_-2px_currentColor]",
                )}
                style={{ width: `${w}%`, backgroundColor: item.color, color: item.color }}
              />
            </div>
            <div
              className={cn(
                "w-24 shrink-0 text-right text-xs font-semibold tabular-nums",
                item.highlight ? "text-white" : "text-zinc-300",
              )}
            >
              {item.display}
            </div>
          </div>
        );
      })}
    </div>
  );
}
