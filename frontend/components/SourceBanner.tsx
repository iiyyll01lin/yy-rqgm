"use client";

import * as React from "react";
import { Database, Radio, TriangleAlert } from "lucide-react";
import { API_BASE, getSource, subscribeSource, type DataSource } from "@/lib/api";
import { cn } from "@/lib/format";

/**
 * Server / first-hydration snapshot. The live-vs-mock source is a client-side
 * detection (it depends on whether the backend actually answers), so the server
 * render and the very first client render must both show the neutral "unknown"
 * state. Returning a constant here — rather than reading the live source — keeps
 * SSR and hydration byte-identical regardless of NEXT_PUBLIC_USE_MOCK, which is
 * inlined into the client bundle at build time and could otherwise disagree with
 * the server at runtime. useSyncExternalStore then swaps in the real source
 * after hydration, without a mismatch.
 */
const getServerSource = (): DataSource => "unknown";

/** Small pill showing whether the UI is talking to the live API or mock data. */
export function SourceBanner({ className }: { className?: string }) {
  const source = React.useSyncExternalStore<DataSource>(
    subscribeSource,
    getSource,
    getServerSource,
  );

  if (source === "live") {
    return (
      <span
        className={cn(
          "inline-flex items-center gap-1.5 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2.5 py-1 text-xs font-medium text-emerald-300",
          className,
        )}
        title={`已連線後端 API：${API_BASE}`}
      >
        <Radio className="h-3.5 w-3.5" />
        Live API
      </span>
    );
  }

  if (source === "mock") {
    return (
      <span
        className={cn(
          "inline-flex items-center gap-1.5 rounded-full border border-amber-500/30 bg-amber-500/10 px-2.5 py-1 text-xs font-medium text-amber-300",
          className,
        )}
        title={`後端 (${API_BASE}) 未連線，使用內建示範資料。數值由第一性原理物理估算，仍完全可互動。`}
      >
        <TriangleAlert className="h-3.5 w-3.5" />
        示範資料 · Mock
      </span>
    );
  }

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border border-line bg-surface-2 px-2.5 py-1 text-xs font-medium text-muted",
        className,
      )}
    >
      <Database className="h-3.5 w-3.5" />
      連線中…
    </span>
  );
}
