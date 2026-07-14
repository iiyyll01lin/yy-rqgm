"use client";

import * as React from "react";
import { Check, Copy, Download, FileCode2 } from "lucide-react";
import { cn } from "@/lib/format";

function langLabel(filename: string): string {
  if (filename.endsWith(".yml") || filename.endsWith(".yaml")) return "YAML";
  if (filename.endsWith(".py")) return "Python";
  if (filename.endsWith(".md")) return "Markdown";
  if (filename.endsWith(".json")) return "JSON";
  if (filename.endsWith(".sh")) return "Shell";
  return "Text";
}

function download(filename: string, content: string) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/** Tabbed viewer for generated deployment files, with copy + download. */
export function CodeFileViewer({ files }: { files: Record<string, string> }) {
  const names = React.useMemo(() => Object.keys(files), [files]);
  const [selected, setSelected] = React.useState<string | null>(null);
  const [copied, setCopied] = React.useState(false);

  // Derive the active file so it stays valid when `files` changes (no effect).
  const active =
    selected && names.includes(selected) ? selected : (names[0] ?? "");
  const content = files[active] ?? "";

  async function copy() {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard may be blocked; ignore */
    }
  }

  if (names.length === 0) return null;

  return (
    <div className="overflow-hidden rounded-xl border border-line bg-surface-2">
      <div className="flex flex-wrap items-center gap-1 border-b border-line bg-surface px-2 py-2">
        {names.map((name) => (
          <button
            key={name}
            type="button"
            onClick={() => setSelected(name)}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium transition-colors",
              active === name
                ? "bg-amd/15 text-red-200"
                : "text-zinc-400 hover:bg-white/5 hover:text-zinc-200",
            )}
          >
            <FileCode2 className="h-3.5 w-3.5" />
            {name}
          </button>
        ))}
        <div className="ml-auto flex items-center gap-1">
          <span className="mr-1 hidden text-[11px] uppercase tracking-wide text-muted sm:inline">
            {langLabel(active)}
          </span>
          <button
            type="button"
            onClick={copy}
            className="inline-flex items-center gap-1 rounded-lg border border-line px-2 py-1.5 text-xs text-zinc-300 hover:bg-white/5"
          >
            {copied ? (
              <>
                <Check className="h-3.5 w-3.5 text-emerald-400" /> 已複製
              </>
            ) : (
              <>
                <Copy className="h-3.5 w-3.5" /> 複製
              </>
            )}
          </button>
          <button
            type="button"
            onClick={() => download(active, content)}
            className="inline-flex items-center gap-1 rounded-lg border border-line px-2 py-1.5 text-xs text-zinc-300 hover:bg-white/5"
          >
            <Download className="h-3.5 w-3.5" /> 下載
          </button>
        </div>
      </div>
      <pre className="max-h-[420px] overflow-auto p-4 text-xs leading-relaxed">
        <code className="font-mono text-zinc-200">{content}</code>
      </pre>
    </div>
  );
}
