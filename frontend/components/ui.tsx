"use client";

/** Shared, themed UI primitives for the AgentForge wizard. */

import * as React from "react";
import { cn } from "@/lib/format";

/* --------------------------------------------------------------------- */
/* Surfaces                                                               */
/* --------------------------------------------------------------------- */

export function Card({
  className,
  children,
  ...rest
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-2xl border border-line bg-surface/80 backdrop-blur-sm",
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
}

export function SectionTitle({
  eyebrow,
  title,
  desc,
  right,
}: {
  eyebrow?: string;
  title: React.ReactNode;
  desc?: React.ReactNode;
  right?: React.ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-4">
      <div>
        {eyebrow ? (
          <div className="mb-1 text-xs font-semibold uppercase tracking-widest text-amd">
            {eyebrow}
          </div>
        ) : null}
        <h2 className="text-xl font-semibold text-white sm:text-2xl">{title}</h2>
        {desc ? (
          <p className="mt-1.5 max-w-2xl text-sm leading-relaxed text-muted">
            {desc}
          </p>
        ) : null}
      </div>
      {right ? <div className="shrink-0">{right}</div> : null}
    </div>
  );
}

export function Badge({
  children,
  tone = "neutral",
  className,
}: {
  children: React.ReactNode;
  tone?: "neutral" | "amd" | "green" | "amber" | "sky" | "red";
  className?: string;
}) {
  const tones: Record<string, string> = {
    neutral: "bg-white/5 text-zinc-300 border-line",
    amd: "bg-amd/15 text-red-200 border-amd/40",
    green: "bg-emerald-500/10 text-emerald-300 border-emerald-500/30",
    amber: "bg-amber-500/10 text-amber-300 border-amber-500/30",
    sky: "bg-sky-500/10 text-sky-300 border-sky-500/30",
    red: "bg-red-500/10 text-red-300 border-red-500/30",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium",
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function Stat({
  label,
  value,
  sub,
  accent,
}: {
  label: React.ReactNode;
  value: React.ReactNode;
  sub?: React.ReactNode;
  accent?: string;
}) {
  return (
    <div className="rounded-xl border border-line bg-surface-2/60 px-4 py-3">
      <div className="text-xs font-medium text-muted">{label}</div>
      <div
        className="mt-1 text-2xl font-semibold tabular-nums text-white"
        style={accent ? { color: accent } : undefined}
      >
        {value}
      </div>
      {sub ? <div className="mt-0.5 text-xs text-muted">{sub}</div> : null}
    </div>
  );
}

export function Callout({
  tone = "neutral",
  title,
  children,
  icon,
}: {
  tone?: "neutral" | "success" | "warning" | "danger" | "info";
  title?: React.ReactNode;
  children?: React.ReactNode;
  icon?: React.ReactNode;
}) {
  const tones: Record<string, string> = {
    neutral: "border-line bg-surface-2/60 text-zinc-300",
    success: "border-emerald-500/30 bg-emerald-500/10 text-emerald-100",
    warning: "border-amber-500/30 bg-amber-500/10 text-amber-100",
    danger: "border-red-500/30 bg-red-500/10 text-red-100",
    info: "border-sky-500/30 bg-sky-500/10 text-sky-100",
  };
  return (
    <div className={cn("flex gap-3 rounded-xl border p-3.5 text-sm", tones[tone])}>
      {icon ? <div className="mt-0.5 shrink-0">{icon}</div> : null}
      <div className="min-w-0">
        {title ? <div className="font-semibold">{title}</div> : null}
        {children ? (
          <div className={cn(title ? "mt-1" : null, "leading-relaxed")}>
            {children}
          </div>
        ) : null}
      </div>
    </div>
  );
}

/* --------------------------------------------------------------------- */
/* Controls                                                               */
/* --------------------------------------------------------------------- */

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

export function Button({
  variant = "primary",
  className,
  children,
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
}) {
  const variants: Record<ButtonVariant, string> = {
    primary:
      "bg-amd text-white hover:bg-amd-bright shadow-[0_6px_20px_-8px_rgba(237,28,36,0.7)]",
    secondary:
      "border border-line bg-surface-2 text-zinc-100 hover:border-zinc-600 hover:bg-surface-2/70",
    ghost: "text-zinc-300 hover:bg-white/5",
    danger: "bg-red-600 text-white hover:bg-red-500",
  };
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-sm font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-50",
        variants[variant],
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
}

export function Field({
  label,
  hint,
  htmlFor,
  children,
  className,
}: {
  label: React.ReactNode;
  hint?: React.ReactNode;
  htmlFor?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("space-y-1.5", className)}>
      <label
        htmlFor={htmlFor}
        className="flex items-center justify-between text-sm font-medium text-zinc-200"
      >
        <span>{label}</span>
        {hint ? <span className="text-xs font-normal text-muted">{hint}</span> : null}
      </label>
      {children}
    </div>
  );
}

const inputBase =
  "w-full rounded-xl border border-line bg-surface-2 px-3.5 py-2.5 text-sm text-zinc-100 outline-none transition-colors placeholder:text-zinc-500 focus:border-amd focus:ring-2 focus:ring-amd/30";

export const TextInput = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(function TextInput({ className, ...rest }, ref) {
  return <input ref={ref} className={cn(inputBase, className)} {...rest} />;
});

export const TextArea = React.forwardRef<
  HTMLTextAreaElement,
  React.TextareaHTMLAttributes<HTMLTextAreaElement>
>(function TextArea({ className, ...rest }, ref) {
  return (
    <textarea
      ref={ref}
      className={cn(inputBase, "resize-none leading-relaxed", className)}
      {...rest}
    />
  );
});

export function Select({
  className,
  children,
  ...rest
}: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <div className="relative">
      <select
        className={cn(
          inputBase,
          "appearance-none pr-9 [&>option]:bg-surface-2 [&>option]:text-zinc-100",
          className,
        )}
        {...rest}
      >
        {children}
      </select>
      <svg
        className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted"
        viewBox="0 0 20 20"
        fill="currentColor"
        aria-hidden
      >
        <path
          fillRule="evenodd"
          d="M5.23 7.21a.75.75 0 011.06.02L10 11.17l3.71-3.94a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z"
          clipRule="evenodd"
        />
      </svg>
    </div>
  );
}

export function Range({
  value,
  min,
  max,
  step = 1,
  onChange,
  className,
  "aria-label": ariaLabel,
}: {
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
  className?: string;
  "aria-label"?: string;
}) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <input
      type="range"
      className={cn("amd-range", className)}
      style={{ "--range-pct": `${pct}%` } as React.CSSProperties}
      value={value}
      min={min}
      max={max}
      step={step}
      aria-label={ariaLabel}
      onChange={(e) => onChange(Number(e.target.value))}
    />
  );
}

export interface SegOption<T extends string> {
  value: T;
  label: React.ReactNode;
}

export function Segmented<T extends string>({
  options,
  value,
  onChange,
  className,
  size = "md",
}: {
  options: SegOption<T>[];
  value: T;
  onChange: (v: T) => void;
  className?: string;
  size?: "sm" | "md";
}) {
  return (
    <div
      className={cn(
        "inline-flex rounded-xl border border-line bg-surface-2 p-1",
        className,
      )}
    >
      {options.map((opt) => (
        <button
          key={opt.value}
          type="button"
          onClick={() => onChange(opt.value)}
          className={cn(
            "rounded-lg font-medium transition-colors",
            size === "sm" ? "px-2.5 py-1 text-xs" : "px-3.5 py-1.5 text-sm",
            value === opt.value
              ? "bg-amd text-white shadow-sm"
              : "text-zinc-400 hover:text-zinc-100",
          )}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <svg
      className={cn("h-4 w-4 animate-spin", className)}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-90"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.37 0 0 5.37 0 12h4z"
      />
    </svg>
  );
}
