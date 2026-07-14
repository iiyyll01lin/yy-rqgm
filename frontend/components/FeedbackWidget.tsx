"use client";

import * as React from "react";
import { Sparkles, Star, ThumbsDown, ThumbsUp } from "lucide-react";
import { postFeedback } from "@/lib/api";
import { cn } from "@/lib/format";
import { Button, TextArea } from "@/components/ui";
import { useWizard } from "./wizard/WizardContext";

/**
 * Rating + notes control. Feedback is the ground-truth anchor that teaches the
 * evolving evaluator, so we make that role explicit to the user.
 */
export function FeedbackWidget() {
  const { sessionId } = useWizard();
  const [rating, setRating] = React.useState(0);
  const [hover, setHover] = React.useState(0);
  const [correct, setCorrect] = React.useState<boolean | null>(null);
  const [notes, setNotes] = React.useState("");
  const [submitting, setSubmitting] = React.useState(false);
  const [storedAs, setStoredAs] = React.useState<string | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  async function submit() {
    if (rating < 1 || !sessionId) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await postFeedback(sessionId, {
        rating,
        correct: correct ?? undefined,
        notes,
      });
      if (res.ok) setStoredAs(res.stored_as);
      else setError("回饋未能儲存，請重試。");
    } catch {
      setError("回饋送出失敗，請重試。");
    } finally {
      setSubmitting(false);
    }
  }

  if (storedAs) {
    return (
      <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-4 text-sm text-emerald-100">
        <div className="flex items-center gap-2 font-semibold">
          <Sparkles className="h-4 w-4" />
          感謝回饋 — 已記錄為進化錨點
        </div>
        <p className="mt-1 text-emerald-200/80">
          您的評分已存為 <code className="font-mono text-xs">{storedAs}</code>
          ，將作為評估器 (Evaluator) 冷路徑自我進化的 ground-truth 依據。
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-start gap-2 text-xs text-muted">
        <Sparkles className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amd" />
        <span>
          您的回饋是系統自我進化的 <strong className="text-zinc-300">ground-truth 錨點</strong>
          ：告訴我們判斷是否正確，就是在教這套系統變得更準。
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-1">
          {[1, 2, 3, 4, 5].map((n) => (
            <button
              key={n}
              type="button"
              aria-label={`${n} 星`}
              onMouseEnter={() => setHover(n)}
              onMouseLeave={() => setHover(0)}
              onClick={() => setRating(n)}
              className="p-0.5 transition-transform hover:scale-110"
            >
              <Star
                className={cn(
                  "h-6 w-6",
                  (hover || rating) >= n
                    ? "fill-amber-400 text-amber-400"
                    : "text-zinc-600",
                )}
              />
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs text-muted">判定正確？</span>
          <button
            type="button"
            onClick={() => setCorrect(correct === true ? null : true)}
            className={cn(
              "inline-flex items-center gap-1 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors",
              correct === true
                ? "border-emerald-500/50 bg-emerald-500/15 text-emerald-300"
                : "border-line text-zinc-400 hover:text-zinc-200",
            )}
          >
            <ThumbsUp className="h-3.5 w-3.5" /> 正確
          </button>
          <button
            type="button"
            onClick={() => setCorrect(correct === false ? null : false)}
            className={cn(
              "inline-flex items-center gap-1 rounded-lg border px-2.5 py-1.5 text-xs font-medium transition-colors",
              correct === false
                ? "border-red-500/50 bg-red-500/15 text-red-300"
                : "border-line text-zinc-400 hover:text-zinc-200",
            )}
          >
            <ThumbsDown className="h-3.5 w-3.5" /> 有誤
          </button>
        </div>
      </div>

      <TextArea
        rows={2}
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        placeholder="補充說明（選填）：例如「MI300X 對我們的多線並發搜尋很關鍵」…"
      />

      {error ? <p className="text-xs text-red-400">{error}</p> : null}

      <div className="flex items-center gap-3">
        <Button
          variant="secondary"
          onClick={submit}
          disabled={rating < 1 || submitting || !sessionId}
        >
          {submitting ? "送出中…" : "送出回饋"}
        </Button>
        {rating < 1 ? (
          <span className="text-xs text-muted">請先給予星級評分</span>
        ) : null}
      </div>
    </div>
  );
}
