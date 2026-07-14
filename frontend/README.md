# AgentForge — Frontend (4-Step Wizard)

Polished Next.js UI for **AgentForge**, an AMD-centric, self-evolving AI-agent
sizing & education platform. A domain expert describes their AI need and current
hardware; the system explains what their **current AMD hardware can run**,
generates the **fastest verifiable PoC**, and shows what a **hardware upgrade
unlocks** — all in the customer's own domain language.

Core narrative surfaced by the UI:

- **Static Hardware Gatekeeper** — deterministic VRAM/bandwidth physics; a hard
  boundary that never evolves.
- **Evaluator** — a fuzzy, evolving judge of domain fit that learns from feedback.

## Stack

- Next.js 16 (App Router) + React 19 + TypeScript
- Tailwind CSS v4 (AMD-flavored dark theme)
- `react-markdown` + `remark-gfm` (TCO proposal), `lucide-react` (icons)

## The 4 steps

1. **Domain Definition** — enter domain + description + workload type; matched
   workflow templates are returned and selectable.
2. **Constraint Diagnostic** — pick current hardware (AMD tier or custom specs) +
   requirements (model, seq_len, concurrency, dtype); get a feasibility verdict,
   a stacked VRAM breakdown, and any gaps explained in domain language.
3. **Hardware Simulation Lab** — interactive controls (model, seq_len,
   population/batch, dtype, prefix-share ratio) that visualize the VRAM breakdown
   as stacked bars and compare across the AMD tiers
   (Ryzen AI → Radeon → Radeon PRO → Instinct MI300X), highlighting what each
   tier unlocks. Instinct/MI300X results are clearly labeled **SIMULATED**.
4. **Export** — generate the "AMD TCO & Procurement ROI Proposal" (rendered
   markdown) and a runnable deployment template (`docker-compose.yml`,
   LangGraph `app.py`, `README.md`) with copy/download. Includes a feedback
   control (rating + notes) that anchors the system's self-evolution.

## Run it

```bash
npm install
npm run dev      # http://localhost:3000
```

Production build:

```bash
npm run build
npm run start
```

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `NEXT_PUBLIC_API_BASE` | `http://localhost:8000` | Backend REST base URL |
| `NEXT_PUBLIC_USE_MOCK` | *(unset = auto)* | `true` = always mock, `false` = live only, unset = auto-fallback |

Copy `.env.example` to `.env.local` to customize.

## Mock / fallback layer

The backend may not be running while the UI is built or demoed, so
`lib/api.ts` has a graceful fallback:

- **`NEXT_PUBLIC_USE_MOCK=true`** → always serves built-in mock data.
- **`NEXT_PUBLIC_USE_MOCK=false`** → always hits the live API (errors surface).
- **unset (auto)** → tries the live API and, on any network/HTTP error,
  transparently serves realistic mock data.

Mock data (`lib/mock.ts`) matches the backend contract and includes realistic
AMD tiers (MI300X 192GB/5.3TB/s, W7900 48GB, RX 7900 XTX 24GB, Ryzen AI Max+ 395
128GB unified) and models. VRAM/throughput are computed from first-principles
physics (`lib/vram.ts`), so every control in the Simulation Lab produces
believable, responsive results even with no backend. The active data source
(Live vs Mock) is shown as a pill in the header.
```
