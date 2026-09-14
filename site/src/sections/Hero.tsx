import { motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { REPO_URL, DOCS_URL } from "../lib/links";

const nodes = [
  { id: "start", x: 60, y: 180, label: "start", accent: "#FFFFFF" },
  { id: "search", x: 200, y: 100, label: "search", accent: "#A88EFF" },
  { id: "fetch", x: 200, y: 260, label: "fetch", accent: "#5EE2FF" },
  { id: "verify", x: 360, y: 140, label: "verify", accent: "#7C5CFF" },
  { id: "answer", x: 520, y: 200, label: "answer", accent: "#FFFFFF" },
];

const edges = [
  { from: "start", to: "search" },
  { from: "start", to: "fetch" },
  { from: "search", to: "verify" },
  { from: "fetch", to: "verify" },
  { from: "verify", to: "answer" },
];

function getNode(id: string) {
  return nodes.find((n) => n.id === id)!;
}

function buildPath() {
  return ["start", "search", "verify", "answer"] as const;
}

export default function Hero() {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick((t) => (t + 1) % 4), 1600);
    return () => clearInterval(id);
  }, []);

  const path = useMemo(() => buildPath(), []);
  const activeSet = useMemo(() => {
    const set = new Set<string>();
    for (let i = 0; i <= tick; i++) set.add(path[i]);
    return set;
  }, [tick, path]);

  const activeEdges = useMemo(() => {
    const set = new Set<string>();
    for (let i = 0; i < tick; i++) {
      const a = path[i];
      const b = path[i + 1];
      set.add(`${a}->${b}`);
    }
    return set;
  }, [tick, path]);

  return (
    <section id="top" className="relative pt-32 md:pt-44 pb-24">
      <div className="container-x">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
          className="flex flex-col items-center text-center"
        >
          <span className="pill mb-7">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-accent-400 shadow-[0_0_12px_2px_rgba(124,92,255,0.7)]" />
            v0.1.0 — paper-faithful, mypy strict, ≥95% coverage
          </span>

          <h1 className="text-balance text-[44px] leading-[1.04] tracking-tightest font-semibold md:text-[72px] md:leading-[0.98]">
            <span className="gradient-text">Procedural memory</span>
            <br />
            <span className="gradient-text-accent">that learns from itself.</span>
          </h1>

          <p className="mt-7 max-w-2xl text-balance text-base md:text-lg leading-relaxed text-chalk-400">
            methodos wraps any ReAct-style agent with a queryable, self-evolving
            procedural graph. Guidance is generated from a localized subgraph,
            refined offline from real execution feedback, and validated before
            it ever ships.
          </p>

          <div className="mt-10 flex flex-col sm:flex-row items-center gap-3">
            <a href={`${DOCS_URL}`} target="_blank" rel="noreferrer" className="btn-primary">
              Get started
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" className="transition-transform group-hover:translate-x-0.5">
                <path d="M5 12h14M13 5l7 7-7 7" />
              </svg>
            </a>
            <a href={REPO_URL} target="_blank" rel="noreferrer" className="btn-ghost">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
                <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.58.11.79-.25.79-.56v-2.02c-3.2.7-3.88-1.54-3.88-1.54-.52-1.33-1.28-1.69-1.28-1.69-1.05-.72.08-.71.08-.71 1.16.08 1.78 1.2 1.78 1.2 1.03 1.77 2.7 1.26 3.36.97.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.19 1.18a11 11 0 0 1 5.8 0c2.22-1.49 3.19-1.18 3.19-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.84 1.19 3.1 0 4.42-2.69 5.39-5.25 5.68.41.36.78 1.06.78 2.14v3.18c0 .31.21.68.8.56C20.21 21.38 23.5 17.08 23.5 12 23.5 5.65 18.35.5 12 .5Z" />
              </svg>
              Star on GitHub
            </a>
          </div>
        </motion.div>

        {/* Graph Visual */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.2, ease: [0.22, 1, 0.36, 1] }}
          className="relative mx-auto mt-20 max-w-5xl"
        >
          <div className="absolute -inset-x-12 -inset-y-8 -z-10 rounded-[40px] bg-[radial-gradient(ellipse_at_top,rgba(124,92,255,0.25),transparent_60%)] blur-2xl" />
          <div className="relative overflow-hidden rounded-3xl border border-white/[0.07] bg-gradient-to-b from-ink-800/80 to-ink-900/80 shadow-glow">
            <div className="flex items-center justify-between border-b border-white/[0.05] px-5 py-3">
              <div className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-white/15" />
                <span className="h-2.5 w-2.5 rounded-full bg-white/15" />
                <span className="h-2.5 w-2.5 rounded-full bg-white/15" />
              </div>
              <div className="text-[11px] font-medium uppercase tracking-[0.22em] text-chalk-500">
                graph.live — agent trajectory
              </div>
              <div className="flex items-center gap-2 text-[11px] text-chalk-400">
                <span className="inline-block h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
                refining
              </div>
            </div>

            <div className="relative aspect-[16/8] w-full">
              <svg
                viewBox="0 0 600 360"
                className="absolute inset-0 h-full w-full"
                role="img"
                aria-label="Animated procedural graph"
              >
                <defs>
                  <linearGradient id="edge-grad" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="#7C5CFF" />
                    <stop offset="100%" stopColor="#5EE2FF" />
                  </linearGradient>
                  <linearGradient id="edge-faint" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="rgba(255,255,255,0.18)" />
                    <stop offset="100%" stopColor="rgba(255,255,255,0.05)" />
                  </linearGradient>
                  <radialGradient id="node-glow" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="rgba(124,92,255,0.45)" />
                    <stop offset="100%" stopColor="rgba(124,92,255,0)" />
                  </radialGradient>
                </defs>

                {/* dot grid */}
                <g opacity="0.35">
                  {Array.from({ length: 7 }).map((_, r) =>
                    Array.from({ length: 12 }).map((_, c) => (
                      <circle
                        key={`${r}-${c}`}
                        cx={30 + c * 50}
                        cy={40 + r * 50}
                        r={0.8}
                        fill="rgba(255,255,255,0.18)"
                      />
                    ))
                  )}
                </g>

                {/* edges */}
                {edges.map((e) => {
                  const a = getNode(e.from);
                  const b = getNode(e.to);
                  const active = activeEdges.has(`${e.from}->${e.to}`);
                  return (
                    <g key={`${e.from}-${e.to}`}>
                      <line
                        x1={a.x}
                        y1={a.y}
                        x2={b.x}
                        y2={b.y}
                        stroke={active ? "url(#edge-grad)" : "url(#edge-faint)"}
                        strokeWidth={active ? 2.4 : 1.2}
                        strokeLinecap="round"
                      />
                      {active && (
                        <circle r="3" fill="#fff">
                          <animateMotion
                            dur="1.8s"
                            repeatCount="indefinite"
                            path={`M${a.x},${a.y} L${b.x},${b.y}`}
                          />
                        </circle>
                      )}
                    </g>
                  );
                })}

                {/* nodes */}
                {nodes.map((n) => {
                  const isActive = activeSet.has(n.id);
                  return (
                    <g key={n.id} transform={`translate(${n.x},${n.y})`}>
                      {isActive && (
                        <circle r="28" fill="url(#node-glow)">
                          <animate
                            attributeName="r"
                            values="22;32;22"
                            dur="2.2s"
                            repeatCount="indefinite"
                          />
                        </circle>
                      )}
                      <circle
                        r="11"
                        fill="#0A0B10"
                        stroke={n.accent}
                        strokeWidth={isActive ? 2.2 : 1.2}
                        opacity={isActive ? 1 : 0.6}
                      />
                      <circle r="3.5" fill={n.accent} opacity={isActive ? 1 : 0.5} />
                      <text
                        y="28"
                        textAnchor="middle"
                        fontFamily="ui-monospace, SF Mono, Menlo, monospace"
                        fontSize="11"
                        fill="rgba(255,255,255,0.65)"
                        style={{ letterSpacing: "0.08em" }}
                      >
                        {n.label}
                      </text>
                    </g>
                  );
                })}
              </svg>

              {/* Floating guidance card */}
              <motion.div
                key={tick}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.45 }}
                className="absolute right-5 bottom-5 w-[260px] rounded-2xl border border-white/[0.07] bg-ink-900/80 p-4 backdrop-blur-xl shadow-glow"
              >
                <div className="flex items-center justify-between">
                  <div className="text-[10px] uppercase tracking-[0.2em] text-chalk-500">
                    Ψ guidance
                  </div>
                  <div className="text-[10px] text-chalk-500">{tick + 1}/4</div>
                </div>
                <div className="mt-2 text-[12px] leading-relaxed text-chalk-200">
                  {guidanceFor(tick)}
                </div>
                <div className="mt-3 flex items-center gap-2 text-[10px] text-chalk-500">
                  <span className="rounded-full border border-white/10 px-2 py-0.5">cached</span>
                  <span className="rounded-full border border-white/10 px-2 py-0.5">2-hop</span>
                </div>
              </motion.div>
            </div>
          </div>

          {/* Subtle badges below */}
          <div className="mt-8 flex flex-wrap items-center justify-center gap-3 text-[11px] uppercase tracking-[0.18em] text-chalk-500">
            <span>·</span>
            <span>paper §3.2</span>
            <span>·</span>
            <span>Algorithm 1</span>
            <span>·</span>
            <span>Pydantic v2</span>
            <span>·</span>
            <span>FastAPI</span>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

function guidanceFor(step: number) {
  const lines = [
    "Match last action → node. Extract 2-hop neighborhood.",
    "Translate subgraph into a tight guidance paragraph.",
    "Inject into AgentState.context. Call host solver.",
    "Record trajectory. Queue for offline refinement.",
  ];
  return lines[step] ?? lines[0];
}
