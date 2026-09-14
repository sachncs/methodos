import { motion, useInView } from "framer-motion";
import { useRef } from "react";

const steps = [
  {
    n: "01",
    title: "Online — guidance per step",
    body:
      "On every step we match the agent's last action to a node, extract the 2-hop neighborhood, and translate that subgraph into a short guidance paragraph. The result is cached and injected into your solver's context.",
    bullets: ["match_node(last_action)", "extract 2-hop subgraph", "Ψ prompt → cached guidance"],
  },
  {
    n: "02",
    title: "Offline — Algorithm 1",
    body:
      "Trajectories from real runs are scored. A refiner LLM proposes Edits — add or delete nodes and edges, update attributes. We apply them structurally, reject cycles, and accept only if validation does not regress.",
    bullets: ["score trajectories", "propose Edits (LLM)", "validate → accept or reject"],
  },
  {
    n: "03",
    title: "Always — local and typed",
    body:
      "Everything is Pydantic v2. The graph persists in SQLite, with optional sqlite-vec for fuzzy matching and a FastAPI surface for serving. No background magic. No semi-private names. No lazy imports.",
    bullets: ["Pydantic v2 models", "SQLite · sqlite-vec", "FastAPI on port 8000"],
  },
];

export default function HowItWorks() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section id="how-it-works" ref={ref} className="relative py-24 md:py-32">
      <div className="container-x">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          className="max-w-2xl"
        >
          <span className="pill">How it works</span>
          <h2 className="mt-5 text-[32px] md:text-[48px] font-semibold tracking-tightest leading-[1.05] gradient-text">
            A graph that updates itself, without ever hurting production.
          </h2>
          <p className="mt-5 text-base md:text-lg text-chalk-400 leading-relaxed">
            methodos is the procedural counterpart to a knowledge graph. It
            answers <em>what-to-do</em> questions by exposing a localized
            subgraph of admissible next procedures — and it gets sharper every
            run.
          </p>
        </motion.div>

        <div className="mt-16 md:mt-20 grid gap-6 md:grid-cols-3">
          {steps.map((s, i) => (
            <motion.article
              key={s.n}
              initial={{ opacity: 0, y: 24 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.7, delay: 0.1 + i * 0.08, ease: [0.22, 1, 0.36, 1] }}
              className="group relative overflow-hidden rounded-2xl border border-white/[0.06] bg-gradient-to-b from-ink-800/60 to-ink-900/40 p-7 hover:border-white/[0.12] transition-colors"
            >
              <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-accent-500/60 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
              <div className="flex items-center gap-3">
                <span className="font-mono text-[11px] tracking-[0.2em] text-accent-300">
                  STEP {s.n}
                </span>
              </div>
              <h3 className="mt-3 text-[20px] font-semibold tracking-tight text-chalk-50">
                {s.title}
              </h3>
              <p className="mt-3 text-[14px] leading-relaxed text-chalk-400">
                {s.body}
              </p>
              <ul className="mt-5 space-y-2">
                {s.bullets.map((b) => (
                  <li key={b} className="flex items-start gap-2.5 text-[12.5px] font-mono text-chalk-300">
                    <span className="mt-1.5 inline-block h-1 w-1 rounded-full bg-accent-400" />
                    {b}
                  </li>
                ))}
              </ul>
            </motion.article>
          ))}
        </div>
      </div>
    </section>
  );
}
