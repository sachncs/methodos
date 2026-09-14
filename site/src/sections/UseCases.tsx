import { motion, useInView } from "framer-motion";
import { useRef } from "react";

const cases = [
  {
    tag: "Research agents",
    title: "Multi-hop QA with HotpotQA-grade evaluations.",
    body:
      "Pair the adapter with your agent and run paired with-PG vs without-PG benchmarks. Cached guidance keeps hot-paths cheap; refinements move the curve on every batch.",
  },
  {
    tag: "Tool-using assistants",
    title: "Stable behavior as the toolset grows.",
    body:
      "Add a tool, the graph learns where it belongs and where it doesn't. Edits are validated for cycles and reachability before they ship.",
  },
  {
    tag: "Production copilots",
    title: "Auditable decisions, not vibes.",
    body:
      "Every guidance paragraph is sourced from a specific subgraph keyed by (graph, last_action, last_observation). Inspectable. Reproducible. Reversible.",
  },
];

export default function UseCases() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-100px" });

  return (
    <section id="use-cases" ref={ref} className="relative py-24 md:py-32">
      <div className="container-x">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
          className="max-w-2xl"
        >
          <span className="pill">Use cases</span>
          <h2 className="mt-5 text-[32px] md:text-[48px] font-semibold tracking-tightest leading-[1.05] gradient-text">
            One primitive. Many surfaces.
          </h2>
        </motion.div>

        <div className="mt-14 grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-5">
          {cases.map((c, i) => (
            <motion.article
              key={c.tag}
              initial={{ opacity: 0, y: 24 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.7, delay: 0.05 + i * 0.08, ease: [0.22, 1, 0.36, 1] }}
              className="relative overflow-hidden rounded-2xl border border-white/[0.06] bg-ink-900/40 p-7 transition-colors hover:border-white/[0.12]"
            >
              <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/15 to-transparent" />
              <div className="text-[11px] uppercase tracking-[0.2em] text-chalk-500">
                {c.tag}
              </div>
              <h3 className="mt-4 text-[18px] font-semibold tracking-tight text-chalk-50 leading-snug">
                {c.title}
              </h3>
              <p className="mt-3 text-[13.5px] leading-relaxed text-chalk-400">
                {c.body}
              </p>
            </motion.article>
          ))}
        </div>
      </div>
    </section>
  );
}
