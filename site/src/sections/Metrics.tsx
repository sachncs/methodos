import { motion, useInView } from "framer-motion";
import { useRef } from "react";

const stats = [
  { value: "≥ 95%", label: "Test coverage" },
  { value: "mypy --strict", label: "Type checked" },
  { value: "2-hop", label: "Default neighborhood" },
  { value: "0", label: "Lazy imports" },
];

export default function Metrics() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <section ref={ref} className="relative py-16 md:py-24">
      <div className="container-x">
        <div className="grid grid-cols-2 gap-y-10 md:grid-cols-4 md:gap-y-0">
          {stats.map((s, i) => (
            <motion.div
              key={s.label}
              initial={{ opacity: 0, y: 16 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.6, delay: i * 0.08, ease: [0.22, 1, 0.36, 1] }}
              className="flex flex-col items-center md:items-start md:border-l md:border-white/[0.06] md:pl-8 first:md:border-l-0 first:md:pl-0"
            >
              <div className="text-[34px] md:text-[44px] font-semibold tracking-tightest gradient-text leading-none">
                {s.value}
              </div>
              <div className="mt-2 text-[12px] uppercase tracking-[0.18em] text-chalk-500">
                {s.label}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
