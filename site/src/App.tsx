import Nav from "./components/Nav";
import Hero from "./sections/Hero";
import Metrics from "./sections/Metrics";
import HowItWorks from "./sections/HowItWorks";
import Features from "./sections/Features";
import Architecture from "./sections/Architecture";
import CodePreview from "./sections/CodePreview";
import UseCases from "./sections/UseCases";
import CTA from "./sections/CTA";
import Footer from "./components/Footer";

export default function App() {
  return (
    <div className="relative min-h-screen bg-ink-950 text-chalk-100 selection:bg-accent/40">
      <BackgroundFX />
      <Nav />
      <main>
        <Hero />
        <Metrics />
        <HowItWorks />
        <Features />
        <Architecture />
        <CodePreview />
        <UseCases />
        <CTA />
      </main>
      <Footer />
    </div>
  );
}

function BackgroundFX() {
  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      <div className="absolute inset-x-0 top-0 h-[120vh] bg-grid-fade" />
      <div className="absolute left-1/2 top-[-20%] h-[80vh] w-[120vw] -translate-x-1/2 rounded-[50%] bg-[radial-gradient(ellipse_at_center,rgba(124,92,255,0.22),transparent_60%)] blur-3xl" />
      <div className="absolute right-[-10%] top-[40%] h-[40vh] w-[60vw] rounded-[50%] bg-[radial-gradient(ellipse_at_center,rgba(94,226,255,0.12),transparent_60%)] blur-3xl" />
      <div className="absolute inset-0 bg-noise opacity-[0.35] mix-blend-overlay" />
    </div>
  );
}
