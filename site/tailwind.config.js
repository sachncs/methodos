/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          "SF Pro Display",
          "SF Pro Text",
          "Inter",
          "system-ui",
          "Segoe UI",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: [
          "SF Mono",
          "JetBrains Mono",
          "Menlo",
          "Monaco",
          "Consolas",
          "Liberation Mono",
          "Courier New",
          "monospace",
        ],
      },
      colors: {
        ink: {
          950: "#05060A",
          900: "#0A0B10",
          800: "#0F1018",
          700: "#15161F",
          600: "#1B1D29",
          500: "#222435",
        },
        chalk: {
          50: "#FFFFFF",
          100: "#F5F6FA",
          200: "#E6E8F0",
          300: "#C7CADB",
          400: "#9BA0BD",
          500: "#6E7395",
          600: "#4A4F6E",
          700: "#2F334A",
        },
        accent: {
          DEFAULT: "#7C5CFF",
          50: "#F2EEFF",
          100: "#E2D9FF",
          200: "#C5B3FF",
          300: "#A88EFF",
          400: "#8E6FFF",
          500: "#7C5CFF",
          600: "#5E3FE6",
          700: "#462BB8",
          800: "#311E80",
          900: "#1E124E",
        },
        electric: {
          DEFAULT: "#5EE2FF",
        },
      },
      letterSpacing: {
        tightest: "-0.045em",
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(255,255,255,0.04), 0 30px 80px -20px rgba(124,92,255,0.35)",
        ring: "0 0 0 1px rgba(255,255,255,0.06)",
      },
      backgroundImage: {
        "grid-fade":
          "radial-gradient(ellipse at top, rgba(124,92,255,0.18), transparent 60%)",
        "noise":
          "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/%3E%3CfeColorMatrix values='0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0.06 0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")",
      },
      animation: {
        "float-slow": "float 14s ease-in-out infinite",
        "shimmer": "shimmer 8s linear infinite",
      },
      keyframes: {
        float: {
          "0%,100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-12px)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
    },
  },
  plugins: [],
};
