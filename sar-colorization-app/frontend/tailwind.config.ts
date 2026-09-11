import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#05070A",
        surface: "#0B1016",
        card: "#11161D",
        accent: "#40D9FF",
      },
      boxShadow: { glow: "0 18px 60px rgba(49, 120, 255, .11)" },
    },
  },
  plugins: [],
};

export default config;
