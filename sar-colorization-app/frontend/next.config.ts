import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Keep Next's deployment trace scoped to this standalone frontend package.
  outputFileTracingRoot: __dirname,
};

export default nextConfig;
