import type { NextConfig } from "next";

// In production (Vercel) set NEXT_PUBLIC_API_URL to your deployed backend,
// e.g. https://<you>-<space-name>.hf.space — no trailing slash.
// Locally it falls back to the FastAPI dev server on 127.0.0.1:7860.
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:7860";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${API_URL}/api/:path*`,
      },
    ];
  },
  allowedDevOrigins: ["192.168.1.11"],
};

export default nextConfig;
