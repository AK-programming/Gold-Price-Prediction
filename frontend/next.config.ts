import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:7860/api/:path*",
      },
    ];
  },
  allowedDevOrigins: ["192.168.1.11"],
};

export default nextConfig;
