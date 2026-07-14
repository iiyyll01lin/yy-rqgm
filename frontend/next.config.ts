import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow accessing the dev server from LAN IPs (e.g. when the preview browser
  // runs on a different host than the dev server). Dev-only; ignored in prod.
  allowedDevOrigins: ["10.96.30.46"],
};

export default nextConfig;
