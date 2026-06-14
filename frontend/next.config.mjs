/** @type {import('next').NextConfig} */
const nextConfig = {
    // Add rewrites to proxy API requests to the Python backend
    async rewrites() {
      return [
        {
          source: '/api/:path*',
          destination: 'http://localhost:8000/api/:path*' // Proxy to Backend
        }
      ]
    }
};

export default nextConfig;
